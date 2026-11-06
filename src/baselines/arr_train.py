"""ARR router training and scoring (our adaptation).

    CUDA_VISIBLE_DEVICES=0 python -m src.baselines.arr_train --dataset ncbi_disease

Builds the graded-utility route labels on the training split, trains the BERT
sequence-classifier router with the original recipe, and writes class-1
probabilities for dev and test.  The checkpoint is selected on the 25% validation
split of the down-sampled training set; dev and test are not used for selection.

Writes ``baselines/arr/scores_{split}.parquet`` and ``baselines/arr/training.json``,
plus a label-agreement table against our own targets (the original utility label
is not the same thing as HARD, OPP or BEN, and the overlap is reported rather
than assumed).
"""
from __future__ import annotations

import argparse
import os
import time

from ..config import add_common_arguments, load_config
from ..utils.efficiency import Clock, eff_row, model_size, write_efficiency
from ..utils.io import jdump, jload, write_table
from ..utils.threads import limit_threads


def measured_route_latencies(cfg):
    """(stop, rerank) ms per mention from this run's own efficiency artefacts.

    ``efficiency/retrieval.csv`` records query encoding plus search, and
    ``efficiency/score_full.csv`` records cross-encoder scoring of the cached pool,
    both as batch-amortised ``ms_per_mention``.  That is the regime the cost term
    is defined in, so the two are read directly.
    """
    import pandas as pd

    rp = cfg.out_path('efficiency', 'retrieval.csv')
    sp = cfg.out_path('efficiency', 'score_full.csv')
    if not (os.path.exists(rp) and os.path.exists(sp)):
        return None
    R = pd.read_csv(rp)
    S = pd.read_csv(sp)
    R = R[R.stage == 'retrieval']
    S = S[S.stage == 'score_candidates']
    if not len(R) or not len(S):
        return None
    stop = float(R.ms_per_mention.mean())
    rerank = float(S.ms_per_mention.mean())
    return stop, stop + rerank, f'measured in this run ({rp}, {sp})'


def route_latencies(cfg, args):
    """(stop, rerank) mean per-mention latency in ms for the utility cost term.

        --route-latency reported   the values the reported runs used (default)
        --route-latency measured   this run's own efficiency artefacts

    ``--retrieval-ms`` / ``--rerank-ms`` take precedence over both.  The cost term
    decides labels: the utility label flips for every mention whose ranking gain is
    below ``lambda_cost * (1 - stop / rerank)``.  Both routes are batch-amortised
    per-mention costs; the batch=1 figures from ``src.evaluation.latency`` are a
    different regime and are not used here.
    """
    if args.retrieval_ms is not None and args.rerank_ms is not None:
        return (float(args.retrieval_ms),
                float(args.retrieval_ms) + float(args.rerank_ms), 'command line')
    if args.route_latency == 'measured':
        m = measured_route_latencies(cfg)
        if m is None:
            raise SystemExit('--route-latency measured needs efficiency/retrieval.csv and '
                             'efficiency/score_full.csv; run src.retrieval.run and '
                             'src.reranking.score first, or use --route-latency reported')
        return m
    spec = cfg.dataset.get('arr_route_latency_ms')
    if spec:
        return (float(spec['retrieval']),
                float(spec['retrieval']) + float(spec['reranking']),
                f'dataset config ({spec.get("reranking_basis", "")})')
    m = measured_route_latencies(cfg)
    if m is not None:
        return m
    raise SystemExit('route latencies unknown: pass --retrieval-ms and --rerank-ms, '
                     'run src.retrieval.run and src.reranking.score for '
                     '--route-latency measured, or set arr_route_latency_ms in the '
                     'dataset config')


def build_labels(cfg, split, np, pd):
    from ..reranking import fusion as FU
    from ..reranking.select_depth import load_scored
    from . import arr as ARR

    fz = jload(cfg.fusion_path())
    k_high = int(jload(cfg.depth_path())['selected_K_high'])
    which = 'train' if split == 'train' else split
    cand, ret, ce, valid, gold = load_scored(cfg, which, -1)
    fused = FU.fuse(ret, ce, valid, fz['calibration'], float(fz['selected_alpha']))
    rr = ARR.route_ranks(fused, None, k_high, gold)
    e0 = ARR.graded_utility(rr['rank_stop'])
    e1 = ARR.graded_utility(rr['rank_ce'])
    return dict(ids=cand.mention_id.astype(str).to_numpy(), eff_stop=e0[0],
                eff_rerank=e1[0], ndcg_stop=e0[1], ndcg_rerank=e1[1],
                mrr_stop=e0[2], mrr_rerank=e1[2], K_high=k_high)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(ap)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--retrieval-ms', type=float, default=None,
                    help='mean retrieval latency per mention (ms) for the cost term')
    ap.add_argument('--rerank-ms', type=float, default=None,
                    help='mean reranking latency per mention (ms) for the cost term')
    ap.add_argument('--route-latency', dest='route_latency', default='reported',
                    choices=['reported', 'measured'],
                    help='cost-term latencies: the values the reported runs used '
                         '(default), or this run\'s own efficiency artefacts')
    ap.add_argument('--lambda-cost', type=float, default=None,
                    help='override baselines.arr.lambda_cost (default: the published 0.05)')
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args(argv)
    limit_threads(args.threads)

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, Subset, TensorDataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    from . import arr as ARR

    cfg = load_config(args.dataset, args, args.method_config)
    A = cfg.method['baselines']['arr']
    lam = float(args.lambda_cost if args.lambda_cost is not None else A['lambda_cost'])
    seed = int(A['seed'])
    device = args.device
    clock = Clock()
    lat0, lat1, lat_src = route_latencies(cfg, args)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = True

    tr = build_labels(cfg, 'train', np, pd)
    y = ARR.utility_labels(tr['eff_stop'], tr['eff_rerank'], lat0, lat1, lam).astype(np.int64)
    q = pd.read_parquet(cfg.queries_path('train'))
    qmap = dict(zip(q.mention_id.astype(str), q.query_text.astype(str)))
    txt = np.asarray([qmap.get(str(i), '') for i in tr['ids']], dtype=object)
    keep = np.array([len(t) > 0 for t in txt])
    y, txt, ids = y[keep], txt[keep], tr['ids'][keep]

    # Label agreement with our own targets: reported, not assumed.
    table = pd.read_parquet(cfg.routing_table_path('train'))
    ref = table.set_index(table.mention_id.astype(str)).reindex(pd.Index(ids))
    agree = []
    for name, col in (('HARD (Difficulty / ARTER target)', 'HARD'),
                      ('OPP (our Opportunity target)', 'OPP'),
                      ('BEN (realised reranker benefit)', 'BEN')):
        v = ref[col].fillna(False).to_numpy().astype(bool)
        inter = int(((y == 1) & v).sum())
        union = int(((y == 1) | v).sum())
        agree.append(dict(dataset=cfg.key, split='train', target=name,
                          arr_positive=int(y.sum()), target_positive=int(v.sum()),
                          intersection=inter,
                          jaccard=round(inter / max(union, 1), 4),
                          identical=bool((y.astype(bool) == v).all())))
    write_table(pd.DataFrame(agree), cfg.out_path('baselines', 'arr',
                                                  'label_agreement.csv'))

    sel = ARR.downsample_balanced(y, seed)
    nval = int(round(float(A['val_frac']) * len(sel)))
    vidx, tidx = sel[:nval], sel[nval:]
    tok = AutoTokenizer.from_pretrained(A['router_lm'])

    def encode(texts):
        e = tok(list(texts), truncation=True, max_length=int(A['max_len']),
                padding='max_length', return_tensors='pt')
        return e['input_ids'], e['attention_mask']

    ii, am = encode(txt)
    D = TensorDataset(ii, am, torch.tensor(y))
    dl = DataLoader(Subset(D, tidx.tolist()), batch_size=int(A['batch']), shuffle=True)
    vdl = DataLoader(Subset(D, vidx.tolist()), batch_size=128)

    model = AutoModelForSequenceClassification.from_pretrained(
        A['router_lm'], num_labels=int(A['num_labels'])).to(device)
    size = model_size(model)
    opt = torch.optim.AdamW(model.parameters(), lr=float(A['lr']))
    sch = get_linear_schedule_with_warmup(opt, 0, max(1, len(dl)) * int(A['epochs']))
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()

    log, best, best_ep, best_state = [], -1.0, -1, None
    t0 = time.time()
    for ep in range(int(A['epochs'])):
        model.train()
        tl, tn, te = 0.0, 0, time.time()
        for b in dl:
            x, msk, yy = [t.to(device, non_blocking=True) for t in b]
            with torch.autocast('cuda', dtype=torch.bfloat16,
                                enabled=device.startswith('cuda')):
                out = model(input_ids=x, attention_mask=msk, labels=yy)
            out.loss.backward()
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            tl += float(out.loss) * len(yy); tn += len(yy)
        model.eval()
        vc, vn, vl = 0, 0, 0.0
        with torch.no_grad():
            for b in vdl:
                x, msk, yy = [t.to(device) for t in b]
                with torch.autocast('cuda', dtype=torch.bfloat16,
                                    enabled=device.startswith('cuda')):
                    o = model(input_ids=x, attention_mask=msk, labels=yy)
                vl += float(o.loss) * len(yy)
                vc += int((o.logits.argmax(1) == yy).sum())
                vn += len(yy)
        acc = vc / max(vn, 1)
        log.append(dict(epoch=ep, train_loss=round(tl / max(tn, 1), 6),
                        val_loss=round(vl / max(vn, 1), 6), val_acc=round(acc, 6),
                        epoch_sec=round(time.time() - te, 2)))
        print(f'  ep {ep:2d} train_loss {tl / max(tn, 1):.4f} val_acc {acc:.4f}', flush=True)
        if acc > best:
            best, best_ep = acc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_s = time.time() - t0
    peak = (round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
            if device.startswith('cuda') else None)
    model.load_state_dict(best_state)
    outdir = cfg.model_path(f'arr_router_{cfg.tag}')
    os.makedirs(outdir, exist_ok=True)
    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)

    model.eval()
    inference = {}
    for split in ('dev', 'test'):
        t = pd.read_parquet(cfg.routing_table_path(split))
        qs = pd.read_parquet(cfg.queries_path(split))
        m = dict(zip(qs.mention_id.astype(str), qs.query_text.astype(str)))
        tx = [m.get(str(i), '') for i in t.mention_id.astype(str)]
        x, msk = encode(tx)
        p = np.empty(len(tx))
        with torch.no_grad():
            for i in range(0, min(256, len(tx)), 256):            # warm-up
                model(input_ids=x[i:i + 256].to(device),
                      attention_mask=msk[i:i + 256].to(device))
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            ti = time.time()
            for i in range(0, len(tx), 256):
                with torch.autocast('cuda', dtype=torch.bfloat16,
                                    enabled=device.startswith('cuda')):
                    o = model(input_ids=x[i:i + 256].to(device),
                              attention_mask=msk[i:i + 256].to(device))
                p[i:i + 256] = torch.softmax(o.logits.float(), 1)[:, 1].cpu().numpy()
            if device.startswith('cuda'):
                torch.cuda.synchronize()
        s = time.time() - ti
        inference[split] = dict(n=len(tx), sec=round(s, 3),
                                examples_per_sec=round(len(tx) / max(s, 1e-9), 1),
                                ms_per_mention=round(1000 * s / max(len(tx), 1), 4))
        write_table(pd.DataFrame(dict(mention_id=t.mention_id.astype(str), score=p,
                                      native_route=(p >= 0.5).astype(int))),
                    cfg.out_path('baselines', 'arr', f'scores_{split}.parquet'))

    write_table(pd.DataFrame(log), cfg.out_path('baselines', 'arr', 'training_log.csv'))
    jdump(dict(dataset=cfg.key, method='ARR (our adaptation)',
               fidelity='adaptation of a published method, not the original implementation',
               router_lm=A['router_lm'], model=size, hyperparameters=A,
               utility=A['utility'], lambda_cost=lam,
               lambda_cost_source='the value reported by the original work; not tuned here',
               route_latency_ms=dict(stop=lat0, rerank=round(lat1, 3),
                                     source=lat_src),
               n_train_all=int(len(y)), n_class1_all=int(y.sum()),
               class1_pct_all=round(100 * float(y.mean()), 3),
               n_after_downsample=int(len(sel)), n_fit=int(len(tidx)), n_val=int(len(vidx)),
               downsampling=A['training_data'], checkpoint_rule=A['checkpoint_rule'],
               selected_epoch=int(best_ep), best_val_acc=round(best, 6),
               train_wall_min=round(train_s / 60, 2), inference=inference,
               model_dir=outdir, gpu_peak_GiB=peak,
               TEST_USED_FOR_SELECTION=False, DEV_USED_FOR_SELECTION=False),
          cfg.out_path('baselines', 'arr', 'training.json'))
    write_efficiency([eff_row(
        'train_arr_router', clock, dataset=cfg.key,
        device=('CUDA' if device.startswith('cuda') else 'CPU'),
        gpu=(torch.cuda.get_device_name() if device.startswith('cuda') else None),
        model=A['router_lm'], model_parameters_M=size['n_parameters_M'],
        epochs=int(A['epochs']), n_fit=int(len(tidx)),
        train_wall_min=round(train_s / 60, 2), gpu_peak_GiB=peak,
        router_overhead_ms=inference['test']['ms_per_mention'])],
        cfg.out_path('efficiency', 'train_arr_router.csv'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
