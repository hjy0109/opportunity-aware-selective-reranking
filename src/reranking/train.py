"""Cross-encoder training entry point.

    CUDA_VISIBLE_DEVICES=0 python -m src.reranking.train --dataset ncbi_disease --fold -1
    CUDA_VISIBLE_DEVICES=0 python -m src.reranking.train --dataset ncbi_disease --fold 0

``--fold -1`` trains on the whole training split; this is the reranker used for
dev and test.  ``--fold {0,1,2}`` trains on the other two folds and is only
needed for the leakage-safe out-of-fold reranking behind the BEN diagnostic label.

The only checkpoint-selection signal is reranker-only Acc@1 on the train-internal
validation holdout.  Dev and test are never read here.  ``BOUNDARY_SELECTED`` in
the audit records whether the selected epoch equals the epoch cap without an
early stop, i.e. whether training length was still an active constraint.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time

from ..config import add_common_arguments, load_config
from ..utils.efficiency import Clock, eff_row, model_size, write_efficiency
from ..utils.io import jdump, sha256_file, write_table
from ..utils.threads import limit_threads


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(ap)
    ap.add_argument('--fold', type=int, required=True,
                    help='-1 = train on the whole training split; 0/1/2 = out-of-fold')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0,
                    help='truncate the fitting set (smoke test only)')
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args(argv)
    limit_threads(args.threads)

    import numpy as np
    import pandas as pd
    import torch
    from transformers import get_linear_schedule_with_warmup

    from ..data.ontology import load_concept_representation
    from ..data.splits import split_internal_val
    from . import cross_encoder as CE

    cfg = load_config(args.dataset, args, args.method_config)
    R = cfg.method['reranker']
    seed = int(cfg.method['seeds']['backbone'])
    k_pool = int(cfg.method['retriever']['k_pool'])
    max_len = int(cfg.method['context']['max_pair_tokens'])
    fold = args.fold
    fkey = 'full' if fold < 0 else f'f{fold}'
    ckpt = cfg.ce_checkpoint(fold)
    device = args.device
    clock = Clock()

    CE.setup_precision()
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)

    q = pd.read_parquet(cfg.queries_path('train'))
    queries = dict(zip(q.mention_id.astype(str), q.query_text.astype(str)))
    repr_map = load_concept_representation(cfg.concept_repr_path())
    cand = pd.read_parquet(cfg.candidates_path('train'))
    folds = pd.read_parquet(cfg.folds_path())
    if not np.array_equal(folds.mention_id.astype(str).to_numpy(),
                          cand.mention_id.astype(str).to_numpy()):
        raise SystemExit(f'{cfg.key}: fold table and candidate table are not row-aligned')
    fid = folds.fold.to_numpy()
    fit = cand if fold < 0 else cand[fid != fold].reset_index(drop=True)
    held = set() if fold < 0 else set(cand.mention_id.astype(str)[fid == fold])

    tr, iv, iv_rule, iv_meta = split_internal_val(
        fit, int(cfg.method['seeds']['internal_val']), R['internal_val'])
    if held and (held & set(tr.mention_id.astype(str))
                 or held & set(iv.mention_id.astype(str))):
        raise SystemExit(f'{cfg.key}/{fkey}: held-out fold mentions leaked into fitting')
    has_pos = (tr['gold_rank_in_pool'] > 0) | tr['gold_in_universe']
    n_drop = int((~has_pos).sum())
    tr = tr[has_pos].reset_index(drop=True)
    if args.limit:
        tr = tr.head(args.limit).reset_index(drop=True)
        iv = iv.head(min(len(iv), 120)).reset_index(drop=True)
    print(f'[{cfg.key}/{fkey}] fitting on {len(tr):,} mentions '
          f'(dropped without a positive: {n_drop}) · internal validation {len(iv):,} · '
          f'max_epochs {R["max_epochs"]} · device {device}', flush=True)

    model, tok = CE.build_model(R['backbone'], device)
    size = model_size(model)
    steps_per_epoch = math.ceil(len(tr) * 9 / int(R['batch_size']))
    total = steps_per_epoch * int(R['max_epochs'])
    opt = torch.optim.AdamW(model.parameters(), lr=float(R['lr']),
                            weight_decay=float(R['weight_decay']))
    sched = get_linear_schedule_with_warmup(opt, int(float(R['warmup_ratio']) * total), total)
    lossf = torch.nn.CrossEntropyLoss()
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()

    hist, best, best_state, bad, self_hard = [], -1.0, None, 0, None
    train_sec = mine_sec = eval_sec = 0.0
    first_epoch, mine_pairs, n_steps, early, n_nonfinite = None, 0, 0, False, 0
    t_all = time.time()
    for ep in range(1, int(R['max_epochs']) + 1):
        rng = np.random.default_rng(seed * 1000 + ep)
        if ep > 1 and (ep - 2) % int(R['mine_every']) == 0:
            t0 = time.time()
            self_hard, mp = CE.mine_self_hard(
                model, tok, tr, queries, repr_map, device, int(R['eval_batch']), max_len,
                rng, int(R['n_look']), int(R['n_keep']), R['truncation'])
            mine_sec += time.time() - t0
            mine_pairs += mp
        A, B, Y, n_inj, n_dup = CE.build_pairs(tr, queries, repr_map, rng, self_hard)
        if first_epoch is None:
            first_epoch = dict(n_pairs=int(len(A)), n_pos=int((Y == 1).sum()),
                               n_neg=int((Y == 0).sum()), n_injected_positive=int(n_inj),
                               n_negatives_identical_text_to_positive=int(n_dup))
            print(f'  [{cfg.key}/{fkey}] pairs {len(A):,} '
                  f'(positives {first_epoch["n_pos"]:,} · injected {n_inj:,}) · '
                  f'steps/epoch {steps_per_epoch:,}', flush=True)
        order = rng.permutation(len(A))
        model.train()
        t0, run, nb = time.time(), 0.0, 0
        for s in range(0, len(order), int(R['batch_size'])):
            idx = order[s:s + int(R['batch_size'])]
            enc = tok([A[i] for i in idx], [B[i] for i in idx], padding=True,
                      truncation=R['truncation'], max_length=max_len,
                      return_tensors='pt').to(device)
            with CE.amp(device.startswith('cuda')):
                loss = lossf(model(**enc).logits.float(),
                             torch.as_tensor(Y[idx], device=device))
            lv = float(loss.detach())
            if not np.isfinite(lv):
                n_nonfinite += 1
                raise SystemExit(f'[{cfg.key}/{fkey}] epoch {ep} step {nb}: '
                                 f'non-finite loss ({lv})')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(R['clip']))
            opt.step(); sched.step()
            run += lv; nb += 1
        ep_sec = time.time() - t0
        train_sec += ep_sec
        n_steps += nb
        del A, B
        t0 = time.time()
        acc, acc_in, acc_out, n_iv_pairs = CE.internal_val_metrics(
            model, tok, iv, queries, repr_map, device, int(R['eval_batch']), max_len,
            k_pool, R['truncation'])
        eval_sec += time.time() - t0
        sel = acc > best
        if sel:
            best, bad = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        rec = dict(epoch=ep, step_cumulative=n_steps, train_loss=round(run / max(nb, 1), 5),
                   n_pairs=int(len(Y)), lr_end=float(sched.get_last_lr()[0]),
                   iv_acc1=round(acc, 5), iv_acc1_given_gold_in_pool=round(acc_in, 5),
                   iv_acc1_gold_out_of_pool=round(acc_out, 5), selected=bool(sel),
                   patience_used=bad, train_sec=round(ep_sec, 1),
                   train_pairs_per_sec=round(len(Y) / max(ep_sec, 1e-9), 1),
                   iv_pairs=int(n_iv_pairs))
        hist.append(rec)
        print(f'  [{cfg.key}/{fkey}] {rec}', flush=True)
        if bad >= int(R['patience']):
            early = True
            print(f'  [{cfg.key}/{fkey}] early stop (patience {R["patience"]})', flush=True)
            break

    peak = (round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
            if device.startswith('cuda') else None)
    model.load_state_dict(best_state)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    write_table(pd.DataFrame(hist),
                cfg.out_path('ce_training', f'history_{fkey}.csv'))
    sel_ep = [h['epoch'] for h in hist if h['selected']][-1]
    boundary = bool(sel_ep == int(R['max_epochs']) and not early)
    wall = time.time() - t_all
    jdump(dict(
        dataset=cfg.key, fold=int(fold), fold_key=fkey, seed=seed, checkpoint=ckpt,
        candidate_representation=cfg.method['concept_representation']['name'],
        query_representation='target-centred document context',
        precision=R['precision'], truncation=R['truncation'],
        model=dict(backbone=R['backbone'], **size),
        checkpoint_sha256=(sha256_file(f'{ckpt}/model.safetensors')
                           if os.path.exists(f'{ckpt}/model.safetensors') else None),
        n_nonfinite_loss_steps=int(n_nonfinite),
        leakage_audit=dict(
            trained_on=('train split' if fold < 0 else f'train folds != {fold}'),
            n_held_out_fold_mentions=len(held),
            held_out_mentions_in_training=0, held_out_mentions_in_internal_val=0,
            group_key=cfg.dataset['grouping']['key'],
            DEV_USED_FOR_FITTING=False, TEST_READ=False),
        n_fit_mentions=int(len(tr)), n_dropped_no_positive=n_drop,
        n_internal_val_mentions=int(len(iv)),
        n_documents=int(fit.document_id.astype(str).nunique()),
        first_epoch_pairs=first_epoch, epochs_run=int(hist[-1]['epoch']),
        max_epochs=int(R['max_epochs']), max_epochs_frozen_before_training=True,
        patience=int(R['patience']), early_stopped=early,
        BOUNDARY_SELECTED=boundary,
        boundary_rule=('selected epoch == max_epochs without an early stop means '
                       'training length was still an active constraint'),
        steps_total=int(n_steps), steps_per_epoch=int(steps_per_epoch),
        selected_epoch=int(sel_ep), selected_iv_acc1=float(best),
        selection_metric=R['selection'], internal_val_rule=iv_rule,
        internal_val_meta=iv_meta, validation_trajectory=hist,
        hyperparameters=R, hyperparameter_sweep=False, architecture_search=False,
        train_wall_hours=round(wall / 3600, 4), train_sec=round(train_sec, 1),
        self_hard_mining_sec=round(mine_sec, 1), internal_val_eval_sec=round(eval_sec, 1),
        self_hard_mining_pairs=int(mine_pairs),
        gpu=(torch.cuda.get_device_name() if device.startswith('cuda') else None),
        gpu_peak_GiB=peak),
        cfg.out_path('ce_training', f'audit_{fkey}.json'))
    write_efficiency([eff_row(
        'train_cross_encoder', clock, dataset=cfg.key, fold_key=fkey,
        device=('CUDA' if device.startswith('cuda') else 'CPU'),
        gpu=(torch.cuda.get_device_name() if device.startswith('cuda') else None),
        model=R['backbone'], model_parameters_M=size['n_parameters_M'],
        n_mentions=int(len(tr)), n_pairs=int(first_epoch['n_pairs']),
        epochs_run=int(hist[-1]['epoch']), selected_epoch=int(sel_ep),
        train_wall_hours=round(wall / 3600, 4),
        train_pairs_per_sec=round(first_epoch['n_pairs'] * hist[-1]['epoch']
                                  / max(train_sec, 1e-9), 1),
        gpu_peak_GiB=peak, batch=int(R['batch_size']), max_len=max_len,
        precision=R['precision'], selected_iv_acc1=round(best, 5),
        BOUNDARY_SELECTED=boundary)],
        cfg.out_path('efficiency', f'train_cross_encoder_{fkey}.csv'))
    print(f'  -> {ckpt} · selected epoch {sel_ep}/{R["max_epochs"]} · '
          f'internal-val Acc@1 {best:.5f} · BOUNDARY_SELECTED={boundary} · '
          f'{wall / 3600:.3f} h', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
