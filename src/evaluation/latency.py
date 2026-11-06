"""End-to-end latency benchmark.

    CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.latency --dataset ncbi_disease

What is timed, per mention, from an inference-ready mention with its context to a
final linked concept:

    retrieval           encode the mention, score it against the KB embeddings,
                        aggregate to concepts, take the top-k
    routing features    the routing input the policy needs, built from the
                        retrieval scores this mention just produced -- the eight
                        CURRENT8 features for a learned router, the top-1/top-2
                        margin for Margin, nothing for the endpoints
    router inference    the routing score for the policy under test
    routing decision    the rerank/keep branch
    conditional rerank  the cross-encoder over the first K_high candidates, only
                        for the mentions the policy selected
    fusion              standardise and combine the two scores, take the argmax

The routing-feature stage is a real cost of deployment: a served mention arrives
with no precomputed feature row, so the router has to build one from the
retrieval scores before it can score anything.  This benchmark therefore builds
them inside the timed region (``--routing-feature-mode live``, the default).  The
``cached`` mode reads a precomputed feature row instead and exists only as a
diagnostic control; it does not reproduce the reported latency.

Protocol, fixed before measuring:

    * batch = 1 (online serving), not batch-amortised.  Batched and batch=1
      numbers are not comparable, not even as speedups, so they are never mixed.
    * ``torch.cuda.synchronize()`` around every timed stage.
    * 3 warm-up sweeps, then 9 timed sweeps.  A sweep runs every cell once, in a
      fixed order, so thermal and clock drift lands on all cells alike instead of
      accumulating against whichever cell was timed last.  The reported value is
      the median over the 9 timed sweeps.
    * a fixed random subset of the split, drawn with the configured sample seed;
      every policy and rate is timed on the same subset in the same process.
    * model and KB-index loading are offline setup and excluded; they are timed
      separately and reported.
    * both endpoints (rate 0 and 1) are re-timed **in the same process** as the
      intermediate rates, because absolute milliseconds shift between processes
      and thermal states.
    * the reranking decisions are the frozen policy selections; see
      ``sel_source`` in the output and section 11 of the README.

Absolute milliseconds are hardware- and run-dependent.  Compare only within one
dataset, against the full-reranking cell of the same process, via the reported
speedup.  Accuracy in this output is the whole-split Acc@1 under the same
selection, not the accuracy of the timed subset.

Writes ``latency/e2e_latency.csv``, ``latency/stage_breakdown.csv``,
``latency/repetitions.csv`` and ``latency/protocol.json``.
"""
from __future__ import annotations

import argparse
import time

STAGES = ('retrieval', 'routing_features', 'router_inference', 'routing_decision',
          'conditional_reranking', 'fusion')

# What each policy makes the two routing stages do.  ``current8`` resolves to a
# live construction or to a cached row according to --routing-feature-mode.
FEATURE_PATH = {'retriever': 'none', 'full': 'none', 'random': 'none',
                'margin': 'margin_live', 'difficulty': 'current8',
                'opportunity': 'current8', 'benefit': 'current8',
                'arr': 'none', 'arter': 'none'}
ROUTER_PATH = {'retriever': None, 'full': None, 'random': 'random',
               'margin': 'margin', 'difficulty': 'router', 'opportunity': 'router',
               'benefit': 'router', 'arr': 'arr', 'arter': 'arter'}


class Stopwatch:
    """Accumulates synchronised wall time per stage."""

    def __init__(self, torch, sync: bool):
        self.torch, self.sync = torch, sync
        self.t = {k: 0.0 for k in STAGES}

    def __call__(self, name):
        return _Segment(self, name)


class _Segment:
    def __init__(self, sw, name):
        self.sw, self.name = sw, name

    def __enter__(self):
        if self.sw.sync:
            self.sw.torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        if self.sw.sync:
            self.sw.torch.cuda.synchronize()
        self.sw.t[self.name] += time.perf_counter() - self.t0


def cv_pct(v) -> float:
    import numpy as np
    v = np.asarray(v, float)
    return float(100.0 * v.std() / max(v.mean(), 1e-12))


def main(argv=None) -> int:
    from ..config import add_common_arguments, load_config
    from ..utils.efficiency import Clock, eff_row, write_efficiency
    from ..utils.io import jdump, jload, write_table
    from ..utils.threads import limit_threads

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(ap)
    ap.add_argument('--split', default='test')
    ap.add_argument('--policies', nargs='+',
                    default=['random', 'margin', 'difficulty', 'opportunity'],
                    help='routed policies to time (arr and arter need their routers)')
    ap.add_argument('--rates', nargs='+', type=float, default=[0.10, 0.30, 0.50],
                    help='intermediate reranking rates; the two endpoints are always added')
    ap.add_argument('--n-timed', type=int, default=500,
                    help='mentions in the timed subset; the reported runs used the '
                         'per-dataset values in README section 11')
    ap.add_argument('--routing-feature-mode', dest='routing_feature_mode',
                    default='live', choices=['live', 'cached'],
                    help='live (default) builds CURRENT8 from the retrieval scores of '
                         'the mention being served; cached reads a precomputed feature '
                         'row and is a diagnostic control that does not reproduce the '
                         'reported latency')
    ap.add_argument('--warmup', type=int, default=None)
    ap.add_argument('--reps', type=int, default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args(argv)
    limit_threads(args.threads)

    import numpy as np
    import pandas as pd
    import torch

    from ..data.ontology import kb_arrays, load_concept_representation
    from ..reranking import cross_encoder as CEmod
    from ..reranking import fusion as FU
    from ..reranking.select_depth import load_scored
    from ..retrieval.encoder import encode_dense, load_encoder
    from ..routing import classifiers as CL
    from ..routing.budget import apply_selection, select_top_q
    from ..retrieval.index import dedup_to_concepts
    from ..routing.features import apply_standardiser, feature_matrix
    from . import policies as PO

    cfg = load_config(args.dataset, args, args.method_config)
    Lt = cfg.method['latency']
    Rt = cfg.method['retriever']
    warmup = int(args.warmup if args.warmup is not None else Lt['warmup'])
    reps = int(args.reps if args.reps is not None else Lt['timed_reps'])
    live_features = args.routing_feature_mode == 'live'
    device = args.device
    sync = device.startswith('cuda')
    clock = Clock()
    if sync:
        torch.backends.cuda.matmul.allow_tf32 = True

    t = pd.read_parquet(cfg.routing_table_path(args.split))
    K = int(t.K_high.iloc[0])
    fz = jload(cfg.fusion_path())
    cal, alpha = fz['calibration'], float(fz['selected_alpha'])
    cand, ret, ce, valid, _ = load_scored(cfg, args.split, -1)
    if not np.array_equal(cand.mention_id.astype(str).to_numpy(),
                          t.mention_id.astype(str).to_numpy()):
        raise SystemExit(f'{cfg.key}: candidate cache and routing table are not row-aligned')
    n_all = len(t)
    # Diagnostic control only: the live path never reads this.
    X8_cached = t[[c for c in cfg.method['features']['list']]].to_numpy(float)
    mentions = cand.mention_text.astype(str).to_numpy()
    q_all = pd.read_parquet(cfg.queries_path(args.split))
    qmap = dict(zip(q_all.mention_id.astype(str), q_all.query_text.astype(str)))
    ids_ordered = t.mention_id.astype(str)
    queries = np.array([qmap[str(i)] for i in ids_ordered], dtype=object)
    if 'context_text' in q_all.columns:
        cmap = dict(zip(q_all.mention_id.astype(str), q_all.context_text.astype(str)))
        contexts = np.array([cmap.get(str(i), '') for i in ids_ordered], dtype=object)
    else:
        contexts = queries
    repr_map = load_concept_representation(cfg.concept_repr_path())
    from ..retrieval.index import pool_concept_ids
    pool = pool_concept_ids(cand, int(Rt['k_pool']))

    rng = np.random.default_rng(int(Lt['sample_seed']))
    idx = np.sort(rng.choice(n_all, size=min(int(args.n_timed), n_all), replace=False))
    nt = len(idx)

    # ---------------------------------------------------------------- offline setup
    setup = Clock()
    names, name2c, _ = kb_arrays(cfg.kb_path())
    ret_model, ret_tok = load_encoder(Rt['model'], device)
    E, _ = encode_dense(ret_model, ret_tok, names, device, max_len=int(Rt['max_len']),
                        batch=int(Rt['encode_batch']), desc='kb-index')
    if Rt['normalize']:
        E = torch.nn.functional.normalize(E, dim=1)
    name2c_np = np.asarray(name2c).astype(np.int64)
    cinv = torch.from_numpy(name2c_np).to(device)
    n_concept = int(name2c_np.max()) + 1
    # The timed retrieval stage runs the aggregation this dataset actually uses.
    agg = Rt['aggregation']
    if agg not in ('concept_max_streaming', 'alias_topn_dedup'):
        raise SystemExit(f'unknown retriever.aggregation for timing: {agg}')
    topn = int(Rt['name_topn'])
    k_pool = int(Rt['k_pool'])
    ce_model, ce_tok = CEmod.load_checkpoint(cfg.ce_checkpoint(-1), device)
    routers, zstd = {}, {}
    for pol in args.policies:
        tgt = PO.SUPERVISION.get(pol)
        if tgt:
            routers[pol] = CL.load(cfg.method['router']['family'],
                                   cfg.router_model_path(tgt))
            zstd[pol] = np.load(cfg.router_scores_path(tgt), allow_pickle=True)
    arr_pack = at_pack = None
    if 'arr' in args.policies:
        arr_pack = PO.load_arr_router(cfg, device)
    if 'arter' in args.policies:
        at_pack = PO.load_arter_encoder(cfg, device)
    scores = PO.available(cfg, t, args.split)
    setup_s = setup.s()
    if sync:
        torch.cuda.reset_peak_memory_stats()

    Bm = cfg.method['baselines']['random']
    m_random = int(round(float(cfg.method['budget']['q_representative']) * n_all))
    rrng = np.random.default_rng(int(Bm['seed']))
    sel_random = np.zeros(n_all, bool)
    if m_random:
        sel_random[rrng.choice(n_all, size=m_random, replace=False)] = True

    def cell_of(pol, q, sel, src):
        feat = FEATURE_PATH[pol]
        if feat == 'current8':
            feat = 'current8_live' if live_features else 'current8_cached'
        key = ('retriever_only' if pol == 'retriever' else
               'full_reranking' if pol == 'full' else
               f'{pol}@q{int(round(100 * q))}')
        if pol in ('difficulty', 'opportunity', 'benefit') and not live_features:
            key += '_cachedfeat'
        return dict(key=key, policy=pol, rate=float(q), sel=sel, feat=feat,
                    router=ROUTER_PATH[pol], sel_source=src,
                    label=(PO.POLICY_LABEL[pol] if pol in ('retriever', 'full') else
                           f'{PO.POLICY_LABEL[pol]} @ rate={int(round(100 * q))}%'))

    cells = [cell_of('retriever', 0.0, np.zeros(n_all, bool), 'endpoint: nothing reranked'),
             cell_of('full', 1.0, np.ones(n_all, bool), 'endpoint: everything reranked')]
    for pol in args.policies:
        if pol == 'random':
            cells.append(cell_of(
                'random', float(cfg.method['budget']['q_representative']), sel_random,
                f'first draw of the {int(Bm["replicates"])}-replicate Random protocol '
                f'(seed {int(Bm["seed"])}); the reported Random accuracy is the mean '
                f'over all replicates'))
            continue
        if pol not in scores:
            print(f'  [skip] {pol}: no routing score on disk', flush=True)
            continue
        for q in args.rates:
            cells.append(cell_of(pol, q, select_top_q(scores[pol], q, n_all),
                                 'frozen policy score, top-q'))
    print(f'[{cfg.key}] aliases {len(names):,} · concepts {n_concept:,} · '
          f'timed {nt}/{n_all} mentions · K_high {K} · cells {len(cells)} · '
          f'features {args.routing_feature_mode} · '
          f'warmup {warmup} + {reps} timed sweeps (round-robin)', flush=True)

    live_rng = np.random.default_rng(int(Bm['seed']))

    def one_pass(cell):
        sw = Stopwatch(torch, sync)
        pol, feat, router = cell['policy'], cell['feat'], cell['router']
        z = zstd.get(pol)
        for i in idx:
            with sw('retrieval'):
                e = ret_tok([mentions[i]], truncation=True, max_length=int(Rt['max_len']),
                            padding='max_length', return_tensors='pt').to(device)
                with torch.no_grad():
                    h = ret_model(**e).last_hidden_state[:, 0].float()
                if Rt['normalize']:
                    h = torch.nn.functional.normalize(h, dim=1)
                sim = E @ h[0]
                if agg == 'concept_max_streaming':
                    cs = torch.full((n_concept,), -1e30, device=device)
                    cs.scatter_reduce_(0, cinv, sim, reduce='amax')
                    tv, _ti = torch.topk(cs, k_pool)
                else:
                    # alias_topn_dedup: a global alias ranking folded to concepts,
                    # which is a host-side pass over the top-N aliases
                    ts, ti = torch.topk(sim, min(topn, sim.numel()))
                    _, _, csn, _ = dedup_to_concepts(
                        ti.to('cpu').numpy()[None, :], ts.to('cpu').numpy()[None, :],
                        name2c_np, k_pool)
                    tv = torch.from_numpy(
                        np.nan_to_num(csn[0].astype(np.float64), nan=-np.inf)).to(device)
            x = None
            with sw('routing_features'):
                if feat == 'current8_live':
                    # what a served mention actually costs: build CURRENT8 from the
                    # retrieval scores this mention just produced
                    r5 = tv[:5].to('cpu').numpy().astype(np.float64)[None, :]
                    x, _ = feature_matrix(r5)
                elif feat == 'margin_live':
                    r2v = tv[:2].to('cpu').numpy().astype(np.float64)
                    x = float(r2v[0] - r2v[1])
                elif feat == 'current8_cached':
                    x = X8_cached[i:i + 1]          # diagnostic control
            with sw('router_inference'):
                if router == 'router':
                    routers[pol].predict_proba(apply_standardiser(x, z['mu'], z['sd']))[:, 1]
                elif router == 'margin':
                    -x
                elif router == 'random':
                    float(live_rng.random()) < cell['rate']
                elif router == 'arr':
                    ea = arr_pack['tok']([queries[i]], truncation=True,
                                         max_length=int(arr_pack['max_len']),
                                         padding='max_length',
                                         return_tensors='pt').to(device)
                    with torch.no_grad():
                        arr_pack['model'](**ea).logits
                elif router == 'arter':
                    A = cfg.method['baselines']['arter']
                    ec = at_pack['tok']([contexts[i]], truncation=True,
                                        max_length=int(A['ctx_max_len']),
                                        padding='max_length', return_tensors='pt').to(device)
                    em = at_pack['tok']([mentions[i]], truncation=True,
                                        max_length=int(A['mention_max_len']),
                                        padding='max_length', return_tensors='pt').to(device)
                    ents = [repr_map.get(str(c), str(c))
                            for c in pool[i, :int(A['top_m'])] if c]
                    ee = at_pack['tok'](ents or [''], truncation=True,
                                        max_length=int(A['entity_max_len']),
                                        padding='max_length', return_tensors='pt').to(device)
                    with torch.no_grad():
                        at_pack['model'](**ec).last_hidden_state[:, 0]
                        at_pack['model'](**em).last_hidden_state[:, 0]
                        at_pack['model'](**ee).last_hidden_state[:, 0]
            with sw('routing_decision'):
                do = bool(cell['sel'][i])
            if do:
                with sw('conditional_reranking'):
                    pairs = [repr_map.get(str(c), str(c)) for c in pool[i, :K] if c]
                    ec = ce_tok([queries[i]] * len(pairs), pairs,
                                truncation=cfg.method['reranker']['truncation'],
                                max_length=int(cfg.method['context']['max_pair_tokens']),
                                padding='max_length', return_tensors='pt').to(device)
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16,
                                                         enabled=sync):
                        ce_model(**ec).logits[:, 1].float()
                with sw('fusion'):
                    FU.fuse(ret[i:i + 1, :K], ce[i:i + 1, :K], valid[i:i + 1, :K],
                            cal, alpha)
        return dict(sw.t)

    # ---------------------------------------------------------------- timed sweeps
    # One sweep = every cell once.  Warm-up sweeps first, then the timed ones.
    t_start = time.time()
    passes = {c['key']: [] for c in cells}
    for sweep in range(warmup + reps):
        for c in cells:
            passes[c['key']].append(one_pass(c))
        phase = 'warmup' if sweep < warmup else 'timed'
        print(f'  sweep {sweep + 1}/{warmup + reps} ({phase}) · '
              f'{(time.time() - t_start) / 60:.1f} min', flush=True)

    c1 = t.correct_K1.to_numpy().astype(bool)
    ckh = t.correct_Khigh.to_numpy().astype(bool)
    rows, srows, rrows = [], [], []
    for cell in cells:
        R = passes[cell['key']][warmup:]
        tot = np.array([sum(x[s] for s in STAGES) for x in R]) * 1000.0 / nt
        med = float(np.median(tot))
        rmed = {s: float(np.median([x[s] for x in R]) * 1000.0 / nt) for s in STAGES}
        correct = apply_selection(cell['sel'], c1, ckh)
        nr = int(cell['sel'].sum())
        for j, v in enumerate(tot):
            rrows.append(dict(dataset=cfg.key, cell=cell['key'], repetition=j,
                              total_ms_per_mention=round(float(v), 5)))
        rows.append(dict(
            dataset=cfg.label, dataset_key=cfg.key, cell=cell['key'],
            policy=cell['policy'], method=PO.POLICY_LABEL[cell['policy']],
            label=cell['label'], reranking_rate=round(cell['rate'], 6),
            reranking_rate_pct=round(100 * cell['rate'], 4),
            feature_path=cell['feat'], router_path=str(cell['router']),
            sel_source=cell['sel_source'],
            n_timed=nt, n_split=n_all, number_reranked=nr,
            actual_rate=round(nr / n_all, 6),
            acc_at_1_full_split=round(float(correct.mean()), 6),
            ce_pairs_per_mention=round((nr / n_all) * K, 4),
            median_ms_per_mention=round(med, 4), mean_ms=round(float(tot.mean()), 4),
            p25_ms=round(float(np.percentile(tot, 25)), 4),
            p75_ms=round(float(np.percentile(tot, 75)), 4),
            min_ms=round(float(tot.min()), 4), max_ms=round(float(tot.max()), 4),
            cv_pct=round(cv_pct(tot), 3), warmup=warmup, timed_reps=reps,
            retrieval_ms=round(rmed['retrieval'], 4),
            routing_features_ms=round(rmed['routing_features'], 5),
            router_inference_ms=round(rmed['router_inference'], 5),
            router_overhead_ms=round(rmed['routing_features'] + rmed['router_inference'], 4),
            routing_decision_ms=round(rmed['routing_decision'], 5),
            conditional_reranking_ms=round(rmed['conditional_reranking'], 4),
            fusion_ms=round(rmed['fusion'], 5),
            e2e_throughput_mentions_per_s=round(1000.0 / max(med, 1e-9), 3),
            hardware=(torch.cuda.get_device_name() if sync else 'CPU'),
            regime=Lt['regime'], candidate_depth_K=K,
            kb_aliases=len(names), kb_concepts=n_concept,
            gpu_peak_GiB=(round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
                          if sync else None)))
        for s in STAGES:
            srows.append(dict(dataset=cfg.key, cell=cell['key'], stage=s,
                              median_ms_per_mention=round(rmed[s], 6)))
        print(f"[{cell['key']:34s}] rerank {100 * nr / n_all:6.2f}% · "
              f"median {med:9.3f} ms (retrieval {rmed['retrieval']:7.3f} · "
              f"rerank {rmed['conditional_reranking']:7.3f} · "
              f"features {rmed['routing_features']:7.4f} · "
              f"router {rmed['router_inference']:7.4f}) · "
              f"CV {cv_pct(tot):5.2f}%", flush=True)

    L = pd.DataFrame(rows)
    full = float(L[L.cell == 'full_reranking'].median_ms_per_mention.iloc[0])
    L['speedup_vs_full'] = (full / L.median_ms_per_mention).round(4)
    L['latency_saving_pct_vs_full'] = (100 * (1 - L.median_ms_per_mention / full)).round(3)
    write_table(L, cfg.out_path('latency', 'e2e_latency.csv'))
    write_table(pd.DataFrame(srows), cfg.out_path('latency', 'stage_breakdown.csv'))
    write_table(pd.DataFrame(rrows), cfg.out_path('latency', 'repetitions.csv'))
    jdump(dict(dataset=cfg.key, split=args.split, protocol=Lt, stages=list(STAGES),
               warmup=warmup, timed_reps=reps, n_timed=nt, n_split=n_all,
               sweep_order='pass-wise round-robin: every cell once per sweep',
               routing_feature_mode=args.routing_feature_mode,
               routing_feature_note=(
                   'live: CURRENT8 is built inside the timed region from the retrieval '
                   'scores of the mention being served, which is what the reported '
                   'latency measures. cached: a precomputed feature row is read instead '
                   '-- diagnostic only, does not reproduce the reported latency.'),
               selection_source=(
                   'frozen policy selections; the router is executed and timed but its '
                   'output does not change which mentions are reranked, so the accuracy '
                   'in this file is the frozen accuracy of the same policy'),
               sample_seed=int(Lt['sample_seed']), candidate_depth_K=K,
               random_selection_seed=int(Bm['seed']),
               offline_setup_seconds=setup_s,
               offline_setup_excluded=('retriever and cross-encoder loading, KB embedding, '
                                       'router loading'),
               hardware=(torch.cuda.get_device_name() if sync else 'CPU'),
               torch_version=torch.__version__,
               cuda_version=(torch.version.cuda if sync else None),
               blas_threads=args.threads,
               caveat=('absolute milliseconds are hardware- and run-dependent; compare only '
                       'within one dataset against the full-reranking cell of the same '
                       'process, and never mix batch=1 with batch-amortised numbers')),
          cfg.out_path('latency', 'protocol.json'))
    write_efficiency([eff_row(
        'latency', clock, dataset=cfg.key,
        device=('CUDA' if sync else 'CPU'),
        gpu=(torch.cuda.get_device_name() if sync else None),
        offline_setup_seconds=setup_s,
        gpu_peak_GiB=(round(torch.cuda.max_memory_allocated() / 2 ** 30, 3) if sync
                      else None),
        neural_inference_calls=int(nt * (warmup + reps) * len(cells)))],
        cfg.out_path('efficiency', 'latency.csv'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
