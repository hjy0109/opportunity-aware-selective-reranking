"""ARTER feature construction, router fitting and scoring (our adaptation).

    CUDA_VISIBLE_DEVICES=0 python -m src.baselines.arter_run --dataset ncbi_disease

Encodes the candidate concept representations, the mention and its context;
builds the eight adapted features; fits the Random Forest on the training split
with the HARD target; picks the native operating point by dev Youden's J; and
writes dev/test routing scores.  The test split is used for evaluation only.

Writes ``baselines/arter/features.parquet``, ``scores_{split}.parquet``,
``fidelity.csv`` (what was preserved, substituted, degenerate or omitted) and
``run.json``.
"""
from __future__ import annotations

import argparse
import time

from ..config import add_common_arguments, load_config
from ..utils.efficiency import Clock, eff_row, model_size, write_efficiency
from ..utils.io import jdump, write_table
from ..utils.threads import limit_threads

FIDELITY = [
    ('Routing target', 'easy/hard: frozen base linker top-1 differs from gold',
     'HARD: retriever top-1 differs from gold', 'PRESERVED',
     'the two definitions coincide'),
    ('Router architecture', 'Random Forest', 'Random Forest, library defaults, fixed seed',
     'PRESERVED', 'the original reports no hyperparameters'),
    ('Threshold rule', 'validation Youden J', 'our dev Youden J', 'PRESERVED',
     'the native rule is kept rather than our own delta rule'),
    ('Entity representation', 'Wikipedia title + Wikidata description',
     'concept representation (alias pair), biomedical encoder', 'SUBSTITUTED',
     'our knowledge bases carry no definitions'),
    ('theta1 = cos(context, e_i)', 'present', 'recomputed with a separate encoder pass',
     'SUBSTITUTED',
     'the retriever encodes the mention surface form only (max_len 32)'),
    ('theta2 = cos(mention, e_i)', 'present',
     'same encoder as theta1/theta3; the frozen retrieval score is reported alongside',
     'SUBSTITUTED',
     'retrieval scores are on a per-dataset scale and would dominate the sum'),
    ('theta3 = max_{j != i} cos(e_i, e_j)', 'present', 'pairwise max within the top-10',
     'SUBSTITUTED', 'alias-based entity vectors change what the value means'),
    ('penalised score', '(theta1 + theta2 - theta3 + phi) / 3',
     '(theta1 + theta2 - theta3) / 2', 'SUBSTITUTED',
     'denominator matched to the two remaining terms'),
    ('margin', 'penalised top-1 minus top-2', 'identical', 'PRESERVED', ''),
    ('entropy', 'base-2 Shannon over a flattened candidate-score distribution',
     'base-2 Shannon over p_i = (s_i - min) / sum(s - min) on the top-10 retrieval scores',
     'SUBSTITUTED', 'the original normalisation is unstated; ours is fixed and stated'),
    ('sent_len', 'token length of the mention sentence', 'token length of our context',
     'PRESERVED', ''),
    ('n_cands', 'variable candidate count from the base linker', 'constant 64',
     'DEGENERATE', 'excluded from the router input and stated'),
    ('phi (LLM confidence)', 'an instruction-tuned LLM confidence term', 'omitted',
     'OMITTED', ''),
    ('Downstream action', 'targeted LLM reasoning',
     'our frozen cross-encoder reranking at K_high', 'SUBSTITUTED',
     'the action space of this work'),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(ap)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args(argv)
    limit_threads(args.threads)

    import numpy as np
    import pandas as pd
    import torch
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModel, AutoTokenizer

    from ..data.ontology import load_concept_representation
    from . import arter as AT

    cfg = load_config(args.dataset, args, args.method_config)
    A = cfg.method['baselines']['arter']
    device = args.device
    clock = Clock()
    top_m = int(A['top_m'])

    if device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(A['encoder'])
    model = AutoModel.from_pretrained(A['encoder']).to(device).eval()
    size = model_size(model)
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    repr_map = load_concept_representation(cfg.concept_repr_path())

    cands = {sp: pd.read_parquet(cfg.candidates_path(sp),
                                 columns=['mention_id', 'mention_text', 'cand_concept_id',
                                          'cand_ret_score'])
             for sp in ('train', 'dev', 'test')}
    concepts = sorted({str(c) for d in cands.values() for arr in d.cand_concept_id.values
                       for c in np.asarray(arr)[:top_m] if c is not None})
    t0 = time.time()
    Ev = AT.encode_normalised(model, tok, [repr_map.get(c, c) for c in concepts],
                              int(A['entity_max_len']), device, 512)
    ent_sec = time.time() - t0
    cpos = {c: i for i, c in enumerate(concepts)}

    rows, timing = [], dict(entity_encode_sec=round(ent_sec, 2), n_entities=len(concepts))
    for sp in ('train', 'dev', 'test'):
        d = cands[sp]
        q = pd.read_parquet(cfg.queries_path(sp))
        cmap = dict(zip(q.mention_id.astype(str), q.context_text.astype(str)))
        lmap = dict(zip(q.mention_id.astype(str), q.ctx_tokens.astype(int)))
        ids = d.mention_id.astype(str).to_numpy()
        ctxs = [cmap.get(i, '') for i in ids]
        t0 = time.time()
        Cv = AT.encode_normalised(model, tok, ctxs, int(A['ctx_max_len']), device,
                                  int(A['batch']))
        Mv = AT.encode_normalised(model, tok, d.mention_text.astype(str).tolist(),
                                  int(A['mention_max_len']), device, 512)
        timing[f'{sp}_context_mention_encode_sec'] = round(time.time() - t0, 2)

        idx = np.full((len(d), top_m), -1, dtype=np.int64)
        for i, arr in enumerate(d.cand_concept_id.values):
            for j, c in enumerate(np.asarray(arr)[:top_m]):
                if c is not None:
                    idx[i, j] = cpos[str(c)]
        Ei = Ev[torch.as_tensor(idx.clip(min=0))]
        th1, th2, th3, pen = AT.theta_block(Ei, Cv, Mv)
        srt = -np.sort(-pen, axis=1)
        ent = np.array([AT.entropy_top10(s, top_m) for s in d.cand_ret_score.values])
        sl = np.array([lmap.get(i, len(tok.tokenize(c))) for i, c in zip(ids, ctxs)],
                      dtype=np.float64)
        rows.append(pd.DataFrame(dict(
            dataset=cfg.key, split=sp, mention_id=ids,
            penalized_top1=srt[:, 0], penalized_top2=srt[:, 1],
            penalized_margin=srt[:, 0] - srt[:, 1],
            avg_theta1=th1.mean(1), avg_theta2=th2.mean(1), avg_theta3=th3.mean(1),
            entropy_top10=ent, sent_len=sl, n_cands=np.full(len(d), 64.0),
            retrieval_top1=np.array([np.asarray(s)[0] for s in d.cand_ret_score.values]))))
    F = pd.concat(rows, ignore_index=True)
    write_table(F, cfg.out_path('baselines', 'arter', 'features.parquet'))

    sub = {}
    for sp in ('train', 'dev', 'test'):
        t = pd.read_parquet(cfg.routing_table_path(sp))
        sub[sp] = F[F.split == sp].merge(
            t[['mention_id', 'HARD', 'correct_K1', 'correct_Khigh']],
            on='mention_id', how='inner')
    rf = RandomForestClassifier(n_estimators=int(A['rf']['n_estimators']),
                                random_state=int(A['rf']['random_state']), n_jobs=1)
    t0 = time.time()
    rf.fit(sub['train'][AT.FEATURES].values, sub['train'].HARD.values.astype(int))
    fit_s = time.time() - t0
    sc = {sp: rf.predict_proba(sub[sp][AT.FEATURES].values)[:, 1] for sp in sub}
    tau, j_star = AT.youden_threshold(sub['dev'].HARD.values, sc['dev'])

    for sp in ('dev', 'test'):
        t = pd.read_parquet(cfg.routing_table_path(sp))
        s = pd.Series(dict(zip(sub[sp].mention_id.astype(str), sc[sp])))
        v = s.reindex(t.mention_id.astype(str)).to_numpy()
        if pd.isna(v).any():
            raise SystemExit(f'{cfg.key}/{sp}: {int(pd.isna(v).sum())} mentions have no '
                             f'ARTER score')
        write_table(pd.DataFrame(dict(mention_id=t.mention_id.astype(str), score=v,
                                      native_route=(v >= tau).astype(int))),
                    cfg.out_path('baselines', 'arter', f'scores_{sp}.parquet'))

    write_table(pd.DataFrame(FIDELITY, columns=['component', 'original', 'ours', 'status',
                                                 'justification']),
                cfg.out_path('baselines', 'arter', 'fidelity.csv'))
    write_table(pd.DataFrame([dict(dataset=cfg.key, feature=f, importance=round(float(v), 6))
                              for f, v in zip(AT.FEATURES, rf.feature_importances_)]),
                cfg.out_path('baselines', 'arter', 'feature_importance.csv'))
    peak = (round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
            if device.startswith('cuda') else None)
    jdump(dict(dataset=cfg.key, method='ARTER (our adaptation)',
               fidelity='adaptation of a published method, not the original implementation',
               encoder=A['encoder'], encoder_model=size,
               encoder_choice_reason=('the original does not name a sentence-transformer '
                                      'checkpoint; a biomedical on-premise encoder from the '
                                      'same family as our retriever is used and stated'),
               features_used=AT.FEATURES, features_omitted=AT.OMITTED,
               penalized_formula=A['penalized'], entropy_formula=A['entropy'],
               rf=A['rf'], rf_fit_seconds=round(fit_s, 3), target=A['target'],
               threshold_rule=A['threshold_rule'], tau=round(float(tau), 6),
               youden_J=round(j_star, 6),
               dev_auroc=round(float(roc_auc_score(sub['dev'].HARD, sc['dev'])), 6),
               dev_rerank_rate=round(float((sc['dev'] >= tau).mean()), 6),
               timing=timing, gpu_peak_GiB=peak,
               TEST_USED_FOR_SELECTION=False, DEV_USED_FOR_THRESHOLD=True),
          cfg.out_path('baselines', 'arter', 'run.json'))
    write_efficiency([eff_row(
        'arter', clock, dataset=cfg.key,
        device=('CUDA' if device.startswith('cuda') else 'CPU'),
        gpu=(torch.cuda.get_device_name() if device.startswith('cuda') else None),
        model=A['encoder'], model_parameters_M=size['n_parameters_M'],
        n_rows=int(len(F)), n_entities=len(concepts), rf_fit_seconds=round(fit_s, 3),
        gpu_peak_GiB=peak)],
        cfg.out_path('efficiency', 'arter.csv'))
    print(f'[{cfg.key}] ARTER tau {tau:.6f} (Youden J {j_star:.4f}) · '
          f'dev rerank rate {(sc["dev"] >= tau).mean():.4f}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
