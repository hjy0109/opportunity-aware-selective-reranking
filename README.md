# Opportunity-Aware Selective Reranking for Efficient Biomedical Entity Linking

Reference implementation and evaluation code for the paper
*Opportunity-Aware Selective Reranking for Efficient Biomedical Entity Linking*,
Machine Learning for Health (ML4H) 2026, Proceedings of Machine Learning Research 346.

**Authors:** JiYe Han<sup>1,2</sup>, Minkyoung Kim<sup>2,3</sup>, Young-Hak Kim<sup>4</sup>,
Sanghyun Park<sup>3,\*</sup>, Tae Joon Jun<sup>2,\*</sup>

<sup>1</sup> Department of Artificial Intelligence, Yonsei University, Republic of Korea<br>
<sup>2</sup> Department of Information Medicine, Asan Medical Center, University of Ulsan College of Medicine, Republic of Korea<br>
<sup>3</sup> Department of Computer Science, Yonsei University, Republic of Korea<br>
<sup>4</sup> Division of Cardiology, Department of Internal Medicine, Asan Medical Center, University of Ulsan College of Medicine, Republic of Korea<br>
<sup>\*</sup> Co-corresponding authors

## 1. What this is

A cross-encoder reranks only a budgeted share of mentions, chosen by a small
router over retrieval-score features. The router is trained on **Opportunity** —
`1[retriever top-1 != gold AND 1 < gold rank <= K_high]`, a mention that is wrong
*and* still reachable at the deployed candidate depth. The controlled baseline,
**Difficulty**, drops the reachability clause (`1[retriever top-1 != gold]`) and
is otherwise identical: same features, classifier, hyperparameters, split, pool,
reranker, depth and rate, trained by the same command with a different `--target`.

## 2. Installation

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the reported runs. PyTorch was the
`2.5.1+cu121` build (`pip install torch==2.5.1 --index-url
https://download.pytorch.org/whl/cu121`), and XGBoost was the CPU-only build, with
the routers fitted at `device="cpu"`. Three roots locate everything, as
environment variables or as `--data-root` / `--model-root` / `--output-dir`:

```bash
export DATA_ROOT=/path/to/data      # corpora, ontologies, licensed resources
export MODEL_ROOT=/path/to/models   # trained checkpoints
export OUTPUT_DIR=/path/to/outputs  # everything this code writes
```

## 3. Data preparation

No corpus, ontology or UMLS file is included here. Obtain each dataset from its
own source, place it under `DATA_ROOT` at the paths in `configs/datasets/*.yaml`,
and run the preparation command.

| dataset | corpus | knowledge base |
| --- | --- | --- |
| MedMentions ST21pv | `github.com/chanzuckerberg/MedMentions` (`st21pv/data/corpus_pubtator.txt.gz` + official PMID lists) | UMLS 2017AA, ST21pv subset (licensed, user-supplied) |
| BC5CDR-Disease | BioCreative V CDR task | MeSH 2015, `type = disease` |
| NCBI Disease | NCBI BioNLP disease corpus | MEDIC (MeSH + OMIM disease) |
| NLM-Chem | `ftp.ncbi.nlm.nih.gov/pub/lu/NLMChem/` | MeSH 2021, `type = chemical` |

BioC corpora are read as one gzipped BioC XML per official split, with the
concept in the `concept_id` infon. Ontologies are gzipped JSON lists of
`{"id", "name", "aliases"}`; the candidate universe is the full ontology, and the
extended dictionaries that ship with some benchmark releases are not used.
BC5CDR additionally reads the original CDR release to restore the annotation
types its preprocessed files drop. MedMentions needs your own UMLS installation
(`umls_meta`, `umls_net` in its config); the ST21pv universe is rebuilt from it.

```bash
python -m src.data.prepare --dataset ncbi_disease
```

Counts, exclusions, gold coverage and fold leakage land in
`OUTPUT_DIR/<dataset>/audit/data_preparation.json`.

## 4. Reproducing a reported cell

For one dataset, in order; each command reads what the previous one wrote.

```bash
D=ncbi_disease
python -m src.data.prepare                           --dataset $D
CUDA_VISIBLE_DEVICES=0 python -m src.retrieval.run   --dataset $D
python -m src.data.queries                           --dataset $D
CUDA_VISIBLE_DEVICES=0 python -m src.reranking.train --dataset $D
CUDA_VISIBLE_DEVICES=0 python -m src.reranking.score --dataset $D --fold -1 \
    --splits dev test
python -m src.reranking.select_depth                 --dataset $D   # alpha, K_high
python -m src.routing.table                          --dataset $D
python -m src.routing.train                          --dataset $D --target OPP
python -m src.routing.train                          --dataset $D --target HARD
python -m src.routing.infer                          --dataset $D --target OPP \
    --split test --rate 0.10
python -m src.evaluation.evaluate                    --dataset $D --split test
python -m src.evaluation.bootstrap                   --dataset $D --split test
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.latency --dataset $D \
    --policies random margin difficulty opportunity arr arter \
    --rates 0.10 --n-timed 896
```

`--target BEN` needs the three out-of-fold rerankers (`--fold 0 1 2`);
`src.routing.table --no-benefit` skips them. The external baselines are
`src.baselines.arr_train` and `src.baselines.arter_run`. Retrieval and
cross-encoder training need a GPU; everything from `src.routing.table` onward
runs on CPU. `--n-timed` was 500 for MedMentions, 2000 for BC5CDR-Disease and
NLM-Chem, and 896 (the full split) for NCBI Disease.

## 5. Where the reported numbers come from

Everything below is under `OUTPUT_DIR/<dataset>/`.

| table column | file | field |
| --- | --- | --- |
| Acc@1 per method and rate | `evaluation/results.csv` | `method`, `reranking_rate`, `acc_at_1` |
| paired interval | `evaluation/contrasts.csv` | `delta_acc_pp`, `ci_low`, `ci_high` |
| latency, speedup | `latency/e2e_latency.csv` | `median_ms_per_mention`, `speedup_vs_full` |
| per-stage latency | `latency/stage_breakdown.csv` | `cell`, `stage`, `median_ms_per_mention` |
| budget composition | `evaluation/composition.csv` | `pct_*`, `*_capture` |
| selected alpha, K_high | `fusion/fusion_selection.json`, `fusion/candidate_depth.json` | `selected_alpha`, `selected_K_high` |

Absolute milliseconds are hardware-dependent; compare within one dataset against
the `full_reranking` cell of the same process, through `speedup_vs_full`.

## 6. Licensing

This code is released under the MIT License (`LICENSE.txt`).

- **Nothing licensed is redistributed here** — no corpus, ontology, UMLS file,
  UMLS-derived resource or credential.
- **UMLS** requires a licence from the U.S. National Library of Medicine. The
  code reads a path you supply.
- **MeSH, MEDIC and the four corpora** carry their own terms; obtain them from
  their own sources and follow those terms.
- **Pretrained models** (SapBERT, BiomedBERT) are downloaded from the Hugging
  Face Hub under their own licences and are not vendored here.
- **ARR and ARTER** here are our own adaptations, written against the published
  descriptions; no code from either original repository is included. Cite the
  original papers for those methods.

## 7. Citation

A BibTeX entry will be added here once the ML4H 2026 proceedings are published.
