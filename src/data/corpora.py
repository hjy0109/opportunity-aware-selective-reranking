"""Mention-set construction from the released corpora.

Two corpus readers, matching the two used in the reported experiments:

    ``bioc``                  BioC XML (one file per official split) for
                              BC5CDR-Disease, NCBI Disease and NLM-Chem.
    ``medmentions_pubtator``  the PubTator-format ST21pv corpus plus the official
                              PMID split lists.

Both produce the same frame: one row per mention with ``mention_id``,
``example_id``, ``document_id``, ``mention_text``, ``char_start``, ``char_end``,
``gold_concept_id``, and the full document text is returned alongside so the
context window can be cut from it.

Composite annotations (a gold id containing ``|`` or ``+``) are excluded and the
excluded share is reported; annotations with more than one location
(non-contiguous mentions) are excluded as well.
"""
from __future__ import annotations

import gzip
from typing import Any

import numpy as np
import pandas as pd

COMPOSITE_RULE = ('gold ids containing "|" or "+" are excluded (composite annotations); '
                  'the excluded count is reported per split')


# ---------------------------------------------------------------- BioC
def bioc_document_text(doc) -> str:
    """Re-lay the passages at their own offsets, so global offsets stay valid."""
    end = max((p.offset + len(p.text or '')) for p in doc.passages)
    buf = [' '] * end
    for p in doc.passages:
        t = p.text or ''
        for i, ch in enumerate(t):
            buf[p.offset + i] = ch
    return ''.join(buf)


def load_entity_types(source_dir: str) -> dict[tuple[str, int, int], str]:
    """Annotation types keyed by (document, offset, length), from an original release.

    Needed for BC5CDR, whose preprocessed BioC files do not carry the annotation
    type that separates the disease and chemical subtasks.
    """
    from bioc import biocxml
    m: dict[tuple[str, int, int], str] = {}
    for fn in ('CDR_TrainingSet.BioC.xml', 'CDR_DevelopmentSet.BioC.xml',
               'CDR_TestSet.BioC.xml'):
        with open(f'{source_dir}/{fn}') as f:
            coll = biocxml.load(f)
        for doc in coll.documents:
            for p in doc.passages:
                for a in p.annotations:
                    if len(a.locations) == 1:
                        loc = a.locations[0]
                        m[(str(doc.id), loc.offset, loc.length)] = a.infons.get('type')
    return m


def load_bioc_split(path: str, dataset: str, split: str, mention_type: str | None,
                    types: dict[tuple[str, int, int], str] | None):
    from bioc import biocxml
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf8') as f:
        coll = biocxml.load(f)
    rows: list[dict[str, Any]] = []
    docs: dict[str, str] = {}
    stat = dict(n_annotations=0, n_multi_location=0, n_type_filtered=0,
                n_composite=0, n_span_mismatch=0)
    for doc in coll.documents:
        full = bioc_document_text(doc)
        docs[str(doc.id)] = full
        for p in doc.passages:
            for a in p.annotations:
                stat['n_annotations'] += 1
                if len(a.locations) != 1:
                    stat['n_multi_location'] += 1
                    continue
                loc = a.locations[0]
                if mention_type is not None:
                    t = (types or {}).get((str(doc.id), loc.offset, loc.length))
                    if t is None or t.lower() != mention_type:
                        stat['n_type_filtered'] += 1
                        continue
                gold = str(a.infons.get('concept_id', ''))
                if ('|' in gold) or ('+' in gold):
                    stat['n_composite'] += 1
                    continue
                s, e = int(loc.offset), int(loc.offset) + int(loc.length)
                if full[s:e] != a.text:
                    stat['n_span_mismatch'] += 1
                rows.append(dict(
                    mention_id=f'{doc.id}:{s}:{e}', example_id=f'{doc.id}:{s}:{e}:{gold}',
                    document_id=str(doc.id), mention_text=str(a.text), char_start=s,
                    char_end=e, gold_concept_id=gold, split=split, dataset=dataset))
    return _finish(pd.DataFrame(rows), docs, stat)


def load_pubtator_splits(corpus_path: str, pmid_files: dict[str, str], dataset: str):
    """Mentions for every split of a PubTator corpus, using the official PMID lists."""
    pmids = {sp: set(open(p).read().split()) for sp, p in pmid_files.items()}
    of_split = {p: sp for sp, ps in pmids.items() for p in ps}

    docs_raw: dict[str, dict[str, str]] = {}
    ann: list[tuple[str, int, int, str, str, str]] = []
    opener = gzip.open if corpus_path.endswith('.gz') else open
    with opener(corpus_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            if '|t|' in line[:20] or '|a|' in line[:20]:
                pid, kind, txt = line.split('|', 2)
                docs_raw.setdefault(pid, {'t': '', 'a': ''})[kind] = txt
            else:
                p = line.split('\t')
                if len(p) < 6:
                    continue
                ann.append((p[0], int(p[1]), int(p[2]), p[3], p[4], p[5]))
    documents = {p: d['t'] + ' ' + d['a'] for p, d in docs_raw.items()}

    out: dict[str, tuple[pd.DataFrame, dict[str, str], dict[str, Any]]] = {}
    per_split: dict[str, list[dict[str, Any]]] = {sp: [] for sp in pmid_files}
    stat = dict(n_annotations=len(ann), n_outside_split=0, n_composite=0,
                n_span_mismatch=0, n_multi_location=0, n_type_filtered=0)
    for pid, s, e, txt, typ, cui in ann:
        sp = of_split.get(pid)
        if sp is None:
            stat['n_outside_split'] += 1
            continue
        gold = cui.replace('UMLS:', '')
        if ('|' in gold) or ('+' in gold):
            stat['n_composite'] += 1
            continue
        full = documents.get(pid, '')
        if full[s:e] != txt:
            stat['n_span_mismatch'] += 1
        per_split[sp].append(dict(
            mention_id=f'mm:{pid}:{s}:{e}', example_id=f'mm:{pid}:{s}:{e}:{gold}',
            document_id=pid, mention_text=str(txt), char_start=int(s), char_end=int(e),
            gold_concept_id=gold, semantic_type=typ, split=sp, dataset=dataset))
    for sp, rows in per_split.items():
        m = pd.DataFrame(rows)
        keep = set(m.document_id.astype(str)) if len(m) else set()
        docs = {p: t for p, t in documents.items() if p in keep}
        out[sp] = _finish(m, docs, dict(stat))
    return out


# ---------------------------------------------------------------- shared tail
def _finish(m: pd.DataFrame, docs: dict[str, str], stat: dict[str, Any]):
    n0 = len(m)
    if n0:
        m = m.drop_duplicates('example_id', keep='first').reset_index(drop=True)
        stat['n_same_span_multiple_gold'] = int(
            m.groupby(['document_id', 'char_start', 'char_end'])
            .gold_concept_id.nunique().gt(1).sum())
    else:
        stat['n_same_span_multiple_gold'] = 0
    stat['n_exact_duplicate_rows_removed'] = int(n0 - len(m))
    stat['n_mentions'] = int(len(m))
    stat['n_documents'] = int(m.document_id.nunique()) if len(m) else 0
    stat['composite_rule'] = COMPOSITE_RULE
    return m, docs, stat


def gold_coverage(m: pd.DataFrame, universe: set[str]) -> dict[str, Any]:
    inu = m.gold_concept_id.isin(universe)
    return dict(gold_in_universe_pct=round(100 * float(inu.mean()), 4),
                n_gold_out_of_universe=int((~inu).sum()),
                n_distinct_gold=int(m.gold_concept_id.nunique()))


def load_corpus(cfg, splits=('train', 'dev', 'test')):
    """Dispatch on ``corpus.kind``; returns {split: (mentions, documents, stats)}."""
    spec = cfg.dataset['corpus']
    ds = cfg.key
    if spec['kind'] == 'bioc':
        types = None
        if spec.get('mention_type'):
            types = load_entity_types(cfg.data_path(spec['entity_type_source']))
        out = {}
        for sp in splits:
            out[sp] = load_bioc_split(cfg.data_path(spec['files'][sp]), ds, sp,
                                      spec.get('mention_type'), types)
        return out
    if spec['kind'] == 'medmentions_pubtator':
        files = spec['files']
        pmid_files = {sp: cfg.data_path(files[f'{sp}_pmids']) for sp in splits}
        allsp = load_pubtator_splits(cfg.data_path(files['corpus']), pmid_files, ds)
        return {sp: allsp[sp] for sp in splits}
    raise ValueError(f'unknown corpus.kind: {spec["kind"]}')


def mention_context_column(m: pd.DataFrame, documents: dict[str, str], pad: int = 1):
    """Stored fallback window (mention sentence +/- pad), used when the crop cannot
    locate the mention in its document."""
    from .context import sentence_spans, sentence_window
    cache: dict[str, list[tuple[int, int]]] = {}
    out = []
    for r in m.itertuples(index=False):
        did = str(r.document_id)
        full = documents.get(did, '')
        if did not in cache:
            cache[did] = sentence_spans(full)
        w, _, _ = sentence_window(full, int(r.char_start), int(r.char_end), pad=pad,
                                  spans=cache[did])
        out.append(w.strip())
    return np.asarray(out, dtype=object)
