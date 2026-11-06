"""Knowledge-base and concept-representation construction.

Two knowledge-base sources are supported, matching the two used in the reported
experiments:

    ``ontology_json``  a JSON ontology export with ``id``, ``name`` and
                       ``aliases`` per concept, optionally filtered by an entity
                       type map (MEDIC, MeSH 2015 disease, MeSH 2021 chemical).
    ``umls_st21pv``    UMLS MRCONSO/MRSTY plus the Semantic Network SRDEF, used
                       for the MedMentions ST21pv concept universe.

The candidate universe is always the full ontology.  The extended dictionaries
shipped with some benchmark releases add train/dev mention strings as aliases and
are therefore not used.
"""
from __future__ import annotations

import gzip
import json
from typing import Any

import pandas as pd

from ..utils.io import norm_alias

# MedMentions ST21pv semantic types, and the preferred source vocabularies used
# by the comparison branch that picks between the two admissible alias sets.
ST21PV_TUIS = ('T005', 'T007', 'T017', 'T022', 'T031', 'T033', 'T037', 'T038', 'T058',
               'T062', 'T074', 'T082', 'T091', 'T092', 'T097', 'T098', 'T103', 'T168',
               'T170', 'T201', 'T204')
ST21PV_SABS = ('CPT', 'FMA', 'GO', 'HGNC', 'HPO', 'ICD10', 'ICD10CM', 'ICD9CM', 'MDR',
               'MSH', 'MTH', 'NCBI', 'NCI', 'NDDF', 'NDFRT', 'OMIM', 'RXNORM', 'SNOMEDCT_US')

# Concept count reported for the ST21pv entity universe.  The two admissible KB
# definitions below (semantic types only, or semantic types restricted to the
# preferred vocabularies) are resolved by taking whichever is closer to it.
ST21PV_REPORTED_ENTITIES = 3_092_324

MRCONSO_COLUMNS = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI',
                   'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF']


# ---------------------------------------------------------------- ontology JSON
def build_kb_from_ontology(ontology_path: str, type_map_path: str | None,
                           universe_type: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    with gzip.open(ontology_path, 'rt') as f:
        onto = json.load(f)
    if type_map_path and universe_type:
        tmap = json.load(open(type_map_path))
        keep = [e for e in onto if tmap.get(e['id']) == universe_type]
    else:
        keep = onto
    rows = []
    for e in keep:
        seen: set[str] = set()
        for nm in [e['name']] + list(e.get('aliases', [])):
            if isinstance(nm, str) and nm.strip() and nm not in seen:
                seen.add(nm)
                rows.append(dict(concept_id=str(e['id']), name=nm))
    kb = (pd.DataFrame(rows)
          .sort_values(['concept_id', 'name'], kind='stable')
          .reset_index(drop=True))
    meta = dict(source='ontology_json', ontology=ontology_path,
                universe_type=universe_type,
                n_concepts_in_ontology=int(len(onto)),
                n_concepts_in_universe=int(len(keep)),
                n_aliases=int(len(kb)),
                aliases_per_concept=round(len(kb) / max(len(keep), 1), 3),
                biosyn_extended_dictionary_used=False)
    return kb, meta


# ---------------------------------------------------------------- UMLS ST21pv
def _tui_to_tree_number(srdef_path: str) -> dict[str, str]:
    """Semantic tree numbers from the Semantic Network SRDEF (field 4, rows RT='STY').

    MRSTY.RRF in some UMLS releases ships with the STN column empty, so the tree
    numbers are read from SRDEF rather than from MRSTY.
    """
    m: dict[str, str] = {}
    with open(srdef_path, encoding='utf-8') as f:
        for line in f:
            p = line.rstrip('\n').split('|')
            if len(p) > 3 and p[0] == 'STY':
                m[p[1]] = p[3]
    return m


def st21pv_types_in_scope(srdef_path: str) -> tuple[set[str], dict[str, Any]]:
    """The 21 ST21pv types plus every descendant type.

    ST21pv maps fine-grained UMLS types up to their nearest ancestor among the 21,
    so the concept universe must include descendants of those 21 tree numbers.
    Matching on the 21 TUIs alone would drop types that actually occur in the
    annotations.
    """
    tui2stn = _tui_to_tree_number(srdef_path)
    missing = [t for t in ST21PV_TUIS if t not in tui2stn]
    roots = tuple(sorted({tui2stn[t] for t in ST21PV_TUIS if t in tui2stn}))
    in_scope = {t for t, s in tui2stn.items()
                if any(s == r or s.startswith(r + '.') for r in roots)}
    return in_scope, dict(n_semantic_types_in_srdef=len(tui2stn),
                          n_st21pv_types_with_tree_number=21 - len(missing),
                          n_types_in_scope=len(in_scope),
                          st21pv_types_missing_tree_number=missing)


def build_kb_from_umls(meta_dir: str, net_dir: str,
                       chunksize: int = 2_000_000) -> tuple[pd.DataFrame, dict[str, Any]]:
    in_scope, scope_meta = st21pv_types_in_scope(f'{net_dir}/SRDEF')
    # keep_default_na=False is required: UMLS contains literal strings such as
    # 'NA' and 'null' which would otherwise be read as missing values.
    sty = pd.read_csv(f'{meta_dir}/MRSTY.RRF', sep='|', header=None, dtype=str, quoting=3,
                      usecols=[0, 1], names=['CUI', 'TUI'], engine='c',
                      keep_default_na=False)
    keep_cui = set(sty.loc[sty['TUI'].isin(in_scope), 'CUI'])
    n_cui_all = int(sty['CUI'].nunique())
    del sty

    use = ['CUI', 'LAT', 'SAB', 'STR', 'SUPPRESS']
    idx = [MRCONSO_COLUMNS.index(c) for c in use]
    parts = []
    for chunk in pd.read_csv(f'{meta_dir}/MRCONSO.RRF', sep='|', header=None, dtype=str,
                             usecols=idx, names=use, engine='c', quoting=3,
                             keep_default_na=False, chunksize=chunksize,
                             on_bad_lines='skip'):
        chunk = chunk[(chunk['LAT'] == 'ENG') & (chunk['SUPPRESS'] == 'N')]
        parts.append(chunk[chunk['CUI'].isin(keep_cui)][['CUI', 'SAB', 'STR']])
    con = pd.concat(parts, ignore_index=True)
    del parts
    con['STR'] = con['STR'].str.strip()
    con = con[con['STR'] != '']

    stats = dict(n_cui_types_only=int(con['CUI'].nunique()),
                 n_names_types_only=int(len(con.drop_duplicates(['CUI', 'STR']))))
    pv = con[con['SAB'].isin(ST21PV_SABS)]
    stats.update(n_cui_types_and_preferred_sabs=int(pv['CUI'].nunique()),
                 n_names_types_and_preferred_sabs=int(len(pv.drop_duplicates(['CUI', 'STR']))))
    target = ST21PV_REPORTED_ENTITIES
    use_pv = (abs(stats['n_cui_types_and_preferred_sabs'] - target)
              < abs(stats['n_cui_types_only'] - target))
    sel = pv if use_pv else con
    kb = (sel[['CUI', 'STR']].drop_duplicates()
          .rename(columns={'CUI': 'concept_id', 'STR': 'name'})
          .sort_values(['concept_id', 'name'], kind='stable')
          .reset_index(drop=True))
    meta = dict(source='umls_st21pv', meta_dir=meta_dir, net_dir=net_dir,
                filter='LAT=ENG and SUPPRESS=N',
                kb_definition=('21 semantic types incl. descendants, restricted to the '
                               'ST21pv preferred vocabularies' if use_pv else
                               '21 semantic types incl. descendants, all source vocabularies'),
                kb_definition_rule=('whichever of the two admissible definitions is closer '
                                    f'to the reported {target:,} entities'),
                n_cui_in_release=n_cui_all,
                n_concepts=int(kb['concept_id'].nunique()), n_aliases=int(len(kb)),
                biosyn_extended_dictionary_used=False, **scope_meta, **stats)
    return kb, meta


def build_kb(cfg) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = cfg.dataset['knowledge_base']
    if spec['kind'] == 'ontology_json':
        return build_kb_from_ontology(
            cfg.data_path(spec['ontology']),
            cfg.data_path(spec['type_map']) if spec.get('type_map') else None,
            spec.get('universe_type'))
    if spec['kind'] == 'umls_st21pv':
        return build_kb_from_umls(cfg.data_path(spec['umls_meta']),
                                  cfg.data_path(spec['umls_net']))
    raise ValueError(f'unknown knowledge_base.kind: {spec["kind"]}')


# ---------------------------------------------------------------- R2 representation
def build_concept_representation(kb: pd.DataFrame, separator: str = ' [SEP] ') -> pd.DataFrame:
    """The frozen "R2" concept representation.

    Per concept, aliases are de-duplicated on the normalised form, sorted by
    ``(norm(alias), original_alias)``, and the first two distinct aliases are
    joined by ``separator``.  A concept with a single distinct alias degenerates
    to that alias.
    """
    acc: dict[str, dict[str, str]] = {}
    for c, n in zip(kb.concept_id.to_numpy(), kb.name.to_numpy()):
        d = acc.setdefault(str(c), {})
        k = norm_alias(n)
        if k not in d:
            d[k] = str(n)
    rows = []
    for c, d in acc.items():
        ordered = [v for _, v in sorted(d.items(), key=lambda kv: (kv[0], kv[1]))]
        a1 = ordered[0]
        a2 = ordered[1] if len(ordered) > 1 else None
        rows.append(dict(concept_id=c, a1=a1, a2=a2, n_distinct=len(ordered),
                         representation=(a1 if a2 is None else f'{a1}{separator}{a2}')))
    return pd.DataFrame(rows).sort_values('concept_id').reset_index(drop=True)


def load_concept_representation(path: str) -> dict[str, str]:
    t = pd.read_parquet(path)
    return dict(zip(t.concept_id.astype(str), t.representation.astype(str)))


def kb_arrays(kb_path: str):
    """(alias texts, alias -> concept index, concept ids), ordered by concept.

    The alias list must be grouped by concept so that a streaming KB block never
    splits a concept across two blocks.
    """
    import numpy as np
    kb = pd.read_parquet(kb_path)
    kb = kb[kb.name.map(lambda v: isinstance(v, str) and v.strip() != '')]
    cids = kb.concept_id.astype(str).to_numpy()
    uniq, inv = np.unique(cids, return_inverse=True)
    order = np.argsort(inv, kind='stable')
    kb = kb.iloc[order].reset_index(drop=True)
    return kb.name.tolist(), inv[order].astype(np.int64), uniq
