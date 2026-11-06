"""Opportunity-aware selective reranking for efficient biomedical entity linking.

Reference implementation accompanying the ML4H 2026 paper.  Every module here is a
reorganised copy of the code that produced the reported numbers; the frozen
experimental definitions (features, supervision targets, candidate depth rule,
score fusion, budget matching, bootstrap and latency protocols) are unchanged.
"""
