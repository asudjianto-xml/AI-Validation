"""Leaf-value-weighted co-membership geometry for a frozen scalar-output GBDT.

Coordinate (tree, leaf) is the fitted leaf contribution when reached, else zero.
The dot product therefore weights shared leaves by squared fitted contributions.
Dumped XGBoost leaf values already include shrinkage: do not multiply it again.
No centering, coordinate scaling or kernel normalization is applied.
"""
import json
import numpy as np


def leaf_tables(model):
    tables = []
    for raw in model.get_booster().get_dump(dump_format='json'):
        leaves = {}
        def visit(node):
            if 'leaf' in node:
                leaves[int(node['nodeid'])] = float(node['leaf'])
            else:
                for child in node['children']:
                    visit(child)
        visit(json.loads(raw))
        tables.append(leaves)
    return tables


def weighted_leaf_embedding(codes, tables):
    """Use all fitted leaves in sorted order, including leaves absent in this batch.

    Columns are stable across discovery, medoids and confirmation for a frozen
    model. IDs are categorical and local to a tree. Zero-valued leaves have zero
    weight, so this is a pseudometric on observations, as any such embedding is.
    """
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.shape[1] != len(tables):
        raise ValueError('Expected one leaf-ID column per fitted tree')
    if not np.isfinite(codes).all() or not np.equal(codes, np.floor(codes)).all():
        raise ValueError('Leaf IDs must be finite integers')
    out = np.zeros((len(codes), sum(len(t) for t in tables)), dtype=np.float64)
    offset = 0
    for t, table in enumerate(tables):
        ids = np.array(sorted(table), dtype=int)
        values = np.array([table[k] for k in ids], dtype=np.float64)
        if not len(ids) or not np.isfinite(values).all():
            raise ValueError('Each tree must have finite fitted leaf values')
        positions = np.searchsorted(ids, codes[:, t])
        if np.any(positions >= len(ids)) or np.any(ids[positions] != codes[:, t]):
            raise ValueError(f'Unknown leaf ID in tree {t}')
        out[np.arange(len(codes)), offset + positions] = values[positions]
        offset += len(ids)
    return out


def embedding(model, X):
    return weighted_leaf_embedding(model.apply(X), leaf_tables(model))
