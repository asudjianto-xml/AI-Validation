"""GPU implementation of data twinning (Vakayil and Joseph, 2022).

Public entry points mirror the reference package: :func:`twin`, :func:`multiplet`
and :func:`energy`.

Two additions have no counterpart in the reference. :func:`twin_batch` runs
independent chains in lockstep, because one chain leaves most of a GPU idle
whereas cross-validation folds, bootstrap replicates or seed sweeps do not. The
``blocks`` argument partitions the dataset spatially and twins each part
concurrently, which turns the serial chain into ``blocks`` shorter chains that
advance together; it is an approximation, quantified in the benchmarks.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ._chain import run_chains

_DEFAULT_DTYPE = torch.float32

# Rows per block that ``blocks="auto"`` aims for. Split quality tracks this
# number rather than the block count, and at much the same rate for every r: at
# roughly 3000 rows per block the energy distance came to 1.1 to 1.4 times that
# of an exact run for r in {2, 5, 20}, while a few hundred rows per block was
# worse than splitting at random. See benchmarks/bench.py.
_BLOCK_TARGET = 4096
_MAX_BLOCKS = 4096


def _auto_blocks(N: int, target: int = _BLOCK_TARGET) -> int:
    """Largest power of two that leaves at least ``target`` rows in every block."""
    blocks = 1
    while blocks < _MAX_BLOCKS and N // (blocks * 2) >= target:
        blocks *= 2
    return blocks


def _resolve_device(device) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_matrix(data, name: str) -> torch.Tensor:
    X = data if torch.is_tensor(data) else torch.as_tensor(np.asarray(data))
    if X.dim() != 2:
        raise ValueError(f"{name} must be a 2-dimensional array")
    if not torch.isfinite(X).all():
        raise ValueError(f"{name} cannot contain nan or infinity")
    return X


def standardize(data, dtype=_DEFAULT_DTYPE, device=None) -> torch.Tensor:
    """Drop constant columns and scale the rest to zero mean and unit variance.

    The standard deviation is the population one, matching the reference
    implementation, so that both operate on identical coordinates.
    """
    device = _resolve_device(device)
    X = _as_matrix(data, "data").to(device=device, dtype=torch.float64)
    keep = ~(X == X[0]).all(dim=0)
    X = X[:, keep]
    X = (X - X.mean(dim=0)) / X.std(dim=0, correction=0)
    return X.to(dtype).contiguous()


def _validate_r(r: int, N: int) -> None:
    if not isinstance(r, (int, np.integer)) or not 2 <= r <= N // 2:
        raise ValueError(f"r must be an integer with 2 <= r <= {N // 2}")


def _starts(pts: torch.Tensor, u1) -> torch.Tensor:
    """Starting row of every chain, as a (G, 1) tensor of in-group positions.

    ``"far"`` takes the point farthest from the group centroid, the choice used
    for the experiments in the paper, and makes the run deterministic. ``None``
    draws at random, as the reference does.
    """
    G, M, _ = pts.shape
    if isinstance(u1, str):
        if u1 != "far":
            raise ValueError("u1 must be a row index, 'far', or None")
        centre = pts.mean(dim=1, keepdim=True)
        return ((pts - centre) ** 2).sum(-1).argmax(dim=1, keepdim=True)
    if u1 is None:
        return torch.randint(M, (G, 1), dtype=torch.long, device=pts.device)
    starts = _start_rows(u1, M, pts.device)
    if starts.shape[0] != G:
        raise ValueError(f"expected {G} starting rows, got {starts.shape[0]}")
    return starts


def _start_rows(u1, M: int, device) -> torch.Tensor:
    """Validated (B, 1) tensor of starting rows."""
    starts = torch.as_tensor(np.asarray(u1), dtype=torch.long, device=device).reshape(-1, 1)
    if starts.numel() == 0:
        raise ValueError("u1 must contain at least one starting row")
    if not ((starts >= 0) & (starts < M)).all():
        raise ValueError(f"every u1 must satisfy 0 <= u1 < {M}")
    return starts


def kd_blocks(X: torch.Tensor, n_blocks: int) -> list[torch.Tensor]:
    """Partition rows into ``n_blocks`` spatially coherent groups of equal size.

    Each level splits every group at the median of its widest-spread coordinate,
    so sizes stay within one of each other and neighbouring points mostly land
    together. Groups of equal size are split in one batched sort, so the cost is
    ``log2(n_blocks)`` sorts rather than ``n_blocks`` of them.
    """
    if n_blocks & (n_blocks - 1):
        raise ValueError("n_blocks must be a power of 2")
    groups = [torch.arange(X.shape[0], device=X.device)]

    for _ in range(int(math.log2(n_blocks))):
        nxt: list[torch.Tensor] = []
        by_size: dict[int, list[torch.Tensor]] = {}
        for g in groups:
            by_size.setdefault(int(g.numel()), []).append(g)

        for size, same in by_size.items():
            if size < 2:
                nxt.extend(same)
                continue
            idx = torch.stack(same)
            pts = X[idx]
            axis = (pts.amax(1) - pts.amin(1)).argmax(dim=1)
            key = pts.gather(2, axis[:, None, None].expand(-1, size, 1)).squeeze(2)
            ordered = idx.gather(1, key.argsort(dim=1))
            half = (size + 1) // 2
            nxt.extend(ordered[:, :half].unbind(0))
            nxt.extend(ordered[:, half:].unbind(0))
        groups = nxt
    return groups


def _twin_indices(X: torch.Tensor, r: int, blocks, u1, **opts):
    """Seeds and visitation order over standardised coordinates.

    With ``blocks == 1`` this is one exact chain. Otherwise the rows are
    partitioned first and one chain runs per block, all in lockstep; blocks of
    equal size share a batched run, and at most two sizes ever occur.
    """
    blocks = _auto_blocks(X.shape[0]) if blocks == "auto" else int(blocks)
    if blocks == 1:
        pts = X.unsqueeze(0)
        seeds, order = run_chains(pts, r, _starts(pts, u1), **opts)
        return seeds[0], order[0]

    seeds_out: list[torch.Tensor] = []
    order_out: list[torch.Tensor] = []
    by_size: dict[int, list[torch.Tensor]] = {}
    for g in kd_blocks(X, blocks):
        by_size.setdefault(int(g.numel()), []).append(g)

    for size, same in sorted(by_size.items()):
        if size < 2 * r:
            raise ValueError(
                f"blocks={blocks} leaves {size} rows per block, too few for r={r}; "
                "use fewer blocks"
            )
        idx = torch.stack(same)
        pts = X[idx]
        seeds, order = run_chains(pts, r, _starts(pts, u1), **opts)
        seeds_out.append(idx.gather(1, seeds).reshape(-1))
        order_out.append(idx.gather(1, order).reshape(-1))

    return torch.cat(seeds_out), torch.cat(order_out)


def twin(
    data,
    r: int,
    u1: int | str | None = None,
    *,
    blocks: int | str = 1,
    dtype=_DEFAULT_DTYPE,
    device=None,
    use_graphs: bool = True,
    compact: bool = True,
    return_order: bool = False,
):
    """Partition ``data`` into two statistically similar twins.

    Returns the indices of the smaller twin in selection order. ``u1`` fixes the
    starting row and makes the result deterministic, ``"far"`` starts from the
    point farthest from the centroid, and ``None`` draws at random as the
    reference does.

    ``blocks`` greater than one splits the dataset spatially and twins each part
    concurrently, which removes the serial bottleneck at the cost of neighbours
    separated by a block boundary. ``"auto"`` picks the block count from the
    measured quality guideline. The smaller twin then holds ``sum(ceil(m / r))``
    rows over the blocks, which exceeds ``ceil(N / r)`` slightly unless ``r``
    divides the block size.
    """
    X = standardize(data, dtype=dtype, device=device)
    N = X.shape[0]
    _validate_r(r, N)
    if isinstance(u1, (int, np.integer)):
        if blocks != 1:
            raise ValueError(
                "a single u1 row starts a single chain, so it applies only to "
                "blocks=1; with blocking pass u1='far' or u1=None, which choose "
                "a start within each block"
            )
        u1 = [int(u1)]
    seeds, order = _twin_indices(
        X, r, blocks, u1, use_graphs=use_graphs, compact=compact
    )
    return (seeds, order) if return_order else seeds


def twin_batch(
    data,
    r: int,
    u1,
    *,
    dtype=_DEFAULT_DTYPE,
    device=None,
    use_graphs: bool = True,
    compact: bool = True,
    return_order: bool = False,
):
    """Run one chain per starting row in ``u1``, all over the same dataset.

    The chains are independent, so they advance in lockstep over a shared batch
    dimension and the per-step cost, which is dominated by fixed kernel overhead
    rather than by data volume, is paid once for the whole batch. Returns a
    (B, ceil(N / r)) tensor of small-twin indices.
    """
    X = standardize(data, dtype=dtype, device=device)
    N = X.shape[0]
    _validate_r(r, N)
    starts = _start_rows(u1, N, X.device)
    B = starts.shape[0]
    seeds, order = run_chains(
        X.unsqueeze(0).expand(B, -1, -1), r, starts, use_graphs=use_graphs, compact=compact
    )
    return (seeds, order) if return_order else seeds


def multiplet(
    data,
    k: int,
    strategy: int = 1,
    *,
    blocks: int | str = 1,
    dtype=_DEFAULT_DTYPE,
    device=None,
    use_graphs: bool = True,
    compact: bool = True,
):
    """Partition ``data`` into ``k`` statistically similar disjoint sets.

    The three strategies are those of Vakayil and Joseph (2022). Strategy 1 peels
    one multiplet at a time at ratio ``1 / (k - i)``; strategy 2 halves
    recursively and requires ``k`` to be a power of two; strategy 3 reads the
    assignment off a single chain, which costs one run rather than ``k - 1`` at
    some loss of quality. Returns the multiplet id of every row.
    """
    X = standardize(data, dtype=dtype, device=device)
    N = X.shape[0]
    if not isinstance(k, (int, np.integer)) or not 2 <= k <= N // 2:
        raise ValueError(f"k must be an integer with 2 <= k <= {N // 2}")
    opts = dict(use_graphs=use_graphs, compact=compact)

    if strategy == 3:
        _, order = _twin_indices(X, k, blocks, "far", **opts)
        out = torch.empty(N, dtype=torch.long, device=X.device)
        out[order] = torch.arange(N, device=X.device) % k
        return out

    if strategy == 1:
        out = torch.empty(N, dtype=torch.long, device=X.device)
        rows = torch.arange(N, device=X.device)
        pool = X
        for i in range(k - 1):
            seeds, _ = _twin_indices(pool, k - i, blocks, "far", **opts)
            out[rows[seeds]] = i
            keep = torch.ones(pool.shape[0], dtype=torch.bool, device=X.device)
            keep[seeds] = False
            pool, rows = pool[keep].contiguous(), rows[keep]
        out[rows] = k - 1
        return out

    if strategy == 2:
        if k & (k - 1):
            raise ValueError("strategy 2 requires k to be a power of 2")
        return _halve(X, k, blocks, **opts)

    raise ValueError("strategy must be 1, 2 or 3")


def _halve(X: torch.Tensor, k: int, blocks, **opts) -> torch.Tensor:
    """Strategy 2: recursive equal twinning.

    Every group at a level is split into blocks and every block of every group is
    twinned in a single lockstep run, so a level costs one batched chain however
    many groups it holds. Without blocking the first level alone would run
    ``N / 2`` sequential steps, which dominates everything below it.
    """
    N = X.shape[0]
    out = torch.empty(N, dtype=torch.long, device=X.device)
    groups = [torch.arange(N, device=X.device)]

    for _ in range(int(math.log2(k))):
        pieces: list[torch.Tensor] = []
        owner: list[int] = []
        for gi, g in enumerate(groups):
            for piece in _split_into_blocks(X, g, blocks):
                pieces.append(piece)
                owner.append(gi)

        chosen: list[list[torch.Tensor]] = [[] for _ in groups]
        by_size: dict[int, list[int]] = {}
        for pos, piece in enumerate(pieces):
            by_size.setdefault(int(piece.numel()), []).append(pos)

        for size, positions in by_size.items():
            idx = torch.stack([pieces[p] for p in positions])
            if size < 4:
                for row, p in enumerate(positions):
                    chosen[owner[p]].append(idx[row, : size // 2])
                continue
            pts = X[idx]
            seeds, _ = run_chains(pts, 2, _starts(pts, "far"), **opts)
            taken = idx.gather(1, seeds)
            for row, p in enumerate(positions):
                chosen[owner[p]].append(taken[row])

        nxt: list[torch.Tensor] = []
        for gi, g in enumerate(groups):
            mark = torch.zeros(N, dtype=torch.bool, device=X.device)
            mark[torch.cat(chosen[gi])] = True
            picked = mark[g]
            nxt.extend([g[picked], g[~picked]])
        groups = nxt

    for i, g in enumerate(groups):
        out[g] = i
    return out


def _split_into_blocks(X: torch.Tensor, g: torch.Tensor, blocks) -> list[torch.Tensor]:
    """Rows of group ``g`` divided into blocks, as global indices."""
    m = int(g.numel())
    n_blocks = _auto_blocks(m) if blocks == "auto" else int(blocks)
    while n_blocks > 1 and m // n_blocks < 8:
        n_blocks //= 2
    if n_blocks <= 1:
        return [g]
    return [g[b] for b in kd_blocks(X[g], n_blocks)]


def energy(
    data, points, *, full: bool = False, dtype=_DEFAULT_DTYPE, device=None, block: int = 8192
) -> float:
    """Energy distance between ``data`` and ``points``, the criterion twinning minimises.

    By default the distances among the rows of ``data`` are omitted, as in the
    reference implementation, since they are a constant offset once the dataset
    is fixed. That offset is large compared with the differences between
    candidate subsets, so ratios of the default value are close to one and only
    differences carry information. ``full=True`` restores the term and returns
    the energy distance itself, which approaches zero as ``points`` comes to
    resemble ``data``, at a cost quadratic in the size of ``data``.

    ``points`` is scaled by the mean and standard deviation of ``data``. Sums
    accumulate in double precision.
    """
    device = _resolve_device(device)
    Z = _as_matrix(data, "data").to(device=device, dtype=torch.float64)
    U = _as_matrix(points, "points").to(device=device, dtype=torch.float64)
    if Z.shape[1] != U.shape[1]:
        raise ValueError("data and points must have the same number of columns")

    keep = ~(Z == Z[0]).all(dim=0)
    Z, U = Z[:, keep], U[:, keep]
    mean, std = Z.mean(dim=0), Z.std(dim=0, correction=0)
    Z = ((Z - mean) / std).to(dtype).contiguous()
    U = ((U - mean) / std).to(dtype).contiguous()

    N, n = Z.shape[0], U.shape[0]
    ed = 2.0 * _distance_sum(U, Z, block) / (n * N) - _distance_sum(U, U, block) / (n * n)
    if full:
        ed -= _distance_sum(Z, Z, block) / (N * N)
    return float(ed)


def _distance_sum(A: torch.Tensor, B: torch.Tensor, block: int) -> float:
    """Sum of Euclidean distances over all row pairs, blocked to bound memory."""
    total = torch.zeros((), dtype=torch.float64, device=A.device)
    for i in range(0, A.shape[0], block):
        a = A[i : i + block]
        for j in range(0, B.shape[0], block):
            total += torch.cdist(a, B[j : j + block]).sum(dtype=torch.float64)
    return float(total)
