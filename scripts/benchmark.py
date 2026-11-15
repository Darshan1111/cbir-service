"""Benchmarks for the resume numbers: storage, top-5 overlap vs exact, and search latency.

Part 1 (real data): PCA dim sweep -> index size and recall@5 vs exact 512-d search.
Part 2 (synthetic): latency scaling, NumPy brute force (512-d) vs FAISS Flat/HNSW (PCA dim).
"""
import time
from pathlib import Path

import faiss
import numpy as np

ART = Path(__file__).resolve().parent.parent / "artifacts"
K = 5
rng = np.random.default_rng(0)


def l2n(x):
    x = x.astype("float32").copy()
    faiss.normalize_L2(x)
    return x


def topk_exact(db, q, k):
    """Leave-one-out exact search: drop self (index i) from results."""
    sims = q @ db.T
    np.fill_diagonal(sims, -np.inf)
    idx = np.argpartition(-sims, k, axis=1)[:, :k]
    return idx


def overlap(a, b):
    return float(np.mean([len(set(x) & set(y)) / len(x) for x, y in zip(a, b)]))


def timeit_per_query(fn, queries, warmup=20):
    for q in queries[:warmup]:
        fn(q)
    t = time.perf_counter()
    for q in queries:
        fn(q)
    return (time.perf_counter() - t) / len(queries) * 1000


def part1():
    emb = np.load(ART / "embeddings.npy")
    n = len(emb)
    full = l2n(emb)
    truth = topk_exact(full, full, K)
    # PCA mean-centres the data, which alone changes cosine rankings. Compare against
    # exact search on centred 512-d too, so compression loss is isolated from centring.
    cen = l2n(emb - emb.mean(0))
    truth_cen = topk_exact(cen, cen, K)
    print(f"raw-vs-centred 512-d overlap (centring effect alone): {overlap(truth, truth_cen):.3f}")
    base_mb = emb.nbytes / 1e6
    print(f"\n== Part 1: real data, N={n}, raw 512-d float32 = {base_mb:.2f} MB ==")
    print("dim  size_MB  saving  recall@5_vs_raw512  recall@5_vs_centred512")
    for dim in (32, 64, 128, 256):
        pca = faiss.PCAMatrix(512, dim)
        pca.train(emb.astype("float32"))
        red = l2n(pca.apply(emb.astype("float32")))
        idx = faiss.IndexFlatIP(dim)
        idx.add(red)
        _, I = idx.search(red, K + 1)
        res = np.array([[j for j in row if j != i][:K] for i, row in enumerate(I)])
        mb = n * dim * 4 / 1e6
        print(f"{dim:>3}  {mb:6.2f}  {100 * (1 - mb / base_mb):5.1f}%  {overlap(truth, res):.3f}              {overlap(truth_cen, res):.3f}")


def part2(dim=128, queries=300):
    print(f"\n== Part 2: synthetic latency scaling (dim 512 -> {dim}), {queries} single-query searches ==")
    emb = np.load(ART / "embeddings.npy").astype("float32")
    pca = faiss.PCAMatrix(512, dim)
    pca.train(emb)
    # synthetic corpora: perturbed copies of real embeddings (keeps realistic spectrum)
    print("N        numpy512_ms  faissFlat_ms  faissHNSW_ms  HNSW_recall@5_vs_flat")
    for n in (10_000, 100_000, 500_000):
        base = emb[rng.integers(0, len(emb), n)]
        db512 = l2n(base + rng.normal(0, 0.15 * base.std(), base.shape).astype("float32"))
        qs512 = db512[rng.integers(0, n, queries)]
        db = l2n(pca.apply(db512))
        qs = l2n(pca.apply(qs512))
        flat = faiss.IndexFlatIP(dim)
        flat.add(db)
        hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = 100
        hnsw.add(db)
        hnsw.hnsw.efSearch = 64

        def np_search(q):
            s = db512 @ q
            return np.argpartition(-s, K)[:K]

        t_np = timeit_per_query(np_search, qs512)
        t_flat = timeit_per_query(lambda q: flat.search(q[None], K), qs)
        t_hnsw = timeit_per_query(lambda q: hnsw.search(q[None], K), qs)
        _, If = flat.search(qs, K)
        _, Ih = hnsw.search(qs, K)
        print(f"{n:<8} {t_np:10.3f}  {t_flat:11.3f}  {t_hnsw:11.3f}  {overlap(If, Ih):.3f}")


if __name__ == "__main__":
    part1()
    part2()
