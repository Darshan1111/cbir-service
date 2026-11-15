"""Fit PCA (512 -> DIM), L2-normalise, build a FAISS inner-product index, save artifacts."""
import sys
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 128


def main():
    emb = np.load(ART / "embeddings.npy").astype("float32")
    pca = faiss.PCAMatrix(emb.shape[1], DIM)
    pca.train(emb)
    red = pca.apply(emb)
    faiss.normalize_L2(red)
    index = faiss.IndexFlatIP(DIM)
    index.add(red)
    faiss.write_index(index, str(ART / "index.faiss"))
    faiss.write_VectorTransform(pca, str(ART / "pca.faiss"))
    print(f"index: {index.ntotal} vectors x {DIM}d -> {(ART / 'index.faiss').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
