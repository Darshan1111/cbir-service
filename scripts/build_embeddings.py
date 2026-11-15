"""Embed every image with ResNet18 and compute spatial color histograms."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.features import embed_batch, load_resnet18, open_image, spatial_color_hist  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "images"
OUT = ROOT / "artifacts"
BATCH = 32


def main():
    OUT.mkdir(exist_ok=True)
    paths = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    model = load_resnet18()
    deep, color = [], []
    for i in range(0, len(paths), BATCH):
        ims = [open_image(p) for p in paths[i:i + BATCH]]
        deep.append(embed_batch(model, ims))
        color.extend(spatial_color_hist(im) for im in ims)
        print(f"{min(i + BATCH, len(paths))}/{len(paths)}", flush=True)
    np.save(OUT / "embeddings.npy", np.concatenate(deep))
    np.save(OUT / "color.npy", np.stack(color))
    (OUT / "paths.json").write_text(json.dumps([p.name for p in paths]))
    print("saved", np.concatenate(deep).shape, np.stack(color).shape)


if __name__ == "__main__":
    main()
