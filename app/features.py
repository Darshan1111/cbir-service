"""Feature extractors: ResNet18 deep embeddings and region-aware spatial color histograms."""
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms

_PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_resnet18():
    """ResNet18 with the classifier removed -> 512-d global-pooled embedding."""
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    return model.eval()


@torch.no_grad()
def embed_batch(model, pil_images):
    batch = torch.stack([_PREPROCESS(im.convert("RGB")) for im in pil_images])
    return model(batch).numpy().astype("float32")


def spatial_color_hist(pil_image, grid=3, bins=(8, 4, 4)):
    """Region-aware color feature: per-cell HSV histograms over a grid, concatenated.

    Each cell histogram is L1-normalised, so the vector keeps *where* colors occur.
    """
    img = cv2.cvtColor(np.array(pil_image.convert("RGB").resize((192, 192))), cv2.COLOR_RGB2HSV)
    h, w = img.shape[:2]
    feats = []
    for gy in range(grid):
        for gx in range(grid):
            cell = img[gy * h // grid:(gy + 1) * h // grid, gx * w // grid:(gx + 1) * w // grid]
            hist = cv2.calcHist([cell], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256]).ravel()
            feats.append(hist / max(hist.sum(), 1e-9))
    vec = np.concatenate(feats).astype("float32")
    return vec / max(np.linalg.norm(vec), 1e-9)


def open_image(path_or_file):
    return Image.open(path_or_file).convert("RGB")
