# Multi-item food detection for a single fridge photo.
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from model import FoodCNN

DEFAULT_CHECKPOINT = Path(__file__).parent / "artifacts" / "best_model.pth"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FoodDetector:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device=None, threshold=None,
                 grid=2, crop_weight=0.9):
        self.device = torch.device(device) if device else pick_device()
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.classes = ckpt["classes"]
        self.size = ckpt["img_size"]
        self.mean = np.array(ckpt["mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(ckpt["std"], dtype=np.float32).reshape(3, 1, 1)
        self.thresholds = (np.full(len(self.classes), threshold, dtype=np.float32)
                           if threshold is not None
                           else np.array(ckpt["thresholds"], dtype=np.float32))
        self.grid, self.crop_weight = grid, crop_weight

        self.net = FoodCNN(ckpt["num_classes"])
        self.net.load_state_dict(ckpt["model_state"])
        self.net.to(self.device).eval()

    # preprocessing
    @staticmethod
    def _load(image):
        if isinstance(image, Image.Image):
            img = image
        elif isinstance(image, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image))
        else:
            img = Image.open(Path(image))
        return ImageOps.exif_transpose(img).convert("RGB")  # as phone photos carry rotation in EXIF

    def _to_array(self, img):
        a = np.asarray(img.resize((self.size, self.size), Image.BILINEAR), dtype=np.float32) / 255.0
        return (a.transpose(2, 0, 1) - self.mean) / self.std

    def _crops(self, img):
        # n x n grid of crops, each ~60% of the image per side, evenly spread (overlapping).
        n, (w, h) = self.grid, img.size
        cw, ch = int(w * 0.6), int(h * 0.6)
        xs = np.linspace(0, w - cw, n).astype(int)
        ys = np.linspace(0, h - ch, n).astype(int)
        return [img.crop((x, y, x + cw, y + ch)) for y in ys for x in xs]

    # inference
    @torch.no_grad()
    def _probs(self, imgs):
        x = torch.from_numpy(np.stack([self._to_array(i) for i in imgs])).to(self.device)
        return torch.sigmoid(self.net(x)).cpu().numpy()

    def predict_probs(self, image):
        img = self._load(image)
        full = self._probs([img, ImageOps.mirror(img)]).mean(0)
        if self.grid and self.grid >= 2:
            crops = self._crops(img)
            crop_p = self._probs(crops + [ImageOps.mirror(c) for c in crops])
            full = np.maximum(full, self.crop_weight * crop_p.max(0))
        return full

    def detect(self, image, top_k=None):
        p = self.predict_probs(image)
        idx = np.where(p >= self.thresholds)[0]
        idx = idx[np.argsort(-p[idx])]
        if top_k:
            idx = idx[:top_k]
        return [{"name": self.classes[i], "confidence": round(float(p[i]), 4)} for i in idx]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Detect foods in a fridge photo")
    ap.add_argument("image")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--threshold", type=float, default=None, help="global threshold override")
    ap.add_argument("--grid", type=int, default=2, help="crop grid size (0 = full image only)")
    a = ap.parse_args()
    det = FoodDetector(a.checkpoint, threshold=a.threshold, grid=a.grid)
    print(json.dumps(det.detect(a.image), indent=2))
