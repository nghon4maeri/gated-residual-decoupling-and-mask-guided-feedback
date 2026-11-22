"""
Clinical Domain-Specific & Geometric/Photometric Augmentation Suite (Section 4).

Includes:
  - RandomVignettingTransform: Circular optical lens vignetting
  - SyntheticHairTransform: Curved dark/light hair strands occluding skin borders
  - SyntheticRulerMarkTransform: Dermatoscope millimeter scale tick marks
  - SpecularGlareTransform: Immersion fluid / mucosal specular reflection spots
  - ImageOnlyCutoutTransform: Context occlusion keeping ground-truth mask intact
  - build_train_transforms / build_eval_transforms: Albumentations pipelines
"""

import random
import cv2
import numpy as np
import albumentations as A


class RandomVignettingTransform:
    """Simulates circular dermoscopic/endoscopic optical lens vignetting."""

    def __init__(self, p: float = 0.25, min_radius: float = 0.50, max_radius: float = 0.80):
        self.p = p
        self.min_radius = min_radius
        self.max_radius = max_radius

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image
        H, W = image.shape[:2]
        cx = random.uniform(W * 0.42, W * 0.58)
        cy = random.uniform(H * 0.42, H * 0.58)
        radius = random.uniform(self.min_radius, self.max_radius) * min(H, W)
        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        sigma = max(radius * 0.18, 1.0)
        mask = np.clip(1.0 - 1.0 / (1.0 + np.exp(-(dist - radius) / sigma)), 0.08, 1.0)
        if image.ndim == 3:
            mask = mask[..., None]
        return (image * mask).astype(np.uint8)


class SyntheticHairTransform:
    """Simulates curved dark and light hair strands occluding lesion margins."""

    def __init__(self, p: float = 0.30, max_hairs: int = 5):
        self.p = p
        self.max_hairs = max_hairs

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image
        H, W = image.shape[:2]
        img_aug = image.copy()
        for _ in range(random.randint(1, self.max_hairs)):
            x0, y0 = random.randint(0, W), random.randint(0, H)
            x1, y1 = random.randint(0, W), random.randint(0, H)
            cx, cy = random.randint(0, W), random.randint(0, H)
            pts = np.array(
                [
                    (
                        int((1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1),
                        int((1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1),
                    )
                    for t in np.linspace(0, 1, 30)
                ],
                np.int32,
            ).reshape((-1, 1, 2))
            if random.random() < 0.8:
                c = random.randint(10, 45)
                color = (c, max(0, c - 5), max(0, c - 10))
            else:
                c = random.randint(180, 230)
                color = (c, c, c)
            cv2.polylines(
                img_aug, [pts], isClosed=False, color=color, thickness=random.randint(1, 2), lineType=cv2.LINE_AA
            )
        return img_aug


class SyntheticRulerMarkTransform:
    """Simulates dermatoscope millimeter ruler tick marks near image borders."""

    def __init__(self, p: float = 0.15):
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image
        H, W = image.shape[:2]
        img_aug = image.copy()
        step = random.randint(12, 24)
        color = (random.randint(15, 50),) * 3
        if random.random() < 0.5:
            y_base = random.choice([random.randint(5, int(H * 0.15)), random.randint(int(H * 0.85), H - 5)])
            for i, x in enumerate(range(10, W - 10, step)):
                tick_h = 14 if i % 5 == 0 else 7
                cv2.line(img_aug, (x, y_base), (x, min(H - 1, y_base + tick_h)), color, thickness=random.randint(1, 2))
        else:
            x_base = random.choice([random.randint(5, int(W * 0.15)), random.randint(int(W * 0.85), W - 5)])
            for i, y in enumerate(range(10, H - 10, step)):
                tick_w = 14 if i % 5 == 0 else 7
                cv2.line(img_aug, (x_base, y), (min(W - 1, x_base + tick_w), y), color, thickness=random.randint(1, 2))
        return img_aug


class SpecularGlareTransform:
    """Simulates bright specular glare reflections from immersion fluid or mucosal folds."""

    def __init__(self, p: float = 0.15, max_spots: int = 4):
        self.p = p
        self.max_spots = max_spots

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image
        H, W = image.shape[:2]
        img_aug = image.copy()
        for _ in range(random.randint(1, self.max_spots)):
            cx = random.randint(int(W * 0.15), int(W * 0.85))
            cy = random.randint(int(H * 0.15), int(H * 0.85))
            rx, ry = random.randint(3, 10), random.randint(2, 7)
            overlay = img_aug.copy()
            cv2.ellipse(overlay, (cx, cy), (rx, ry), random.randint(0, 180), 0, 360, (245, 245, 245), -1)
            alpha = random.uniform(0.4, 0.75)
            img_aug = cv2.addWeighted(overlay, alpha, img_aug, 1.0 - alpha, 0)
        return img_aug


class ImageOnlyCutoutTransform:
    """Applies small rectangular occlusions to the RGB image while keeping the GT mask intact."""

    def __init__(self, p: float = 0.25, max_holes: int = 4, max_size: int = 28):
        self.p = p
        self.max_holes = max_holes
        self.max_size = max_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image
        H, W = image.shape[:2]
        img_aug = image.copy()
        mean_color = img_aug.mean(axis=(0, 1)).astype(np.uint8).tolist()
        for _ in range(random.randint(1, self.max_holes)):
            h, w = random.randint(8, self.max_size), random.randint(8, self.max_size)
            y0 = random.randint(0, max(1, H - h))
            x0 = random.randint(0, max(1, W - w))
            img_aug[y0 : y0 + h, x0 : x0 + w] = mean_color
        return img_aug


def build_train_transforms(img_size: int = 384) -> A.Compose:
    """Standard training geometric, elastic, and photometric augmentation pipeline (Section 4)."""
    return A.Compose(
        [
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.75, 1.0), p=0.5),
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.88, 1.12),
                translate_percent=(-0.06, 0.06),
                rotate=(-30, 30),
                p=0.6,
            ),
            A.OneOf(
                [
                    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.06, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=1.0),
                    A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=1.0),
                ],
                p=0.45,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_eval_transforms(img_size: int = 384) -> A.Compose:
    """Deterministic validation/test resize and ImageNet normalization pipeline."""
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
