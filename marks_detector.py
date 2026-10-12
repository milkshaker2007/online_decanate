
from pathlib import Path

import cv2
try:
    import pymupdf as fitz  # PyMuPDF >= 1.24.3
except ImportError:
    import fitz
import numpy as np
from PIL import Image

MAX_SIDE = 1600  # приводим изображение к единому масштабу, чтобы пороги работали одинаково


def _to_bgr(image):
    """PIL.Image / numpy -> BGR numpy, уменьшенный до MAX_SIDE."""
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        bgr = image
    h, w = bgr.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return bgr


def _ink_mask(bgr):
    """Маска цветных чернил: синие/фиолетовые и красные оттенки."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    blue = (h >= 88) & (h <= 150) & (s >= 60) & (v >= 90)   # v: отсекаем тёмный (чёрный) текст с синеватым оттенком
    red = ((h <= 8) | (h >= 168)) & (s >= 100) & (v >= 100)
    mask = ((blue | red).astype(np.uint8)) * 255
    # убираем одиночные пиксели-шум
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask



def _find_round_stamp(mask, short_side):
    """Круглая печать по кольцу (Хаф): работает, даже если подпись пересекает печать
    и сливается с ней в один объект."""
    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=short_side * 0.2,
        param1=100, param2=30,
        minRadius=int(0.06 * short_side), maxRadius=int(0.30 * short_side))
    if circles is None:
        return []
    found = []
    for cx, cy, r in circles[0][:5]:
        # какая доля окружности (±3 px) покрыта чернилами
        hits = 0
        for ang in np.linspace(0, 2 * np.pi, 120, endpoint=False):
            for dr in (-3, -2, -1, 0, 1, 2, 3):
                px = int(round(cx + (r + dr) * np.cos(ang)))
                py = int(round(cy + (r + dr) * np.sin(ang)))
                if 0 <= px < mask.shape[1] and 0 <= py < mask.shape[0] and mask[py, px]:
                    hits += 1
                    break
        if hits / 120.0 >= 0.55:
            found.append((int(cx - r), int(cy - r), int(2 * r), int(2 * r)))
    return found


def analyze_image(image):
    """
    Возвращает {"stamp": bool, "signature": bool} для одного изображения.
    """
    bgr = _to_bgr(image)
    H, W = bgr.shape[:2]
    result = {"stamp": False, "signature": False}

    mask = _ink_mask(bgr)

    # если «цветным» оказалось слишком много - скорее всего цветовой сдвиг
    # фото (жёлтый свет, синеватая тень), а не чернила; ничего не утверждаем
    if cv2.countNonZero(mask) > 0.30 * H * W:
        return result

    k = max(3, int(0.006 * max(H, W)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    short_side = min(H, W)

    stamps = []   # bbox найденных печатей
    others = []   # остальные цветные объекты (кандидаты в подписи)

    for i in range(1, n):
        x, y, w, h, _area = stats[i]
        ink = cv2.countNonZero(mask[y:y + h, x:x + w] & ((labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255))
        if ink < 0.0004 * H * W:        # слишком мало чернил
            continue
        density = ink / float(w * h)    # доля чернил в рамке объекта
        aspect = w / float(h)

        is_stamp = (min(w, h) >= 0.10 * short_side
                    and 0.6 <= aspect <= 1.6
                    and 0.03 <= density <= 0.65)
        if is_stamp:
            stamps.append((x, y, w, h))
        else:
            others.append((x, y, w, h, density, aspect))

    stamps += _find_round_stamp(mask, short_side)
    result["stamp"] = bool(stamps)

    for (x, y, w, h, density, aspect) in others:
        # объекты внутри печати (её надписи, звезда) - это не подпись
        cx, cy = x + w / 2.0, y + h / 2.0
        if any(sx <= cx <= sx + sw and sy <= cy <= sy + sh for sx, sy, sw, sh in stamps):
            continue
        looks_like_signature = (
            w >= 0.04 * W
            and h >= 0.012 * H
            and 0.7 <= aspect <= 10
            and 0.02 <= density <= 0.5
        )
        if looks_like_signature:
            result["signature"] = True
            break

    return result


def _render_pdf(path):
    images = []
    with fitz.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return images


def detect_marks(file_path):
    """
    Файл (JPG/PNG/PDF) -> {"stamp": bool, "signature": bool}.
    Для PDF смотрим все страницы: достаточно найти на любой.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in (".jpg", ".jpeg", ".png"):
        with Image.open(path) as img:
            return analyze_image(img)

    if ext == ".pdf":
        total = {"stamp": False, "signature": False}
        for img in _render_pdf(path):
            r = analyze_image(img)
            total["stamp"] |= r["stamp"]
            total["signature"] |= r["signature"]
        return total

    raise ValueError("Поддерживаются только JPG, JPEG, PNG и PDF")