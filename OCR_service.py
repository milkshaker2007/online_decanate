import re
from functools import lru_cache
from pathlib import Path

import fitz
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

from extract_fields import extract_fields  # файл с функцией извлечения полей

# Порог для объединения фрагментов в одну строку: доля от медианной высоты текста.
# Если строки рвутся — увеличьте (0.7), если соседние строки склеиваются — уменьшите (0.3).
LINE_MERGE_RATIO = 0.5

# Минимальная длина текстового слоя PDF, при которой OCR всей страницы не нужен.
MIN_PDF_TEXT_LENGTH = 30

# Встроенные в PDF картинки (штамп ЭП, печать, подпись) распознаются отдельно,
# потому что в текстовый слой они не попадают.
OCR_PDF_IMAGES = True
MIN_IMAGE_WIDTH = 40      # пункты PDF; всё мельче считаем мусором
MIN_IMAGE_HEIGHT = 20
MAX_IMAGE_PAGE_SHARE = 0.6  # картинки больше 60% страницы — это скан, а не штамп
MAX_IMAGES_PER_PAGE = 10

# Если в PDF есть настоящая цифровая подпись (поле подписи), добавляем эту строку в текст,
# чтобы extract_fields засчитал ЭП, даже когда штамп на странице не виден.
PDF_SIGNATURE_MARKER = "ДОКУМЕНТ ПОДПИСАН ЭЛЕКТРОННОЙ ПОДПИСЬЮ"


@lru_cache(maxsize=1)
def get_ocr():
    """Создаёт OCR один раз при первом использовании."""
    return PaddleOCR(
        lang="ru",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _lines_from_boxes(texts, boxes):
    """Склеивает фрагменты OCR, лежащие на одной горизонтали, в строки."""
    items = []
    for text, box in zip(texts, boxes):
        text = str(text).strip()
        if not text:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        items.append(
            {
                "text": text,
                "x1": x1,
                "cy": (y1 + y2) / 2,
                "h": max(y2 - y1, 1.0),
            }
        )

    if not items:
        return []

    items.sort(key=lambda i: i["cy"])
    median_h = sorted(i["h"] for i in items)[len(items) // 2]

    rows = []
    for item in items:
        # фрагмент относится к текущей строке, если центры почти совпадают по вертикали
        if rows and abs(item["cy"] - rows[-1]["cy"]) < median_h * LINE_MERGE_RATIO:
            row = rows[-1]
            row["items"].append(item)
            row["cy"] = sum(i["cy"] for i in row["items"]) / len(row["items"])
        else:
            rows.append({"cy": item["cy"], "items": [item]})

    return [
        " ".join(i["text"] for i in sorted(row["items"], key=lambda i: i["x1"]))
        for row in rows
    ]


def recognize_image(image):
    """Распознаёт текст на изображении."""
    image_array = np.array(image.convert("RGB"))
    results = get_ocr().predict(image_array)

    lines = []
    for result in results:
        data = result.json
        if callable(data):
            data = data()
        if isinstance(data, dict) and "res" in data:
            data = data["res"]
        if not isinstance(data, dict):
            continue

        texts = data.get("rec_texts", [])
        boxes = data.get("rec_boxes")
        if boxes is None or len(boxes) != len(texts):
            lines.extend(texts)  # запасной вариант без координат
        else:
            lines.extend(_lines_from_boxes(texts, boxes))

    return "\n".join(lines)


def _pdf_page_text(page):
    """Текст страницы PDF без лишних переносов внутри абзацев."""
    blocks = page.get_text("blocks", sort=True)
    paragraphs = []
    for block in blocks:
        if block[6] != 0:  # 0 = текстовый блок, остальное — картинки
            continue
        text = block[4]
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # слова, перенесённые через дефис
        text = re.sub(r"\s*\n\s*", " ", text).strip()  # переносы внутри блока -> пробел
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _pdf_images_text(page):
    """Распознаёт встроенные картинки страницы (штамп ЭП, печать), которых нет в текстовом слое."""
    page_area = max(page.rect.width * page.rect.height, 1.0)
    texts = []
    for info in page.get_image_info()[:MAX_IMAGES_PER_PAGE]:
        bbox = fitz.Rect(info["bbox"]) & page.rect
        if bbox.is_empty or bbox.width < MIN_IMAGE_WIDTH or bbox.height < MIN_IMAGE_HEIGHT:
            continue
        if bbox.width * bbox.height > page_area * MAX_IMAGE_PAGE_SHARE:
            continue  # скан всей страницы: текст из него уже есть в текстовом слое

        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=bbox, alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        text = recognize_image(image).strip()
        if text:
            texts.append(text)
    return "\n".join(texts)


def _pdf_has_digital_signature(document):
    """True, если в PDF есть цифровая подпись (поле подписи с подписью)."""
    try:
        flags = document.get_sigflags()
    except Exception:
        return False
    # -1 = нет AcroForm, 0 = полей подписи нет, 1 и 3 = подпись существует
    return flags > 0 and bool(flags & 1)


def extract_text(file_path):
    """Извлекает текст из JPG, PNG или PDF."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")

    extension = path.suffix.lower()

    if extension in (".jpg", ".jpeg", ".png"):
        with Image.open(path) as image:
            return recognize_image(image)

    if extension == ".pdf":
        pages_text = []
        with fitz.open(path) as document:
            for page in document:
                # Если в PDF есть текстовый слой, используем его вместо OCR
                text = _pdf_page_text(page).strip()
                if len(text) >= MIN_PDF_TEXT_LENGTH:
                    if OCR_PDF_IMAGES:
                        # штамп ЭП обычно картинка, в текстовом слое его нет
                        images_text = _pdf_images_text(page)
                        if images_text:
                            text = text + "\n" + images_text
                    pages_text.append(text)
                    continue

                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                pages_text.append(recognize_image(image))

            if _pdf_has_digital_signature(document):
                pages_text.append(PDF_SIGNATURE_MARKER)

        return "\n\n".join(pages_text)

    raise ValueError("Поддерживаются только JPG, JPEG, PNG и PDF")


def process_document(file_path, doc_type=None):
    """
    Файл (JPG/PNG/PDF) -> словарь с полями документа, готовый для RAG-модуля.
    doc_type: 'study' | 'leave' | 'practice' | None (определится автоматически).
    """
    text = extract_text(file_path)
    return extract_fields(text, doc_type=doc_type)


