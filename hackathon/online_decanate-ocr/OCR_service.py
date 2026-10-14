import re
from functools import lru_cache
from pathlib import Path

import fitz
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

from extract_fields import extract_fields  
from marks_detector import detect_marks   

# Порог для объединения фрагментов в одну строку: доля от медианной высоты текста.
LINE_MERGE_RATIO = 0.5

MIN_PDF_TEXT_LENGTH = 30

# Встроенные в PDF картинки (штамп ЭП, печать, подпись) распознаются отдельно
OCR_PDF_IMAGES = True
MIN_IMAGE_WIDTH = 40      
MIN_IMAGE_HEIGHT = 20
MAX_IMAGE_PAGE_SHARE = 0.6  
MAX_IMAGES_PER_PAGE = 10

PDF_SIGNATURE_MARKER = "ДОКУМЕНТ ПОДПИСАН ЭЛЕКТРОННОЙ ПОДПИСЬЮ"


@lru_cache(maxsize=1)
def get_ocr():
    """Создаёт OCR один раз при первом использовании."""
    # Переносим управление классификацией угла (cls) прямо в конструктор
    return PaddleOCR(
        lang="ru",
        use_angle_cls=False  # это и есть аналог cls=False на этапе инициализации
    )


def _lines_from_boxes(texts, boxes):
    """Склеивает фрагменты OCR, лежащие на одной горизонтали, в строки."""
    if not boxes or not texts:
        return []

    items = []
    for text, box in zip(texts, boxes):
        text = str(text).strip()
        if not text:
            continue
        
        # Исправлено: преобразуем в numpy-массив и строго проверяем, что он не пустой
        box_np = np.array(box)
        if box_np.size == 0 or len(box_np.shape) < 2:
            continue
            
        x1 = float(np.min(box_np[:, 0]))
        y1 = float(np.min(box_np[:, 1]))
        x2 = float(np.max(box_np[:, 0]))
        y2 = float(np.max(box_np[:, 1]))
        
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
    results = get_ocr().ocr(image_array)

    # Исправлено: строгая проверка на случай, если на фото текст не обнаружен
    if not results or results[0] is None:
        return ""

    lines = []
    for page_res in results:
        if not page_res:
            continue
        texts = []
        boxes = []
        for res in page_res:
            # Защита от пустых или неполных структур со стороны нейросети
            if not res or len(res) < 2:
                continue
            box = res[0]
            text_data = res[1][0] # Текст находится в первом элементе кортежа (текст, уверенность)
            texts.append(text_data)
            boxes.append(box)
        
        if boxes and len(boxes) == len(texts):
            lines.extend(_lines_from_boxes(texts, boxes))
        else:
            lines.extend(texts)

    return "\n".join(lines)


def _pdf_page_text(page):
    """Текст страницы PDF без лишних переносов внутри абзацев."""
    blocks = page.get_text("blocks", sort=True)
    paragraphs = []
    for block in blocks:
        if block[6] != 0:  # 0 = текстовый block, остальное — картинки
            continue
        text = block[4]
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  
        text = re.sub(r"\s*\n\s*", " ", text).strip()  
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _pdf_images_text(page):
    """Распознаёт встроенные картинки страницы (штамп ЭП, печать)."""
    page_area = max(page.rect.width * page.rect.height, 1.0)
    texts = []
    for info in page.get_image_info()[:MAX_IMAGES_PER_PAGE]:
        bbox = fitz.Rect(info["bbox"]) & page.rect
        if bbox.is_empty or bbox.width < MIN_IMAGE_WIDTH or bbox.height < MIN_IMAGE_HEIGHT:
            continue
        if bbox.width * bbox.height > page_area * MAX_IMAGE_PAGE_SHARE:
            continue  

        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=bbox, alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        text = recognize_image(image).strip()
        if text:
            texts.append(text)
    return "\n".join(texts)


def _pdf_has_digital_signature(document):
    """True, если в PDF есть цифровая подпись."""
    try:
        flags = document.get_sigflags()
    except Exception:
        return False
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
                text = _pdf_page_text(page).strip()
                if len(text) >= MIN_PDF_TEXT_LENGTH:
                    if OCR_PDF_IMAGES:
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
    text = extract_text(file_path)
    marks = detect_marks(file_path)
    return extract_fields(text, doc_type=doc_type, marks=marks)
