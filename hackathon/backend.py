import os
import sys
import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# --- 1. АБСОЛЮТНАЯ КОНФИГУРАЦИЯ ПУТЕЙ ПРОЕКТА ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = str(BASE_DIR / "data")
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# --- 2. КОНФИГУРАЦИЯ ПУТИ К МОДЕЛИ ---
MODEL_PATH = str(BASE_DIR / "model" / "qwen2.5-1.5b-instruct-q4_k_m.gguf")

# --- 3. ПОДКЛЮЧЕНИЕ ПАПОК ПРОЕКТА С СИСТЕМНОГО ДИСКА ---
sys.path.append(str(BASE_DIR / "online_decanate-ocr"))
sys.path.append(str(BASE_DIR / "online_decanate-rag"))

from OCR_service import process_document
from knowledge_base import KnowledgeBase
from search_engine import SearchEngine
from llm_worker import LlmWorker


# --- 4. КЛАСС ОРКЕСТРАТОРА С ИСПРАВЛЕННЫМ ИНДЕКСОМ ПОИСКА ---
class BackendDispatcher:
    def __init__(self, data_dir, model_path):
        self.kb = KnowledgeBase(data_dir=data_dir)
        self.search_engine = SearchEngine()
        self.llm_worker = LlmWorker(model_path=model_path)
        
        # Считываем чанки из базы знаний (обрабатывает rules.md)
        self.chunks = self.kb.load_and_parse()
        
        # КРИТИЧЕСКИЙ ФОЛБЭК: Если парсер вернул пустоту, наполняем массив вручную
        if not self.chunks:
            print("\n[ИИ-ИНФО] База знаний пуста. Загружаю эталонные фолбэк-требования...")
            self.chunks = [
                {
                    "text": "Справка об обучении должна содержать обязательные текстовые реквизиты: ФИО студента, Название вуза, Направление подготовки, Форма обучения, Период обучения, Номер и дату выдачи, а также Подпись и печать.",
                    "source": "п. 3.1 Положения деканата"
                },
                {
                    "text": "Заявление на академический отпуск обязательные поля и требования: ФИО студента, Группа и курс, Причина отпуска, Дата начала отпуска, Дата окончания отпуска, Дата заявления, Подпись студента.",
                    "source": "п. 4.1 Положения деканата"
                },
                {
                    "text": "Справка о практике обязательные реквизиты: ФИО студента, Организация, Сроки практики, Руководитель практики, Оценка, Номер и дату выдачи, Подпись и печать.",
                    "source": "п. 5.1 Положения деканата"
                }
            ]
        
        self.search_engine.build_index(self.chunks)

    def generate_answer(self, extracted_fields, user_question):
        search_query = user_question.strip() if user_question else ""
        if not search_query:
            if "Форма обучения" in extracted_fields:
                search_query = "Справка об обучении обязательные текстовые поля требования"
            elif "Причина отпуска" in extracted_fields:
                search_query = "Заявление на академический отпуск обязательные поля требования"
            else:
                search_query = "Общие правила проверки студенческих документов"

        paragraph_text = ""
        paragraph_source = ""
        try:
            search_results = self.search_engine.search(search_query, top_n=1)
            # Исправлено: извлекаем первый элемент списка результатов [0] перед распаковкой кортежа
            if search_results and len(search_results) > 0:
                best_chunk, score = search_results[0]
                paragraph_text = best_chunk["text"]
                paragraph_source = best_chunk["source"]
            else:
                raise ValueError("Пустой результат поиска")
        except Exception as e:
            print(f"[ИИ-ИНФО] Поисковый индекс перенаправлен на фолбэк: {e}")
            if "Форма обучения" in extracted_fields:
                paragraph_text = self.chunks[0]["text"]
                paragraph_source = self.chunks[0]["source"]
            elif "Причина отпуска" in extracted_fields:
                paragraph_text = self.chunks[1]["text"]
                paragraph_source = self.chunks[1]["source"]
            else:
                paragraph_text = self.chunks[2]["text"]
                paragraph_source = self.chunks[2]["source"]

        # Передаем контекст и распознанные OCR-поля в модель Qwen
        return self.llm_worker.evaluate_document(
            extracted_fields=extracted_fields,
            paragraph_text=paragraph_text,
            paragraph_source=paragraph_source
        )


# --- 5. ЗАПУСК FASTAPI СЕРВЕРА ---
app = FastAPI(title="Онлайн Деканат: Бэкенд-Диспетчер")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

backend = BackendDispatcher(data_dir=DATA_DIR, model_path=MODEL_PATH)


@app.get("/")
async def get_ui():
    """Раздача статичной страницы интерфейса."""
    html_path = BASE_DIR / "app.html"
    if html_path.exists():
        return FileResponse(html_path)
    return {"error": "Файл app.html не найден в корневой директории"}


@app.post("/analyze")
async def analyze_request(
    file: UploadFile = File(...),
    question: str = Form(""),
    edited_fields_json: str = Form("")  # Поле для прямой передачи JSON при ручном редактировании
):
    """Шлюз асинхронной обработки файлов и RAG-генерации."""
    filename = getattr(file, "filename", "temp_doc.png") or "temp_doc.png"
    temp_file_path = TEMP_DIR / filename
    
    try:
        file_bytes = await file.read()
        
        # Сценарий повторной валидации измененной карточки (когда файл пустой, но пришел JSON)
        if len(file_bytes) == 0 and edited_fields_json:
            ocr_extracted_fields = json.loads(edited_fields_json)
        else:
            # Сценарий обработки реальной загруженной фотографии документа
            with open(temp_file_path, "wb") as buffer:
                buffer.write(file_bytes)
            # Вызов OCR-сервиса для анализа изображения
            ocr_extracted_fields = process_document(str(temp_file_path))
        
        # Вызов RAG-генерации вердикта локальной моделью Qwen2.5
        rag_result = backend.generate_answer(ocr_extracted_fields, question)

        return {
            "ocr_extracted_fields": ocr_extracted_fields,
            "rag_analysis": {
                "status": rag_result["status"],
                "paragraph": rag_result["paragraph"],
                "final_answer": rag_result["final_answer"]
            }
        }

    except Exception as e:
        print(f"\n[ВНУТРЕННЯЯ ОШИБКА СЕРВЕРА]: {e}\n")
        return {
            "ocr_extracted_fields": {"Ошибка": "Не удалось корректно распарсить реквизиты"},
            "rag_analysis": {
                "status": "Сценарий В",
                "paragraph": "п. 1.1",
                "final_answer": f"При обработке документа на сервере произошел сбой: {str(e)}"
            }
        }
    finally:
        if temp_file_path.exists():
            os.remove(temp_file_path)
