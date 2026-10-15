import re  
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

# Глобальный оперативный кэш для сохранения контекста документа в рамках сессии чата
LAST_EXTRACTED = {}


# --- 4. КЛАСС ОРКЕСТРАТОРА С ИНТЕЛЛЕКТУАЛЬНОЙ МАРШРУТИЗАЦИЕЙ ВОПРОСОВ ---
class BackendDispatcher:
    def __init__(self, data_dir, model_path):
        self.kb = KnowledgeBase(data_dir=data_dir)
        self.search_engine = SearchEngine()
        self.llm_worker = LlmWorker(model_path=model_path)
        
        # Считываем чанки из базы знаний (Rules.md)
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
        # Очищаем и берем свежий текстовый вопрос из текущего POST-пакета
        raw_question = user_question.strip() if user_question else ""
        doc_type = extracted_fields.get("тип_документа", "")
        
        # Программный аудит заполнения обязательных полей (ПАТЧ СТРОГОСТИ)
        missing_fields = []
        for k, v in extracted_fields.items():
            if k == "тип_документа":
                continue
            val_str = str(v).strip()
            
            # Сверхнадежный поиск пустого шаблона через импортированную библиотеку re
            if k == "Номер и дата выдачи":
                if re.search(r"^[\s ]*№?\s*от\s+\d", val_str.lower()) or val_str.lower() == "№" or val_str == "":
                    extracted_fields[k] = "не найдено"
                    val_str = "не найдено"
            
            if val_str.lower() == "не найдено":
                missing_fields.append(k)
                    
        is_empty_or_greet = not raw_question or raw_question.lower() in [
            "привет", "здравствуйте", "ку", "приветик", "добрый день", "добрый вечер", "дорова"
        ]
        
        # Точечный подбор поискового запроса для BM25, чтобы разгрузить контекст модели
        if is_empty_or_greet:
            if "Форма обучения" in extracted_fields:
                search_query = "Справка об обучении обязательные текстовые поля требования регламент"
            elif "Причина отпуска" in extracted_fields or "Дата начала" in extracted_fields or "Заявление" in doc_type:
                search_query = "Заявление на академический отпуск обязательные поля требования правила"
            else:
                search_query = "Общие правила проверки студенческих документов регламент деканата"
        else:
            search_query = raw_question

        # Предотвращаем раздувание промпта. Сюда складываем фолбэк-контекст на случай ошибки BM25
        paragraph_text = "\n\n".join([f"Источник {c['source']}:\n{c['text']}" for c in self.chunks])
        paragraph_source = "Сводное Положение деканата"
        
        try:
            search_results = self.search_engine.search(search_query, top_n=1)
            if search_results and len(search_results) > 0:
                best_chunk, score = search_results
                paragraph_source = best_chunk["source"]
                # Передаем в модель строго один отфильтрованный поисковиком чанк
                paragraph_text = f"Источник {best_chunk['source']}:\n{best_chunk['text']}"
        except Exception as e:
            print(f"[ИИ-ИНФО] Ошибка поиска BM25: {e}")

        # Передаем точечный контекст, вопрос и список force_missing в llm_worker
        return self.llm_worker.evaluate_document(
            extracted_fields=extracted_fields,
            paragraph_text=paragraph_text,
            paragraph_source=paragraph_source,
            user_question="Автоматическая проверка регламента" if is_empty_or_greet else raw_question,
            force_missing=missing_fields
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
    edited_fields_json: str = Form("")  
):
    """Шлюз асинхронной обработки файлов и RAG-генерации."""
    global LAST_EXTRACTED
    filename = getattr(file, "filename", "temp_doc.png") or "temp_doc.png"
    temp_file_path = TEMP_DIR / filename
    
    try:
        file_bytes = await file.read()
        
        # Умное определение текстового вопроса по названию фейковых файлов с фронтенда
        is_text_question = "empty_query.pdf" in filename or "edited_query.pdf" in filename or len(file_bytes) == 0
        
        # Сценарий A: Ручное редактирование полей в модальном окне фронтенда
        if edited_fields_json:
            ocr_extracted_fields = json.loads(edited_fields_json)
            LAST_EXTRACTED = ocr_extracted_fields  # Синхронизируем оперативную память
            
        # Сценарий B: Пользователь задает текстовый вопрос (ИСПРАВЛЕНО И ЖЕСТКО ЗАФИКСИРОВАНО)
        elif is_text_question:
            # ТЗ: Текстовый вопрос без отправки реального нового файла ВСЕГДА 
            # сбрасывает контекст и требует прикрепить документ. Старый кэш игнорируется.
            return {
                "ocr_extracted_fields": {"тип_документа": "не найдено"},
                "rag_analysis": {
                    "status": "Сценарий В",
                    "paragraph": "Общий регламент",
                    "final_answer": f'Я принял ваш вопрос: "{question}". Чтобы проверить документ на соответствие регламентам деканата, пожалуйста, прикрепите файл (скан справки или заявления) через меню скрепки.'
                }
            }
                
        # Сценарий C: Пользователь прикрепил новый реальный файл (картинку или PDF)
        else:
            with open(temp_file_path, "wb") as buffer:
                buffer.write(file_bytes)
            
            # Чистый вызов OCR-сервиса
            ocr_extracted_fields = process_document(str(temp_file_path))
            LAST_EXTRACTED = ocr_extracted_fields

        # Генерируем интеллектуальный RAG-ответ (вызывается только при отправке реального файла или правок)
        rag_result = backend.generate_answer(ocr_extracted_fields, user_question=question)

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
            "ocr_extracted_fields": {"Ошибка": "Не удалось корректно распарсить реквизиты документа"},
            "rag_analysis": {
                "status": "Сценарий В",
                "paragraph": "п. 1.1",
                "final_answer": f"При обработке документа на сервере произошел сбой: {str(e)}"
            }
        }
    finally:
        if temp_file_path.exists():
            os.remove(temp_file_path)
