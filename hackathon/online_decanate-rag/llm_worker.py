import os
import json
import re
from llama_cpp import Llama

class LlmWorker:
    def __init__(self, model_path: str):
        """
        Инициализирует локальную GGUF модель через llama-cpp, оптимизируя под CPU.
        """
        threads = max(1, (os.cpu_count() or 2) - 1)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,  # Расширенное окно до 4096 токенов во избежание падения по лимитам
            n_threads=threads,
            verbose=False
        )

    def evaluate_document(self, extracted_fields: dict, paragraph_text: str, paragraph_source: str, user_question: str = "", force_missing: list = None) -> dict:
        """
        Интеллектуальный анализ сопоставления вопроса, типа документа и регламента.
        """
        fields_str = "\n".join([f"- {k}: {v}" for k, v in extracted_fields.items()])
        doc_type_detected = extracted_fields.get("тип_документа", "не найдено")
        
        # Очищаем входящий вопрос для анализа триггеров
        raw_question = user_question.strip() if user_question else ""
        q_low = raw_question.lower()
        
        # 1. СЦЕНАРИЙ ОБНАРУЖЕНИЯ НЕСООТВЕТСТВИЯ ТЕМЫ ДОКУМЕНТА И ВОПРОСА
        is_cross_topic = False
        if "практик" in q_low and "отпуск" in doc_type_detected.lower():
            is_cross_topic = True
        elif ("академ" in q_low or "отпуск" in q_low) and "справк" in doc_type_detected.lower():
            is_cross_topic = True
        elif "справк" in q_low and "практик" in doc_type_detected.lower():
            is_cross_topic = True

        if is_cross_topic:
            final_text = (
                f"Запрос отклонен (Сценарий В): «Обнаружено критическое несоответствие! "
                f"Вы прикрепили документ типа '{doc_type_detected}', однако ваш текстовый запрос "
                f"относится к другой теме: '{raw_question}'. Наш ИИ-помощник не может верифицировать "
                f"документ по нецелевому регламенту. Пожалуйста, очистите поле ввода или загрузите корректный файл.»"
            )
            return {
                "status": "Сценарий В",
                "paragraph": paragraph_source,
                "final_answer": final_text
            }

        # 2. ПРОВЕРКА НА ПУСТЫЕ ПОЛЯ И СТАТУС ПОДПИСЕЙ/ПЕЧАТЕЙ
        actual_missing = []
        for k, v in extracted_fields.items():
            if k == "тип_документа":
                continue
            val_str = str(v).strip()
            if val_str.lower() == "не найдено":
                actual_missing.append(k)

        # Вытаскиваем реальные значения подписей из полей документа
        sign_status = str(extracted_fields.get("Подпись и печать", "есть")).strip().lower()
        student_sign_status = str(extracted_fields.get("Подпись студента", "есть")).strip().lower()
        
        # Проверяем, есть ли явная пометка "нет" в графе заверений
        has_validation_error = sign_status == "нет" or student_sign_status == "нет"

        # Проверяем, задал ли пользователь конкретный вопрос про значение какого-то поля
        is_asking_for_field = any(x in q_low for x in ["укажи", "значение", "какое", "кто", "что", "когда", "номер", "дата"])

        # Авто-верификация (Сценарий А) должна срабатывать ТОЛЬКО если вопроса нет вообще, 
        # либо если в нем явно просят сделать автоматическую проверку, И при этом нет точечного вопроса про поле!
        is_just_verification = (not raw_question) or ("автоматическая проверка" in q_low) or (any(x in q_low for x in ["проверка", "автомат", "верификац"]) and not is_asking_for_field)

        # Авто-верификация срабатывает ТОЛЬКО если нет пропущенных полей И подпись/печать имеют статус "есть" И это не точечный вопрос
        if not actual_missing and not has_validation_error and is_just_verification:
            final_text = f"Документ успешно верифицирован: «Загруженный документ '{doc_type_detected}' полностью удовлетворяет требованиям регламента ({paragraph_source}). Ошибок оформления и заверения не обнаружено.»"
            return {"status": "Сценарий А", "paragraph": paragraph_source, "final_answer": final_text}

        # Если обнаружена пометка "Нет" в подписи, мы принудительно формируем замечание по регламенту (Сценарий Б)
        if has_validation_error and is_just_verification:
            bad_field = "Подпись и печать" if sign_status == "нет" else "Подпись студента"
            final_text = (
                f"Ошибка валидации (Сценарий Б): «Внимание! Документ '{doc_type_detected}' не может быть принят, "
                f"так как обязательное поле заверений '{bad_field}' имеет статус 'Нет'. Согласно регламенту "
                f"({paragraph_source}), бланки без оригинальных оттисков печатей и подписей считаются недействительными.»"
            )
            return {"status": "Сценарий Б", "paragraph": paragraph_source, "final_answer": final_text}

        if actual_missing and is_just_verification:
            clean_fields = [f"'{str(x)}'" for x in actual_missing]
            missing_str = ", ".join(clean_fields)
            final_text = f"Внимание, документ не принят: «В документе '{doc_type_detected}' не хватает или не распознаны обязательные поля: {missing_str}. Это нарушает регламент {paragraph_source}.»"
            return {"status": "Сценарий Б", "paragraph": paragraph_source, "final_answer": final_text}

        # 3. ТОЧЕЧНЫЙ ДЕТЕРМИНИРОВАННЫЙ ОТВЕТ НА ВОПРОСЫ О ЗНАЧЕНИЯХ ПОЛЕЙ ДОКУМЕНТА (БЕЗ JSON / БЕЗ ИНФЕРЕНСА LLM)
        if is_asking_for_field:
            text_answer = ""
            
            # Проверяем, какой именно реквизит ищет пользователь в строке вопроса
            if any(x in q_low for x in ["фио", "имя", "студент"]):
                text_answer = extracted_fields.get("ФИО студента", "не найдено")
            elif any(x in q_low for x in ["вуз", "университет", "академи"]):
                text_answer = extracted_fields.get("Название вуза", "не найдено")
            elif any(x in q_low for x in ["направление", "специальност", "программ"]):
                text_answer = extracted_fields.get("Направление подготовки", "не найдено")
            elif any(x in q_low for x in ["форм", "очн", "заочн"]):
                text_answer = extracted_fields.get("Форма обучения", "не найдено")
            elif any(x in q_low for x in ["период", "срок", "обучени", "начало", "окончани"]):
                # Проверяем сквозные поля для всех 3-х видов документов
                text_answer = extracted_fields.get("Период обучения") or extracted_fields.get("Сроки практики") or f"с {extracted_fields.get('Дата начала', '—')} по {extracted_fields.get('Дата окончания', '—')}"
            elif any(x in q_low for x in ["номер", "справк", "документ"]):
                text_answer = extracted_fields.get("Номер и дата выдачи", "не найдено")
            elif any(x in q_low for x in ["подпись", "печат", "штамп", "заверен"]):
                text_answer = extracted_fields.get("Подпись и печать") or extracted_fields.get("Подпись студента", "не найдено")
            elif any(x in q_low for x in ["причин", "основани"]):
                text_answer = extracted_fields.get("Причина отпуска", "не найдено")
            elif any(x in q_low for x in ["групп", "курс"]):
                text_answer = extracted_fields.get("Группа и курс", "не найдено")
            elif any(x in q_low for x in ["организац", "предприят"]):
                text_answer = extracted_fields.get("Организация", "не найдено")
            elif any(x in q_low for x in ["руководител"]):
                text_answer = extracted_fields.get("Руководитель практики", "не найдено")
            elif any(x in q_low for x in ["оценк"]):
                text_answer = extracted_fields.get("Оценка", "не найдено")
                
            if text_answer and text_answer != "не найдено":
                return {
                    "status": "Сценарий А",
                    "paragraph": paragraph_source,
                    "final_answer": f"Значение реквизита: «{text_answer}»"
                }

        # 4. СТАНДАРТНЫЙ ЛОГИЧЕСКИЙ АУДИТ ЧЕРЕЗ JSON (Для сложных неструктурированных вопросов)
        system_prompt = (
            "Ты — опытный ИИ-ассистент онлайн-деканата вуза. Твоя задача — сопоставить ВОПРОС ПОЛЬЗОВАТЕЛЯ, ТИП ДОКУМЕНТА и ТЕКСТ РЕГЛАМЕНТА.\n\n"
            "Отвечай строго в формате JSON: {\"status\": \"...\", \"missing_fields\": [], \"reason\": \"...\"}"
        )

        user_content = (
            f"ВОПРОС ПОЛЬЗОВАТЕЛЯ: {user_question}\n"
            f"ТИП ЗАГРУЖЕННОГО ДОКУМЕНТА: {doc_type_detected}\n"
            f"ИЗВЛЕЧЕННЫЕ ИЗ ДОКУМЕНТА РЕКВИЗИТЫ:\n{fields_str}\n"
            f"АКТУАЛЬНЫЙ ТЕКСТ РЕГЛАМЕНТА:\n{paragraph_text}\n"
        )

        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{{"

        response_text = ""
        try:
            response = self.llm(prompt, max_tokens=256, temperature=0.1, top_p=0.9, stop=["<|im_end|>", "</s>"])
            response_text = response["choices"]["text"].strip()
            if not response_text.startswith("{"):
                response_text = "{" + response_text
                
            verdict = json.loads(response_text)
            status = verdict.get("status", "Сценарий В")
            reason_str = verdict.get("reason", "несоответствие регламенту")
        except Exception:
            status = "Сценарий В"
            reason_str = "Требуется дополнительная проверка полей документа сотрудником деканата"
        
        if status == "Сценарий А":
            final_text = f"Документ успешно верифицирован: «Документ '{doc_type_detected}' полностью удовлетворяет требованиям регламента ({paragraph_source}). Ошибок оформления не обнаружено.»"
        else:
            final_text = f"Ошибка валидации: «{reason_str}. Рекомендуется проверить данные через кнопку 'Изменить вручную'.»"

        return {"status": status, "paragraph": paragraph_source, "final_answer": final_text}
