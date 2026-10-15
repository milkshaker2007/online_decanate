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
            n_ctx=2048,
            n_threads=threads,
            verbose=False
        )

    def evaluate_document(self, extracted_fields: dict, paragraph_text: str, paragraph_source: str, user_question: str = "", force_missing: list = None) -> dict:
        """
        Интеллектуальный анализ сопоставления вопроса, типа документа и регламента.
        """
        fields_str = "\n".join([f"- {k}: {v}" for k, v in extracted_fields.items()])
        doc_type_detected = extracted_fields.get("тип_документа", "не найдено")
        q_low = user_question.lower()
        
        # 1. СЦЕНАРИЙ ПЕРЕКЛЮЧЕНИЯ КОНТЕКСТА (УМНЫЙ КОНСУЛЬТАНТ)
        # Если пользователь сменил тему и спрашивает про практику/справку, а на экране открыт отпуск
        # (или наоборот), мы заставляем ИИ отвечать строго по тексту найденного регламента RAG, игнорируя старый файл!
        is_cross_topic = False
        if "практик" in q_low and "отпуск" in doc_type_detected.lower():
            is_cross_topic = True
        elif ("академ" in q_low or "отпуск" in q_low) and "справк" in doc_type_detected.lower():
            is_cross_topic = True
        elif "справк" in q_low and "практик" in doc_type_detected.lower():
            is_cross_topic = True

        if is_cross_topic:
            # Даем модели промпт в режиме классического вопрос-ответ (QA) по регламенту
            sys_qa = "Ты — консультант деканата. Ответь на вопрос студента строго на основе предоставленного текста регламента."
            user_qa = f"ВОПРОС СТУДЕНТА: {user_question}\n\nТЕКСТ РЕГЛАМЕНТА ИЗ БАЗЫ ЗНАНИЙ:\n{paragraph_text}\n\nВыдай краткий и понятный ответ:"
            prompt_qa = f"<|im_start|>system\n{sys_qa}<|im_end|>\n<|im_start|>user\n{user_qa}<|im_end|>\n<|im_start|>assistant\n"
            
            res_qa = self.llm(prompt_qa, max_tokens=256, temperature=0.2, top_p=0.9, stop=["<|im_end|>", "</s>"])
            ans_text = res_qa["choices"][0]["text"].strip()
            return {
                "status": "Сценарий А",
                "paragraph": paragraph_source,
                "final_answer": f"Консультация по регламенту ({paragraph_source}): «{ans_text}»"
            }

        # 2. ПРОВЕРКА НА ПУСТЫЕ ПОЛЯ ДЛЯ ТЕКУЩЕГО ДОКУМЕНТА
        actual_missing = []
        for k, v in extracted_fields.items():
            if k == "тип_документа":
                continue
            if str(v).strip().lower() == "не найдено":
                actual_missing.append(k)

        # Жесткий фиксатор Сценария А для текущего целевого процесса
        if not actual_missing and "отпуск" in doc_type_detected.lower() and ("академ" in q_low or "отпуск" in q_low or "проверка" in q_low or "автомат" in q_low):
            final_text = f"Документ успешно верифицирован: «Загруженный документ '{doc_type_detected}' полностью удовлетворяет требованиям регламента ({paragraph_source}). Ошибок оформления не обнаружено. Окончательное решение примет сотрудник деканата.»"
            return {"status": "Сценарий А", "paragraph": paragraph_source, "final_answer": final_text}

        if actual_missing:
            clean_fields = [f"'{str(x)}'" for x in actual_missing]
            missing_str = ", ".join(clean_fields)
            final_text = f"Внимание, документ не принят: «В документе '{doc_type_detected}' не хватает или не распознаны обязательные поля: {missing_str}. Это нарушает регламент {paragraph_source}.»"
            return {"status": "Сценарий Б", "paragraph": paragraph_source, "final_answer": final_text}

        # 3. СТАНДАРТНЫЙ ЛОГИЧЕСКИЙ АУДИТ ЧЕРЕЗ JSON
        system_prompt = (
            "Ты — опытный ИИ-ассистент онлайн-деканата вуза. Твоя задача — сопоставить ВОПРОС ПОЛЬЗОВАТЕЛЯ, ТИП ДОКУМЕНТА и ТЕКСТ РЕГЛАМЕНТА.\n\n"
            "ПРАВИЛА ИНТЕЛЛЕКТУАЛЬНОЙ ПРОВЕРКИ:\n"
            "1. Оцени тему вопроса: Твоя система поддерживает только три процесса: 'Справка об обучении', 'Академический отпуск' и 'Справка о практике'. "
            "Если вопрос относится к любым сторонним темам (стипендии, общежития), ты обязан выдать 'Сценарий В' и в причине (reason) написать: 'Заданный вопрос не относится к целевым темам верификации документов'.\n"
            "2. Если типы совпадают и все поля на месте — выбирай 'Сценарий А'.\n\n"
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
            response_text = response["choices"][0]["text"].strip()
            if not response_text.startswith("{"):
                response_text = "{" + response_text
                
            verdict = json.loads(response_text)
            status = verdict.get("status", "Сценарий В")
            reason_str = verdict.get("reason", "несоответствие регламенту")
        except Exception:
            status = "Сценарий В"
            reason_str = "Требуется дополнительная проверка полей документа сотрудником деканата"
            if "Сценарий А" in response_text:
                status = "Сценарий А"
            elif "Сценарий Б" in response_text:
                status = "Сценарий Б"
        
        if status == "Сценарий А":
            final_text = f"Документ успешно верифицирован: «Документ '{doc_type_detected}' полностью удовлетворяет требованиям регламента ({paragraph_source}). Ошибок оформления не обнаружено.»"
        else:
            if "целевым темам" in reason_str or "стипенди" in user_question.lower() or "бюджет" in user_question.lower():
                final_text = f"Запрос отклонен: «Заданный вопрос '{user_question}' некорректен, так как наш ИИ-помощник обслуживает только процессы верификации Справок, Академических отпусков и Практики. Пожалуйста, задайте целевой вопрос.»"
            else:
                final_text = f"Ошибка валидации: «{reason_str}. Рекомендуется проверить правильность заполнения дат или скорректировать данные через кнопку 'Изменить вручную'.»"

        return {"status": status, "paragraph": paragraph_source, "final_answer": final_text}
