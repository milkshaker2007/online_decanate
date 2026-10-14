import os
import json
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

    def evaluate_document(self, extracted_fields: dict, paragraph_text: str, paragraph_source: str) -> dict:
        """
        Сопоставляет поля OCR и требования пункта Положения через LLM.
        Возвращает финальный сформированный текст ответа.
        """
        fields_str = "\n".join([f"- {k}: {v}" for k, v in extracted_fields.items()])

        system_prompt = (
            "Ты — строгий верификатор документов в деканате вуза. "
            "Сопоставь извлеченные из документа поля с требованиями пункта Положения. "
            "Отвечай строго в формате JSON с тремя обязательными ключами:\n"
            "1. \"status\": строго строка \"Сценарий А\", \"Сценарий Б\" или \"Сценарий В\"\n"
            "2. \"missing_fields\": список названий полей, которых не хватает (значение равно \"не найдено\"), но которые обязательны по пункту.\n"
            "3. \"reason\": краткое описание сути проблемы (для Сценария В)\n\n"
            "ПРАВИЛА СЦЕНАРИЕВ:\n"
            "- Если ВСЕ обязательные поля из пункта присутствуют в карточке (не равны \"не найдено\"), выбирай \"Сценарий А\".\n"
            "- Если какого-то обязательного поля или печати/подписи не хватает (написано \"не найдено\"), выбирай \"Сценарий Б\".\n"
            "- Если в документе и в Положении вообще нет информации о запрашиваемых сущностях, выбирай \"Сценарий В\".\n"
            "Запрещено писать любой текст, кроме валидного JSON."
        )

        user_content = (
            f"ТРЕБОВАНИЯ ПУНКТА ПОЛОЖЕНИЯ ({paragraph_source}):\n{paragraph_text}\n\n"
            f"ИЗВЛЕЧЕННЫЕ ИЗ ДОКУМЕНТА ПОЛЯ:\n{fields_str}\n\n"
            "Выдай JSON:"
        )

        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

        response = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.1,
            top_p=0.9,
            stop=["<|im_end|>", "</s>"]
        )

        # Исправлено: choices — это список, берём его нулевой элемент, а уже потом ключ ["text"]
        try:
            response_text = response["choices"][0]["text"].strip()
        except (KeyError, IndexError, TypeError):
            response_text = ""
        
        try:
            if "{" in response_text and "}" in response_text:
                response_text = response_text[response_text.find("{"):response_text.rfind("}")+1]
            verdict = json.loads(response_text)
        except Exception:
            verdict = {"status": "Сценарий В", "missing_fields": [], "reason": "информации"}

        status = verdict.get("status", "Сценарий В")
        missing_list = verdict.get("missing_fields", [])
        
        if status == "Сценарий А":
            doc_name = "Документ"
            if "Форма обучения" in extracted_fields:
                doc_name = "Справка"
            elif "Причина отпуска" in extracted_fields:
                doc_name = "Заявление"
                
            final_text = f"Документ в порядке: «{doc_name} содержит все данные, которые требуют {paragraph_source} Положения. Окончательное решение примет сотрудник деканата»"
        
        elif status == "Сценарий Б":
            missing_str = ", ".join(missing_list) if missing_list else "обязательных полей"
            final_text = f"Чего-то не хватает: «Не хватает {missing_str}. Эти поля обязательны по {paragraph_source} Положения»"
        
        else:
            reason_str = verdict.get("reason", "информации").strip()
            # Исправлено: убираем зацикливание и тавтологию "информации об информации"
            if not reason_str or "нет информации" in reason_str.lower() or reason_str == "информации":
                final_text = "Ответа нет: «В предоставленном документе отсутствуют обязательные реквизиты, либо данный тип запроса не регламентируется текущим Положением вуза. Обратитесь в деканат лично»"
            else:
                clean_reason = reason_str.replace("В документе и в Положении нет информации о", "").strip()
                final_text = f"Ответа нет: «Не удалось принять автоматическое решение. Отсутствует или не распознана информация о: {clean_reason}. Пожалуйста, обратитесь в деканат»"

        return {
            "status": status,
            "paragraph": paragraph_source,
            "final_answer": final_text
        }
