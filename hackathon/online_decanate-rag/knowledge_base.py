import re
from pathlib import Path
from typing import List, Dict

class KnowledgeBase:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.chunks: List[Dict[str, str]] = []

    def load_and_parse(self) -> List[Dict[str, str]]:
        """
        Сканирует папку data_dir, находит .md и .txt файлы и режет их на пункты.
        Возвращает список словарей вида: [{'source': 'п. 3.1', 'text': '3.1. Текст...'}]
        """
        self.chunks = []
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Папка с данными не найдена: {self.data_dir.resolve()}")

        # Исправлено: ищем и .md, и .txt файлы, чтобы не пропустить rules.md
        files = list(self.data_dir.glob("*.md")) + list(self.data_dir.glob("*.txt"))
        if not files:
            return []

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            parsed_chunks = self._split_into_chunks(content)
            self.chunks.extend(parsed_chunks)
            
        return self.chunks

    def _split_into_chunks(self, text: str) -> List[Dict[str, str]]:
        """
        Разбивает текст Положения на отдельные нумерованные пункты.
        """
        chunks = []
        lines = text.split("\n")
        current_section = "Общие положения"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Проверяем, является ли строка заголовком верхнего уровня (например: "3. Справка об обучении")
            if re.match(r"^\d+\.\s+[А-Яа-я]", line):
                current_section = line
                # Исправлено: убран continue, чтобы ключевые слова заголовка попадали в индекс RAG
                
            # Ищем строки, начинающиеся с номера пункта (например: "3.1. Текст...")
            match = re.match(r"^(\d+\.\d+(?:\.\d+)?\.?)\s*(.*)", line)
            
            if match:
                point_number = match.group(1).rstrip(".")
                point_text = match.group(2).strip()
                
                chunks.append({
                    "source": f"п. {point_number}",
                    "section": current_section,
                    "text": f"Раздел: {current_section}. Пункт {point_number}. {point_text}"
                })
            else:
                # Если строка без номера — приклеиваем её к текущему пункту
                if chunks:
                    chunks[-1]["text"] += " " + line
                    
        return chunks
