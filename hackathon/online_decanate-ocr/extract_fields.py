import re

NOT_FOUND = "не найдено"
_MONTHS = "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"

# Шаблон DATE защищен жесткими границами (?<!\d), чтобы не путать даты с кодами направлений
DATE = (r"(?<!\d)(?:\d{1,2}[./]\d{1,2}[./](?:\d{4}|\d{2}))(?!\d)"
        r"|\d{1,2}\s+(?:%s)\s+\d{4}(?:\s*г(?:ода|\.)?)?" % _MONTHS)

_W = r"[А-ЯЁа-яё\-]+"
FIO = r"[А-ЯЁ][а-яё\-]+\s+[А-ЯЁ][а-яё\-]+\s+[А-ЯЁ][а-яё\-]+"
FIO_SHORT = r"[А-ЯЁ][а-яё\-]+\s+[А-ЯЁ]\.\s?[А-ЯЁ]\."

_ESIGN_CASE = [r"\bЭЦП\b", r"\bЭП\b"]
_ESIGN_ANY_CASE = [
    r"электронн\w*\s+(?:цифров\w+\s+)?подпис",
    r"электронн\w*\s+печат",
    r"документ\s+подписан",
    r"подписан\w*\s+(?:(?:усиленн|квалифицированн|неквалифицированн)\w*\s+|простой\s+)*(?:электронн|ЭП\b|ЭЦП\b)",
    r"сертификат\w*\s*(?:ключа\s+проверки\s+\w+\s*)?(?:№|N|:)?\s*[0-9A-Fa-f]{16,}",
    r"номер\s+доверенности",
    r"владелец\s+сертификата",
]

def _clean(s):
    res = re.sub(r"\s+", " ", s or "").strip(" \t\n,;:_")
    # Если строка состоит только из линий подчеркивания бланка или пустых знаков — возвращаем пустую строку
    if not res or re.match(r"^[._\- \s]+$", res):
        return ""
    return res

def _first(patterns, text, flags=0):
    for p in patterns:
        m = re.search(p, text, flags)
        if m:
            val = _clean(m.group(1))
            if val:
                return val
    return NOT_FOUND

def _fio(text, patterns):
    val = _first(patterns, text)
    if val != NOT_FOUND and val.isupper():
        val = val.title()
    return val

def _period(text):
    m = re.search(r"\bс\s+(%s)\s*(?:по|до|[-–—])\s*(%s)" % (DATE, DATE), text, re.I)
    return (_clean(m.group(1)), _clean(m.group(2))) if m else (None, None)

def _number_date(text):
    # 1. Гибкий раздельный поиск для бланков, где номер и дата на разных строках
    num_match = re.search(r"(?i:номер\s+(?:справки|документа|регистрации))\s*[:\-№]?\s*([\w\-/]+)", text)
    date_match = re.search(r"(?i:дата\s+(?:выдачи|выпуск\w*|составлен\w*))\s*[:\-]?\s*(%s)" % DATE, text)
    
    if num_match and date_match:
        n_val = _clean(num_match.group(1))
        d_val = _clean(date_match.group(1))
        if not re.match(r"^\d{2}\.\d{2}\.\d{2}$", n_val):
            return "№ %s от %s" % (n_val, d_val)

    # 2. Резервный классический вариант для слитного написания (№ 123 от 01.01.2026)
    num_re = r"(?:№|\bN\b|\bNo\b\.?)\s*([\w\-/]+)"
    cleaned = re.sub(r"(?i)приказ\w*[^№]{0,60}№\s*[\w\-/]+(?:\s*от\s*%s)?" % DATE, " ", text)
    m = re.search(r"(?i)\b(?:выдан[аоы]?|дана|подтверждает\w*)\b", cleaned)
    head = cleaned[:m.start()] if m else cleaned
    num = date = None
    mn = re.search(num_re, head)
    md = re.search(DATE, head)
    if mn or md:
        num = mn.group(1) if mn else None
        date = _clean(md.group(0)) if md else None
    else:
        mn = re.search(num_re, cleaned)
        md = re.search(r"(?i:от)\s+(%s)" % DATE, cleaned)
        num = mn.group(1) if mn else None
        date = _clean(md.group(1)) if md else None
        
    if num and re.match(r"^\d{2}\.\d{2}\.\d{2}$", str(num).strip()):
        num = None
        
    res_val = ("№ " + num) if num else (date or NOT_FOUND)
    if re.match(r"^(№\s*)?\d{2}\.\d{2}\.\d{2}$", str(res_val).strip()):
        return NOT_FOUND
        
    return res_val

def _line_has_esign(line):
    if any(re.search(p, line) for p in _ESIGN_CASE):
        return True
    return any(re.search(p, line, re.I) for p in _ESIGN_ANY_CASE)

def _has_esign(text):
    return any(_line_has_esign(line) for line in text.split("\n"))

def _strip_esign(text):
    kept = []
    for line in text.split("\n"):
        if _line_has_esign(line):
            continue
        if re.match(r"(?i)\s*(?:сертификат|владелец|действителен|срок\s+действия|номер\s+доверенности)\b", line):
            continue
        if re.fullmatch(r"(?i)\s*(?=[0-9a-f\-]*[a-f])[0-9a-f\-]{8,}\s*", line):
            continue
        kept.append(line)
    return "\n".join(kept)

def _has_signature(text):
    return bool(re.search(r"(?i)подпис|_{2,}|/\s*[А-ЯЁ]\.", _strip_esign(text)))

def _has_stamp(text):
    return bool(re.search(r"(?i)\bм\.?\s?п\.?(?!\w)|печат", _strip_esign(text)))

def _yes_no(flag):
    return "есть" if flag else "нет"

def detect_doc_type(text):
    low = text.lower()
    if "академическ" in low and "отпуск" in low:
        return "leave"
    if "практик" in low:
        return "practice"
    if "справк" in low and ("обуча" in low or "студент" in low):
        return "study"
    return None

def _study_period(t):
    start_match = re.search(r"(?i:начал\w+\s+(?:периода\s+)?обучения)\s*[:\-]?\s*(%s)" % DATE, t)
    end_match = re.search(r"(?i:окончани\w+\s+(?:периода\s+)?обучения)\s*[:\-]?\s*(%s)" % DATE, t)
    
    if start_match and end_match:
        return "%s – %s" % (_clean(start_match.group(1)), _clean(end_match.group(1)))
        
    start, end = _period(t)
    return "%s – %s" % (start, end) if start else NOT_FOUND

def _study_form(t):
    m = re.search(r"(?i)(очно-заочн|заочн|очн)\w*", t)
    if not m:
        return NOT_FOUND
    return {"очно-заочн": "очно-заочная", "заочн": "заочная", "очн": "очная"}[m.group(1).lower()]

def _study_direction(t):
    code = r"\d{2}\.\d{2}\.\d{2}"
    op = r"\s*[:\-]?\s*[«\"“„]?\s*"
    return _first([
        r"(?i:направлени[юе]\s+подготовки|специальност[ьи])" + op + r"(%s\s*[-–—]?\s*[^\n]+)" % code,
        r"(?i:программ\w*)" + op + r"(%s\s*[-–—]?\s*[^\n]+)" % code,
                r"(?<![\d.])(%s\s*[-–—]?\s*[А-ЯЁа-яё][^\n]+)" % code
    ], t)

def _extract_study(t, has_esign, marks):
    # Базовое извлечение полей
    fields = {
        "ФИО студента": _fio(t, [
            r"(?i:ФИО\s+студента)\s*[:\-]?\s*(%s)" % FIO,
            r"(?i:выдана|выдан|дана)\s*[:,]?\s*(%s)" % FIO,
            r"(?i:студент(?:ка)?)\s*[:,]?\s*(%s)" % FIO
        ]),
        "Название вуза": _first([
            r"[«\"“]([^»\"”«“]*(?i:университет|институт|академи)[^»\"”«“]*)[»\"”]",
            r"([^\n]{0,120}?(?i:университет|институт|академи)[^\n]{0,80})"], t),
        "Направление подготовки": _study_direction(t),
        "Форма обучения": _study_form(t),
        "Период обучения": _study_period(t),
        "Номер и дата выдачи": _number_date(t),
        "Подпись и печать": _yes_no(has_esign or marks["signature"] or marks["stamp"]),
    }
    
    # ЖЕСТКИЙ АВТОНОМНЫЙ ПАТЧ НАДЕЖНОСТИ ДЛЯ СЛУЧАЕВ СЛОЖНОЙ СТРУКТУРЫ СТРОК БЛАНКА
    clean_line_text = re.sub(r"\s+", " ", t)
    
    # 1. Если направление подготовки пропущено, ищем код специальности и забираем всю строку
    if fields["Направление подготовки"] == NOT_FOUND:
        dir_m = re.search(r"(\d{2}\.\d{2}\.\d{2}\s+[^«\"'\n\d]+)", clean_line_text)
        if dir_m:
            fields["Направление подготовки"] = dir_m.group(1).strip()
            
    # 2. Если номер и дата не склеились, вытаскиваем автономно номер справки вида ОБ-1002
    if fields["Номер и дата выдачи"] == NOT_FOUND:
        num_m = re.search(r"(?i:номер\s+справки)\s*[:\-№]?\s*([\w\-]+)", clean_line_text)
        date_m = re.search(r"(?i:дата\s+выдачи)\s*[:\-]?\s*(" + DATE + ")", clean_line_text)
        if num_match := num_m:
            n_str = num_match.group(1).strip()
            if date_match := date_m:
                fields["Номер и дата выдачи"] = f"№ {n_str} от {date_match.group(1).strip()}"
            else:
                fields["Номер и дата выдачи"] = f"№ {n_str}"
                
    return fields

def _extract_leave(t, has_esign, marks):
    start = _first([r"(?i:начал\w*\s*(?:отпуска|периода)?)\s*[:\-]?\s*(%s)" % DATE, r"(?i:с\s+)(%s)" % DATE], t)
    end = _first([r"(?i:окончани\w*\s*(?:отпуска|периода)?)\s*[:\-]?\s*(%s)" % DATE, r"(?i:по\s+)(%s)" % DATE], t)
    
    if start == NOT_FOUND or end == NOT_FOUND:
        m = re.search(r"(?i:отпуск)[\s\S]{0,300}?\bс\s+(%s)\s*(?i:по|до)\s+(%s)" % (DATE, DATE), t)
        if m:
            start = _clean(m.group(1))
            end = _clean(m.group(2))
        else:
            st_d, en_d = _period(t)
            start = st_d or start
            end = en_d or end

    app_date = _first([r"(?i:дата\s+заявления)\s*[:\-]?\s*(%s)" % DATE, r"(?i:дата)\s*[:\-]?\s*(%s)" % DATE], t)
    if app_date == NOT_FOUND:
        rest = [_clean(d) for d in re.findall(DATE, t) if _clean(d) not in (start, end)]
        app_date = rest[-1] if rest else NOT_FOUND

    group = _first([r"(?i:групп[ыа])\s*[:№]?\s*([A-Za-zА-Яа-я0-9\-]+)"], t)
    course = _first([r"(\d)\s*(?i:курс)"], t)
    gc = "группа %s, %s курс" % (group, course) if group != NOT_FOUND and course != NOT_FOUND else (f"группа {group}" if group != NOT_FOUND else (f"{course} курс" if course != NOT_FOUND else NOT_FOUND))

    reason = _first([r"(?i:по причине|в связи с)\s+([^\n.]+)", r"(?i)(по семейным обстоятельствам|по состоянию здоровья)"], t)

    return {
        "ФИО студента": _fio(t, [r"(?i:ФИО\s+студента)\s*[:\-]?\s*(%s)" % FIO, r"(?i:от student(?:а|ки)?[^\n]*?)(%s)" % FIO, r"(%s)" % FIO]),
        "Группа и курс": gc,
        "Причина отпуска": reason,
        "Дата начала": start,
        "Дата окончания": end,
        "Дата заявления": app_date,
        "Подпись студента": _yes_no(has_esign or marks["signature"]),
    }

def _extract_practice(t, has_esign, marks):
    start, end = _period(t)
    return {
        "ФИО студента": _fio(t, [r"(?i:студент(?:ка)?)\s*[:,]?\s*(%s)" % FIO, r"(%s)" % FIO]),
        "Организация": _first([r"((?:ООО|ЗАО|ОАО|ПАО|АО|ИП)\s*[«\"][^»\"]+[»\"])"], t),
        "Сроки практики": "%s – %s" % (start, end) if start else NOT_FOUND,
        "Руководитель практики": _first([r"(?i:руководител[ьяю]\s+практики)\s+(%s|%s)" % (FIO, FIO_SHORT)], t),
        "Оценка": _first([r"(?i:оценк\w*)\s*[:\-]?\s*[«\"]?(отлично|хорошо|удовлетворительно|неудовлетворительно|зачтено|\d)"], t),
        "Номер и дата выдачи": _number_date(t),
        "Подпись и печать": _yes_no(has_esign or (marks["signature"] and marks["stamp"])),
    }

def extract_fields(ocr_text, doc_type=None, marks=None):
    text = re.sub(r"[ \t\u00a0]+", " ", ocr_text or "")
    doc_type = doc_type or detect_doc_type(text)
    handlers = {
        "study":    ("Справка об обучении", _extract_study),
        "leave":    ("Заявление на академический отпуск", _extract_leave),
        "practice": ("Справка о практике", _extract_practice),
    }
    if doc_type not in handlers:
        return {"тип_документа": NOT_FOUND}
    name, fn = handlers[doc_type]
    result = {"тип_документа": name}
    has_esign = _has_esign(text)
    clean = _strip_esign(text)
    if marks is None:
        marks = {"signature": _has_signature(clean), "stamp": _has_stamp(clean)}
    else:
        marks = {"signature": bool(marks.get("signature")), "stamp": bool(marks.get("stamp"))}
    result.update(fn(clean, has_esign, marks))
    return result

