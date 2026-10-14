import re

NOT_FOUND = "не найдено"
_MONTHS = "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"

# Исправлено: жесткие границы для года, чтобы регулярное выражение не путало дату с кодом специальности 38.03.01
DATE = (r"(?<!\d)(?:\d{1,2}[./]\d{1,2}[./](?:\d{4}|\d{2}))(?!\d)"
        r"|\d{1,2}\s+(?:%s)\s+\d{4}(?:\s*г(?:ода|\.)?)?" % _MONTHS)

_W = r"[А-ЯЁа-яё\-]+"
FIO = r"[А-ЯЁ]%s\s+[А-ЯЁ]%s\s+[А-ЯЁ]%s" % (_W, _W, _W)
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
    return re.sub(r"\s+", " ", s or "").strip(" \t\n,;:")

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
    if num and date:
        return "№ %s от %s" % (num, date)
    return ("№ " + num) if num else (date or NOT_FOUND)

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
    # Исправлено: Раздельный гибкий поиск начала и окончания периода, устойчивый к переносам строк (\n)
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
    stop = (r"(?=\s+в\s+(?i:федеральн|государственн|муниципальн|негосударственн|образовательн|частн|ФГБОУ|ФГАОУ|ФГБУ|ГБОУ)|\s*[»\"”«]|\.\s+[А-ЯЁ]|$)")
    op = r"\s*[:\-]?\s*[«\"“„]?\s*"
    return _first([
        r"(?i:образовательн\w+\s+программ\w*)" + op + r"(%s\s*[-–—]?\s*[\s\S]{1,200}?)" % code + stop,
        r"(?i:направлени[юе]\s+подготовки|специальност[ьи])" + op + r"(%s\s*[-–—]?\s*[\s\S]{1,200}?)" % code + stop,
        r"(?i:направлени[юе]\s+подготовки|специальност[ьи])" + op + r"([\s\S]{1,200}?)" + stop,
        r"(?![\d.])(%s(?![\d.])\s*[-–—]?\s*[А-ЯЁ][\s\S]{1,200}?)" % code + stop,
    ], t)

def _extract_study(t, has_esign, marks):
    return {
        "ФИО студента": _fio(t, [
            r"(?i:выдана|выдан|дана|подтверждает(?:ся)?,? что)\s*[:,]?\s*(%s)" % FIO,
            r"(?i:студент(?:ка)?)\s*[:,]?\s*(%s)" % FIO,
            r"(%s)" % FIO]),
        "Название вуза": _first([
            r"[«\"“]([^»\"”«“]*(?i:университет|институт|академи)[^»\"”«“]*)[»\"”]",
            r"([^\n]{0,120}?(?i:университет|институт|академи)[^\n]{0,80})"], t),
        "Направление подготовки": _study_direction(t),
        "Форма обучения": _study_form(t),
        "Период обучения": _study_period(t),
        "Номер и дата выдачи": _number_date(t),
        "Подпись и печать": _yes_no(has_esign or marks["signature"] or marks["stamp"]),
    }

def _extract_leave(t, has_esign, marks):
    m = re.search(r"(?i:отпуск)[\s\S]{0,150}?\bс\s+(%s)\s*(?i:по|до)\s+(%s)" % (DATE, DATE), t)
    start, end = (_clean(m.group(1)), _clean(m.group(2))) if m else _period(t)
    app_date = _first([r"(?i:дата)\s*[:\-]?\s*(%s)" % DATE], t)
    if app_date == NOT_FOUND:
        rest = [_clean(d) for d in re.findall(DATE, t) if _clean(d) not in (start, end)]
        app_date = rest[-1] if rest else NOT_FOUND
    group = _first([r"(?i:групп[ыа])\s*[:№]?\s*([A-Za-zА-Яа-я0-9\-]+)"], t)
    course = _first([r"(\d)\s*(?i:курс)"], t)
    gc = f"группа {group}, {course} курс" if group != NOT_FOUND and course != NOT_FOUND else (f"группа {group}" if group != NOT_FOUND else (f"{course} курс" if course != NOT_FOUND else NOT_FOUND))
    reason = _first([r"(?i:по причине|в связи с)\s+([^\n.]+)", r"(?i)(по семейным обстоятельствам|по состоянию здоровья|по медицинским показаниям|по беременности и родам)"], t)
    return {
        "ФИО студента": _fio(t, [r"(?i:от student(?:a|ки)?[^\n]*?)(%s)" % FIO, r"(?i:я,)\s*(%s)" % FIO, r"(%s)" % FIO]),
        "Группа и курс": gc,
        "Причина отпуска": reason,
        "Дата начала": start or NOT_FOUND,
        "Дата окончания": end or NOT_FOUND,
        "Дата заявления": app_date,
        "Подпись студента": _yes_no(has_esign or marks["signature"]),
    }

def _extract_practice(t, has_esign, marks):
    start, end = _period(t)
    return {
        "ФИО студента": _fio(t, [r"(?i:студент(?:ка)?|обучающ\w+|подтверждает(?:ся)?,? что)\s*[:,]?\s*(%s)" % FIO, r"(%s)" % FIO]),
        "Organizaciya": _first([r"((?:ООО|ЗАО|ОАО|ПАО|АО|ИП|ГБУ|МБУ|ФГБУ|ФГУП)\s*[«\"][^»\"]+[»\"])", r"(?i:в организации|на предприятии|на базе)\s*[:\-]?\s*([«\"][^»\"]+[»\"])"], t),
        "Сроки практики": "%s – %s" % (start, end) if start else NOT_FOUND,
        "Руководитель практики": _first([r"(?i:руководител[ьяю][^\n:]*?)\s*[:\-]\s*(%s|%s)" % (FIO, FIO_SHORT), r"(?i:руководител[ьяю]\s+практики[^\n]*?)\s+(%s|%s)" % (FIO, FIO_SHORT)], t),
        "Оценка": _first([r"(?i:оценк\w*|аттестован\w* на)\s*[:\-]?\s*[«\"]?(отлично|хорошо|удовлетворительно|неудовлетворительно|зачтено|\d)"], t),
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
