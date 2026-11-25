import re
import unicodedata


def normalize_text(value: str | None) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    text = text.replace("\u00ad", "")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def span_fragments(span: dict) -> list[str]:
    lines = span.get("lines")
    if isinstance(lines, list) and lines:
        return [str(line) for line in lines if str(line).strip()]
    text = span.get("text")
    if isinstance(text, str) and text.strip():
        return [text]
    return []
