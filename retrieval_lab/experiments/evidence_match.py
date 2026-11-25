from retrieval_lab.experiments.textutil import normalize_text, span_fragments


def _span_pages(span: dict) -> list[int]:
    listed = span.get("pdf_pages")
    if isinstance(listed, list) and listed:
        return [int(page) for page in listed]
    return [int(span["pdf_page"])]


def _page_chunks(hits: list[dict], pages: list[int]) -> list[dict]:
    allowed = set(pages)
    return [hit for hit in hits if hit.get("page") in allowed]


def equivalent_targets(span: dict) -> list[tuple[str, list[int]]]:
    """Resolve equivalent evidence.

    A string is accepted on any page listed for the span. An object with
    ``text`` and ``pdf_page`` is accepted only on that page, so one wording
    cannot confirm a chunk from the other page.
    """
    pages = _span_pages(span)
    targets = []
    for item in span.get("equivalent_texts") or []:
        if isinstance(item, dict):
            text = str(item.get("text") or "")
            bound = item.get("pdf_page")
            target_pages = [int(bound)] if bound is not None else pages
        else:
            text = str(item)
            target_pages = pages
        if text.strip():
            targets.append((text, target_pages))
    return targets


def _fragment_state(fragment: str, chunks: list[dict]) -> str:
    needle = normalize_text(fragment)
    if not needle:
        return "miss"
    for chunk in chunks:
        if needle in normalize_text(chunk.get("content")):
            return "text_match"
    return "miss"


def match_spans(question: dict, hits: list[dict]) -> dict:
    """Match required spans against retrieved chunks.

    A table_ambiguous question can show a textual overlap and still stay
    unconfirmed: column association in extracted tables is ambiguous, so
    those spans are needs_review and do not count as evidence_satisfied.
    """
    spans = question.get("minimal_support_spans") or []
    table_ambiguous = bool(question.get("table_ambiguous"))
    span_results = []
    for span in spans:
        if not span.get("required", True):
            continue
        pages = _span_pages(span)
        page = int(span["pdf_page"])
        chunks = _page_chunks(hits, pages)
        equivalents = span.get("equivalent_texts")
        if isinstance(equivalents, list) and equivalents:
            states = [
                _fragment_state(text, _page_chunks(hits, target_pages))
                for text, target_pages in equivalent_targets(span)
            ]
            raw_state = "text_match" if "text_match" in states else "miss"
        else:
            fragments = span_fragments(span)
            states = [_fragment_state(fragment, chunks) for fragment in fragments]
            if fragments and all(state == "text_match" for state in states):
                raw_state = "text_match"
            elif any(state == "text_match" for state in states):
                raw_state = "partial_coverage"
            else:
                raw_state = "miss"
        confirmed = raw_state == "text_match" and not table_ambiguous
        state = raw_state
        if table_ambiguous and raw_state in {"text_match", "partial_coverage"}:
            state = "needs_review"
        span_results.append(
            {
                "span_id": span.get("span_id"),
                "pdf_page": page,
                "state": state,
                "text_overlap": raw_state,
                "confirmed": confirmed,
                "table_ambiguous": table_ambiguous,
            }
        )
    required = [item for item in span_results]
    evidence_satisfied = bool(required) and all(item["confirmed"] for item in required)
    needs_review = any(item["state"] == "needs_review" for item in required)
    if evidence_satisfied:
        coverage_state = "evidence_satisfied"
    elif needs_review:
        coverage_state = "needs_review"
    elif any(item["text_overlap"] == "partial_coverage" for item in required):
        coverage_state = "partial_coverage"
    elif any(item["text_overlap"] == "text_match" for item in required):
        coverage_state = "text_match"
    else:
        coverage_state = "miss"
    return {
        "spans": span_results,
        "evidence_satisfied": evidence_satisfied,
        "needs_review": needs_review,
        "coverage_state": coverage_state,
        "confirmed_span_count": sum(1 for item in required if item["confirmed"]),
        "required_span_count": len(required),
    }
