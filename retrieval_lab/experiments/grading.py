import json
import re
import time
from typing import Any, Callable

from retrieval_lab.experiments.textutil import normalize_text

JudgeFn = Callable[[str, str, str], dict[str, Any]]

_NUMBER = re.compile(r"\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+(?:\.\d+)?")


def _to_float(token: str) -> float | None:
    cleaned = token.strip()
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_numbers(text: str) -> list[float]:
    values = []
    for match in _NUMBER.findall(text or ""):
        number = _to_float(match)
        if number is not None:
            values.append(number)
    return values


def grade_answer(question: dict, answer: str, judge_fn: JudgeFn | None = None) -> dict:
    mode = (question.get("grading") or {}).get("mode")
    text = answer or ""
    normalized = normalize_text(text)
    if not normalized or normalized in {"não sei", "nao sei", "i don't know", "i do not know"}:
        return {
            "answer_correct": 0,
            "grading_mode": mode,
            "uncertain": False,
            "empty_or_refusal": True,
            "evaluation_status": "valid",
        }
    grading = question.get("grading") or {}
    if mode == "contains_all":
        phrases = grading.get("required_phrases") or []
        ok = all(normalize_text(phrase) in normalized for phrase in phrases)
        return {
            "answer_correct": int(ok),
            "grading_mode": mode,
            "uncertain": False,
            "empty_or_refusal": False,
            "evaluation_status": "valid",
        }
    if mode == "numeric":
        expected = float(grading["value"])
        atol = float(grading.get("atol", 0.05))
        found = extract_numbers(text)
        ok = any(abs(number - expected) <= atol for number in found)
        return {
            "answer_correct": int(ok),
            "grading_mode": mode,
            "uncertain": False,
            "empty_or_refusal": False,
            "evaluation_status": "valid",
        }
    if mode in {"judge", "semantic_review"}:
        if judge_fn is None:
            return {
                "answer_correct": None,
                "grading_mode": mode,
                "uncertain": True,
                "empty_or_refusal": False,
                "judge_pending": True,
                "evaluation_status": "not_evaluated",
            }
        if mode == "semantic_review":
            verdict = judge_fn(
                question["question"],
                question["reference_answer"],
                text,
                list(grading.get("key_concepts") or []),
            )
        else:
            verdict = judge_fn(question["question"], question["reference_answer"], text)
        label = verdict.get("verdict")
        if label == "correct":
            correct = 1
            uncertain = False
        elif label == "incorrect":
            correct = 0
            uncertain = False
        else:
            correct = None
            uncertain = True
        return {
            "answer_correct": correct,
            "grading_mode": mode,
            "uncertain": uncertain,
            "empty_or_refusal": False,
            "judge_verdict": label,
            "judge_called": True,
            "judge_prompt_tokens": verdict.get("prompt_tokens"),
            "judge_completion_tokens": verdict.get("completion_tokens"),
            "judge_latency_ms": verdict.get("judge_latency_ms"),
            "judge_latency_source": verdict.get("judge_latency_source"),
            "evaluation_status": "valid" if correct is not None else "inconclusive",
        }
    raise ValueError(f"Unsupported grading mode: {mode}")


def support_label(answer_correct: int | None, evidence_satisfied: bool) -> str:
    if answer_correct is None:
        return "inconclusive"
    if answer_correct == 1 and evidence_satisfied:
        return "supported"
    if answer_correct == 1 and not evidence_satisfied:
        return "correct_unsupported"
    return "incorrect"


def generate_answer(
    client: Any,
    *,
    model: str,
    temperature: float,
    max_output_tokens: int,
    system_prompt: str,
    user_prompt: str,
) -> dict:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_output_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    elapsed = round((time.perf_counter() - started) * 1000, 3)
    choice = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
    return {
        "model_answer": choice,
        "llm_latency_ms": elapsed,
        "llm_latency_source": "measured",
        "prompt_tokens_measured": None if prompt_tokens is None else int(prompt_tokens),
        "completion_tokens_measured": None if completion_tokens is None else int(completion_tokens),
    }


def judge_with_client(
    client: Any,
    model: str,
    question: str,
    gold: str,
    answer: str,
    key_concepts: list[str] | None = None,
) -> dict:
    started = time.perf_counter()
    system_prompt = (
        "Compare the model answer to the reference. "
        'Reply with JSON {"verdict":"correct"|"incorrect"|"uncertain"}.'
    )
    payload = {"question": question, "reference_answer": gold, "model_answer": answer}
    if key_concepts:
        system_prompt = (
            "Compare the model answer to the reference. "
            "Key concepts must be covered in meaning, not necessarily with the same words. "
            "Use uncertain when the answer is too incomplete to decide. "
            'Reply with JSON {"verdict":"correct"|"incorrect"|"uncertain"}.'
        )
        payload["key_concepts"] = key_concepts
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=80,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    raw = response.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
        verdict = parsed.get("verdict", "uncertain")
    except json.JSONDecodeError:
        verdict = "uncertain"
    if verdict not in {"correct", "incorrect", "uncertain"}:
        verdict = "uncertain"
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
    return {
        "verdict": verdict,
        "judge_latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "judge_latency_source": "measured",
        "prompt_tokens": None if prompt_tokens is None else int(prompt_tokens),
        "completion_tokens": None if completion_tokens is None else int(completion_tokens),
    }


def estimate_chat_cost_usd(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    pricing: dict,
) -> dict:
    if prompt_tokens is None or completion_tokens is None:
        return {
            "generation_cost_usd": None,
            "generation_cost_basis": "unknown",
            "api_cost_usd_chat_estimated": None,
            "chat_cost_basis": "unknown",
        }
    input_price = float(pricing["input"]) / 1_000_000
    output_price = float(pricing["output"]) / 1_000_000
    cost = round(prompt_tokens * input_price + completion_tokens * output_price, 8)
    basis = "estimated_unit_price_times_measured_tokens"
    return {
        "generation_cost_usd": cost,
        "generation_cost_basis": basis,
        "api_cost_usd_chat_estimated": cost,
        "chat_cost_basis": basis,
    }


def evaluation_costs(
    grading_mode: str | None,
    graded: dict,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    pricing: dict,
) -> dict:
    """Split the answer cost from the judge cost.

    A missing token count stays unknown. A rubric that does not call the
    judge has a judge cost of zero. A judge rubric that was not executed
    stays unknown.
    """
    generation = estimate_chat_cost_usd(prompt_tokens, completion_tokens, pricing)
    if grading_mode not in {"judge", "semantic_review"}:
        judge_cost = 0.0
        judge_basis = "not_applicable"
    elif not graded.get("judge_called"):
        judge_cost = None
        judge_basis = "not_evaluated"
    else:
        judged = estimate_chat_cost_usd(
            graded.get("judge_prompt_tokens"),
            graded.get("judge_completion_tokens"),
            pricing,
        )
        judge_cost = judged["generation_cost_usd"]
        judge_basis = judged["generation_cost_basis"]
    row_cost = None
    if generation["generation_cost_usd"] is not None and judge_cost is not None:
        row_cost = round(generation["generation_cost_usd"] + judge_cost, 8)
    return {
        **generation,
        "judge_cost_usd": judge_cost,
        "judge_cost_basis": judge_basis,
        "evaluation_row_cost_usd": row_cost,
        "evaluation_row_cost_basis": (
            "generation_plus_judge_excludes_shared_embedding" if row_cost is not None else "unknown"
        ),
    }


def context_from_hits(hits: list[dict]) -> str:
    blocks = []
    for hit in hits:
        blocks.append(f"[page {hit.get('page')}] {hit.get('content') or ''}")
    return "\n---\n".join(blocks)
