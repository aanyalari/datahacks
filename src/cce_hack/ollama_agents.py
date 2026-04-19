"""Multi-agent report using a local Ollama LLM (free, no API key)."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from cce_hack.agent_tools import build_context_package, context_json_for_prompt


def ollama_ping(base_url: str, timeout_s: float = 5.0) -> tuple[bool, str]:
    """Return (ok, message). GET /api/tags is cheap and confirms Ollama is listening."""
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            resp.read()
        return True, "Ollama responded."
    except Exception as e:
        return False, str(e)


def ollama_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_s: int = 120,
    options: dict[str, Any] | None = None,
) -> str:
    """
    POST /api/chat (non-streaming).

    `options` lowers latency on CPU, e.g. num_predict caps output length; smaller = faster.
    See: https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
    """
    url = base_url.rstrip("/") + "/api/chat"
    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    if options:
        payload["options"] = options
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error: {e.code} {e.read().decode('utf-8', errors='replace')[:500]}") from e
    except urllib.error.URLError as e:
        err = str(e).lower()
        if "timed out" in err or "time out" in err:
            raise RuntimeError(
                f"Ollama did not finish within {timeout_s}s. "
                "Try a smaller model (`ollama pull llama3.2:1b`), increase timeout in the sidebar, "
                "or run `ollama run " + model + "` once in a terminal so the model is loaded."
            ) from e
        raise RuntimeError(
            "Cannot reach Ollama. Install from https://ollama.com then run `ollama serve` "
            f"and `ollama pull {model}`. Original error: {e}"
        ) from e
    msg = data.get("message") or {}
    return str(msg.get("content") or data.get("response") or "").strip()


_DEFAULT_OPTIONS = {"num_predict": 256, "temperature": 0.35}
# Smaller num_predict = model stops sooner = less wall time on CPU
_FAST_OPTIONS = {"num_predict": 480, "temperature": 0.35}
_STEP_OPTIONS = {"num_predict": 240, "temperature": 0.35}
_EXPRESS_OPTIONS = {"num_predict": 200, "temperature": 0.25}
_DEEP_OPTIONS = {"num_predict": 640, "temperature": 0.35}


def _chat_options_for_budget(budget: str, *, batched: bool) -> dict[str, Any]:
    """Map UI speed preset to Ollama generation caps (batched = single multi-section call)."""
    b = (budget or "normal").lower()
    if b == "express":
        return _EXPRESS_OPTIONS if batched else {"num_predict": 160, "temperature": 0.25}
    if b == "deep":
        return _DEEP_OPTIONS if batched else {"num_predict": 300, "temperature": 0.35}
    return _FAST_OPTIONS if batched else _STEP_OPTIONS


def _split_tagged_report(text: str) -> dict[str, str]:
    """Parse single-call output with === SECTION === markers."""
    out: dict[str, str] = {k: "" for k in ("qc", "physical", "bio", "report")}
    if "===" not in text:
        out["report"] = text.strip()
        return out
    order = [("qc", "QC"), ("physical", "PHYSICAL"), ("bio", "BIO"), ("report", "SUMMARY")]
    hits: list[tuple[int, str, int]] = []
    for key, label in order:
        m = re.search(rf"===\s*{label}\s*===", text, flags=re.IGNORECASE)
        if m:
            hits.append((m.start(), key, m.end()))
    hits.sort(key=lambda x: x[0])
    for i, (_, key, end) in enumerate(hits):
        nxt = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        out[key] = text[end:nxt].strip()
    if not any(out.values()):
        out["report"] = text.strip()
    return out


def _sys_batched() -> str:
    return (
        "You write ONE response with exactly four labeled parts for mooring ocean data. "
        "Use ONLY the JSON facts; never invent numbers. "
        "Each part: 3–5 short lines, plain English, no # markdown headers.\n\n"
        "Format (copy these lines exactly as section headers):\n"
        "=== QC ===\n"
        "(data coverage, gaps, is file usable)\n"
        "=== PHYSICAL ===\n"
        "(temperature, salinity, conductivity only)\n"
        "=== BIO ===\n"
        "(pH, nitrate, chlorophyll only; pH ~8 is typical seawater)\n"
        "=== SUMMARY ===\n"
        "(5 bullets max for a non-expert reader)\n"
    )


def run_agent_pipeline(
    df,
    base_url: str,
    model: str,
    *,
    fast_mode: bool = True,
    timeout_s: int = 120,
    ai_budget: str = "normal",
) -> dict[str, Any]:
    """
    If fast_mode (default): **one** Ollama call — much faster than four sequential calls.
    If fast_mode False: original QC → Physical → Bio → Editor (4 calls, slower but steadier).
    """
    pkg = build_context_package(df)
    if "error" in pkg:
        return {"error": pkg["error"]}
    ctx = context_json_for_prompt(pkg)

    opts_batched = _chat_options_for_budget(ai_budget, batched=True)
    opts_step = _chat_options_for_budget(ai_budget, batched=False)

    if fast_mode:
        messages = [
            {"role": "system", "content": _sys_batched()},
            {"role": "user", "content": "JSON:\n" + ctx},
        ]
        raw = ollama_chat(base_url, model, messages, timeout_s=timeout_s, options=opts_batched)
        parts = _split_tagged_report(raw)
        return {
            "qc": parts.get("qc") or "(see raw below)",
            "physical": parts.get("physical") or "",
            "bio": parts.get("bio") or "",
            "report": parts.get("report") or raw,
            "context": pkg,
            "raw_model_output": raw,
            "fast_mode": True,
        }

    messages_qc = [
        {
            "role": "system",
            "content": (
                "Data Check agent. ONLY JSON facts. 4–6 short bullets: time range, which columns "
                "are full vs empty, usable for a project. No # headers. No invented numbers."
            ),
        },
        {"role": "user", "content": "JSON:\n" + ctx},
    ]
    qc = ollama_chat(base_url, model, messages_qc, timeout_s=timeout_s, options=opts_step)

    messages_phys = [
        {
            "role": "system",
            "content": (
                "Physical ocean agent. ONLY temperature/salinity/conductivity in JSON. "
                "3–5 bullets. No # headers."
            ),
        },
        {"role": "user", "content": "JSON:\n" + ctx},
    ]
    physical = ollama_chat(base_url, model, messages_phys, timeout_s=timeout_s, options=opts_step)

    messages_bio = [
        {
            "role": "system",
            "content": "Biogeochemistry agent. ONLY pH/nitrate/chlorophyll in JSON. 3–5 bullets. No # headers.",
        },
        {"role": "user", "content": "JSON:\n" + ctx},
    ]
    bio = ollama_chat(base_url, model, messages_bio, timeout_s=timeout_s, options=opts_step)

    prior = f"Data check:\n{qc}\n\nPhysical:\n{physical}\n\nBio:\n{bio}"
    messages_writer = [
        {
            "role": "system",
            "content": (
                "Editor: combine into one short report — title line then 5–7 bullets. "
                "No new numbers. Plain English. No # headers."
            ),
        },
        {"role": "user", "content": prior},
    ]
    report = ollama_chat(base_url, model, messages_writer, timeout_s=timeout_s, options=opts_step)

    return {
        "qc": qc,
        "physical": physical,
        "bio": bio,
        "report": report,
        "context": pkg,
        "fast_mode": False,
    }


def answer_question(
    df, base_url: str, model: str, question: str, timeout_s: int = 90, *, ai_budget: str = "normal"
) -> str:
    """Single follow-up turn: must use JSON facts only."""
    pkg = build_context_package(df)
    if "error" in pkg:
        return "Cannot answer: " + str(pkg["error"])
    ctx = context_json_for_prompt(pkg)
    messages = [
        {
            "role": "system",
            "content": (
                "Answer ONLY using the JSON. If unknown, say: The summary does not say. "
                "Max 6 short sentences. No # markdown headers."
            ),
        },
        {"role": "user", "content": f"Question: {question}\n\nJSON:\n{ctx}"},
    ]
    qopts = _chat_options_for_budget(ai_budget, batched=True)
    # Q&A is a single short reply — cap tokens a bit tighter than multi-section report
    qopts = {**qopts, "num_predict": min(int(qopts.get("num_predict", 256)), 220)}
    return ollama_chat(base_url, model, messages, timeout_s=timeout_s, options=qopts)


_POLISH_OPTIONS = {"num_predict": 220, "temperature": 0.2}


def polish_judge_brief(
    brief_markdown: str,
    base_url: str,
    model: str,
    *,
    timeout_s: int = 60,
) -> str:
    """
    **One** short Ollama call: tighten demo-facing prose only.

    The deterministic ``brief_markdown`` is already complete; this is optional flair for demos.
    Strictly forbid changing numbers so RAG / multi-agent frameworks are unnecessary here.
    """
    clip = (brief_markdown or "").strip()
    if len(clip) > 6500:
        clip = clip[:6500] + "\n\n[… truncated for model input; facts above are complete …]"

    messages = [
        {
            "role": "system",
            "content": (
                "You are an editor. Rewrite the user's markdown to be tighter and clearer. "
                "Rules: Do NOT change any numbers, dates, percentages, mooring ids, inequalities, or column names "
                "inside backticks. Do NOT add new facts. Keep headings roughly similar. "
                "Prefer short bullets. Max ~220 words of new text density."
            ),
        },
        {"role": "user", "content": clip},
    ]
    return ollama_chat(base_url, model, messages, timeout_s=timeout_s, options=_POLISH_OPTIONS)
