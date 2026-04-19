"""Optional cloud LLM completions (Gemini + Groq free tiers) — all behind explicit UI calls."""

from __future__ import annotations

import textwrap

import pandas as pd


def complete_chat(
    user_message: str,
    *,
    provider: str,
    api_key: str,
    model: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.35,
) -> str:
    """
    Single-turn chat completion. ``provider``: ``gemini`` | ``groq``.
    Returns user-facing error string if key missing or package not installed.
    """
    key = (api_key or "").strip()
    if not key:
        return "Add a Gemini or Groq API key in the sidebar (or set GOOGLE_API_KEY / GROQ_API_KEY)."

    p = (provider or "gemini").lower().strip()
    if p in ("claude", "anthropic"):
        p = "gemini"
    if p == "gemini":
        return _gemini_complete(key, user_message, model=model, max_tokens=max_tokens, temperature=temperature)
    if p == "groq":
        return _groq_complete(key, user_message, model=model, max_tokens=max_tokens, temperature=temperature)
    return _gemini_complete(key, user_message, model=model, max_tokens=max_tokens, temperature=temperature)


def _gemini_complete(
    api_key: str,
    user_message: str,
    *,
    model: str | None,
    max_tokens: int,
    temperature: float,
) -> str:
    try:
        import google.generativeai as genai
    except ImportError:
        return "Install Google AI: `pip install google-generativeai` (or `pip install -e \".[llm]\"`)."

    mid = (model or "gemini-1.5-flash").strip()
    try:
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(mid)
        try:
            resp = m.generate_content(
                user_message,
                generation_config={
                    "max_output_tokens": int(max_tokens),
                    "temperature": float(temperature),
                },
            )
        except TypeError:
            resp = m.generate_content(user_message)
        t = getattr(resp, "text", None) or ""
        if not t.strip() and getattr(resp, "candidates", None):
            # blocked / safety
            return "(Empty response — check API key, model id, or safety filters.)"
        return t.strip() or "(empty response)"
    except Exception as e:
        return f"Gemini error: {e}"


def _groq_complete(
    api_key: str,
    user_message: str,
    *,
    model: str | None,
    max_tokens: int,
    temperature: float,
) -> str:
    try:
        from groq import Groq
    except ImportError:
        return "Install Groq: `pip install groq` (or `pip install -e \".[llm]\"`)."

    mid = (model or "llama-3.1-8b-instant").strip()
    try:
        client = Groq(api_key=api_key)
        chat = client.chat.completions.create(
            model=mid,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        msg = chat.choices[0].message
        return (msg.content or "").strip() or "(empty response)"
    except Exception as e:
        return f"Groq error: {e}"


def interpret_top_anomalies_prompt(anomalies_df: pd.DataFrame, *, context_lines: str | None = None) -> str:
    body = anomalies_df.to_string(index=False) if anomalies_df is not None and not anomalies_df.empty else "(no rows)"
    return textwrap.dedent(
        f"""
        You are an oceanographer explaining mooring telemetry to hackathon judges.

        Here are the top multivariate anomalies flagged by an Isolation Forest (most negative score = strangest point in feature space).
        Use ONLY the numbers shown; do not invent timestamps or values.

        ```
        {body}
        ```

        {context_lines or ""}

        Reply with **exactly two sentences**: (1) what kind of oceanographic situation could produce these joint anomalies,
        (2) one concrete next step (e.g. check sensor QA, compare wind vs biology, verify depth assignment).
        """
    ).strip()


def explain_single_anomaly_prompt(*, event_markdown: str, feature_context: str) -> str:
    return textwrap.dedent(
        f"""
        You are an oceanographer coaching hackathon judges.

        **Event snapshot (do not invent numbers):**
        {event_markdown}

        **Feature context:** {feature_context}

        Reply with **exactly two sentences**: likely oceanographic cause for this multivariate deviation,
        then one falsifiable check (instrument, co-located wind, advection, depth assignment).
        """
    ).strip()


def species_correlation_prompt(events_md: str, corr_md: str) -> str:
    return (
        "You are a marine biologist analyzing California Current ecosystem data.\n\n"
        "Here are ocean stress anomaly events detected from CCE mooring sensors:\n"
        f"{events_md}\n\n"
        "Here are species observation counts 2 weeks before vs after each event:\n"
        f"{corr_md}\n\n"
        "In 3-4 sentences, what does this pattern suggest about how these species "
        "respond to ocean acidification and hypoxia stress events?"
    )


def interpret_top_anomalies(
    api_key: str,
    provider: str,
    model: str | None,
    anomalies_df: pd.DataFrame,
    *,
    context_lines: str | None = None,
) -> str:
    prompt = interpret_top_anomalies_prompt(anomalies_df, context_lines=context_lines)
    return complete_chat(
        prompt,
        provider=provider,
        api_key=api_key,
        model=model,
        max_tokens=220,
        temperature=0.35,
    )


def explain_single_anomaly(
    api_key: str,
    provider: str,
    model: str | None,
    *,
    event_markdown: str,
    feature_context: str,
) -> str:
    prompt = explain_single_anomaly_prompt(event_markdown=event_markdown, feature_context=feature_context)
    return complete_chat(
        prompt,
        provider=provider,
        api_key=api_key,
        model=model,
        max_tokens=220,
        temperature=0.35,
    )


def calcofi_mooring_prompt(mooring_summary: str, larvae_summary: str, zoo_summary: str) -> str:
    return textwrap.dedent(
        f"""
        You are a biological oceanographer. CalCOFI ship surveys (larvae/zooplankton) span decades;
        CCE moorings give high-resolution physics/biogeochemistry mostly from ~2009 onward.

        **Mooring-era summary (do not invent numbers beyond this text):**
        {mooring_summary}

        **Larvae data summary:**
        {larvae_summary}

        **Zooplankton data summary:**
        {zoo_summary}

        In 4 short sentences: (1) how to interpret CalCOFI vs mooring time coverage,
        (2) one way they complement each other for ecosystem stress narratives,
        (3) one caution about comparing them directly,
        (4) one judge-friendly takeaway for a hackathon demo.
        """
    ).strip()
