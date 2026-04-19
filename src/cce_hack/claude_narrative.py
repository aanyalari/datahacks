"""Short scientific blurbs via Anthropic Messages API (optional; needs network + key)."""

from __future__ import annotations

import textwrap

import pandas as pd


def interpret_top_anomalies_claude(
    api_key: str,
    model: str,
    anomalies_df: pd.DataFrame,
    *,
    context_lines: str | None = None,
) -> str:
    """
    Two-sentence scientific interpretation of the top anomaly rows (numbers preserved in user message).
    """
    if not api_key.strip():
        return "Set **ANTHROPIC_API_KEY** in the sidebar (or environment) to enable Claude."

    try:
        import anthropic
    except ImportError:
        return "Install the `anthropic` package: `pip install anthropic`."

    body = anomalies_df.to_string(index=False) if anomalies_df is not None and not anomalies_df.empty else "(no rows)"
    user = textwrap.dedent(
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

    client = anthropic.Anthropic(api_key=api_key.strip())
    msg = client.messages.create(
        model=model.strip(),
        max_tokens=220,
        temperature=0.35,
        messages=[{"role": "user", "content": user}],
    )
    parts = []
    for b in msg.content:
        if getattr(b, "type", None) == "text":
            parts.append(b.text)
    return "\n".join(parts).strip() or "(empty model response)"


def explain_single_anomaly_claude(
    api_key: str,
    model: str,
    *,
    event_markdown: str,
    feature_context: str,
) -> str:
    """One anomaly row / event — likely oceanographic cause (two short sentences)."""
    if not api_key.strip():
        return "Set **ANTHROPIC_API_KEY** in the sidebar (or environment) to enable Claude."
    try:
        import anthropic
    except ImportError:
        return "Install the `anthropic` package: `pip install anthropic`."

    user = textwrap.dedent(
        f"""
        You are an oceanographer coaching hackathon judges.

        **Event snapshot (do not invent numbers):**
        {event_markdown}

        **Feature context:** {feature_context}

        Reply with **exactly two sentences**: likely oceanographic cause for this multivariate deviation,
        then one falsifiable check (instrument, co-located wind, advection, depth assignment).
        """
    ).strip()

    client = anthropic.Anthropic(api_key=api_key.strip())
    msg = client.messages.create(
        model=model.strip(),
        max_tokens=220,
        temperature=0.35,
        messages=[{"role": "user", "content": user}],
    )
    parts = []
    for b in msg.content:
        if getattr(b, "type", None) == "text":
            parts.append(b.text)
    return "\n".join(parts).strip() or "(empty model response)"
