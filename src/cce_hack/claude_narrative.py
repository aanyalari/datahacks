"""Narrative helpers — Gemini + Groq via ``llm_providers`` (free-tier defaults)."""

from __future__ import annotations

import pandas as pd

from cce_hack.llm_providers import (
    calcofi_mooring_prompt,
    complete_chat,
    explain_single_anomaly,
    interpret_top_anomalies,
    species_correlation_prompt,
)


def interpret_species_correlation_llm(
    api_key: str,
    provider: str,
    model: str | None,
    *,
    events_df: pd.DataFrame,
    corr_df: pd.DataFrame,
) -> str:
    ev_md = events_df.head(25).to_markdown(index=False) if events_df is not None and not events_df.empty else "(none)"
    try:
        co_md = corr_df.to_markdown(index=False) if corr_df is not None and not corr_df.empty else "(none)"
    except ImportError:
        co_md = corr_df.to_string(index=False) if corr_df is not None and not corr_df.empty else "(none)"
    prompt = species_correlation_prompt(ev_md, co_md)
    return complete_chat(prompt, provider=provider, api_key=api_key, model=model, max_tokens=400, temperature=0.35)


def calcofi_story_llm(
    api_key: str,
    provider: str,
    model: str | None,
    *,
    mooring_summary: str,
    larvae_summary: str,
    zoo_summary: str,
) -> str:
    prompt = calcofi_mooring_prompt(mooring_summary, larvae_summary, zoo_summary)
    return complete_chat(prompt, provider=provider, api_key=api_key, model=model, max_tokens=450, temperature=0.35)


__all__ = [
    "interpret_top_anomalies",
    "explain_single_anomaly",
    "interpret_species_correlation_llm",
    "calcofi_story_llm",
]
