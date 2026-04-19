"""One-page briefing: filled from computed facts only (no LLM)."""

from __future__ import annotations

from typing import Any


def build_judge_brief_markdown(pkg: dict[str, Any]) -> str:
    """
    One-page narrative: problem → what the app did → evidence → caveats.

    Every quantitative claim is derived from ``build_context_package`` output.
    """
    if "error" in pkg:
        return f"## Briefing\n\n**Cannot build briefing:** {pkg['error']}"

    mooring = pkg.get("mooring_id")
    rows = int(pkg.get("rows") or 0)
    tr = pkg.get("time_range_utc") or {}
    t0, t1 = tr.get("start", "—"), tr.get("end", "—")
    cov = pkg.get("coverage_by_column") or {}

    fills: list[tuple[str, float]] = []
    for col, info in cov.items():
        fills.append((col, float(info.get("filled_percent") or 0)))
    fills.sort(key=lambda x: x[1])

    weakest = fills[:5]
    strongest = sorted(fills, key=lambda x: -x[1])[:5]
    low = [c for c, p in fills if p < 20.0]
    high = [c for c, p in fills if p >= 85.0]
    median_fill = sorted(p for _, p in fills)[len(fills) // 2] if fills else 0.0

    lines: list[str] = []
    lines.append("## Briefing")
    lines.append("")
    lines.append("### 1) Real use case (why this exists)")
    lines.append(
        "Ocean **moorings** (fixed buoys) stream temperature, salinity, biology, and chemistry for months or years. "
        "Teams merge exports into wide CSVs where **each sensor has its own gaps and start/stop dates**. "
        "Before anyone plots trends or trains models, someone has to ask: **Which columns are actually usable, and for which calendar window?** "
        "This app automates that first-line **data trust** check and pairs it with exploratory plots and optional wording polish."
    )
    lines.append("")
    lines.append("### 2) What to look for in ~60 seconds")
    lines.append(
        "- **Speed & clarity:** Does the briefing below make the file’s strengths/weaknesses obvious without domain jargon?\n"
        "- **Honesty:** Numbers come straight from the CSV summary — nothing is “live from the ocean.”\n"
        "- **Depth on demand:** Charts and the analysis lab are there if you want evidence behind the headline."
    )
    lines.append("")
    lines.append("### 3) Headline facts (from this upload / built-in file)")
    head = f"- **Rows (time steps):** {rows:,}\n- **Overall UTC window:** `{t0}` → `{t1}`"
    if mooring:
        head = f"- **Mooring id:** `{mooring}`\n" + head
    lines.append(head)
    lines.append("")
    lines.append(
        f"- **Typical column completeness (median filled):** {median_fill:.0f}% "
        f"— interpret together with the table in **Data quality**."
    )
    if high:
        lines.append(f"- **Columns that look strong (≥85% filled):** {', '.join(f'`{c}`' for c in high[:8])}")
    if low:
        lines.append(
            f"- **Columns that look weak (<20% filled — risky for analysis):** {', '.join(f'`{c}`' for c in low[:10])}"
            + (" …" if len(low) > 10 else "")
        )
    lines.append("")
    lines.append("### 4) Worst and best covered signals (by % filled)")
    lines.append("| Column | Filled % | First date | Last date |")
    lines.append("|---|---:|---|---|")
    for col, pct in weakest:
        info = cov.get(col, {})
        lines.append(
            f"| `{col}` | {pct:.1f}% | {info.get('first_date', '—')} | {info.get('last_date', '—')} |"
        )
    lines.append("")
    lines.append("**Strongest (sample):** " + ", ".join(f"`{c}` ({p:.0f}%)" for c, p in strongest[:5]) + "")
    lines.append("")

    def _block_md(title: str, block: dict[str, Any]) -> None:
        if not block:
            return
        lines.append(f"### {title}")
        for col, stats in block.items():
            if isinstance(stats, dict) and stats.get("note"):
                lines.append(f"- **{col}:** {stats['note']}")
                continue
            if not isinstance(stats, dict):
                continue
            lines.append(
                f"- **{col}:** mean {stats.get('mean')}, std {stats.get('std')}, "
                f"min {stats.get('min')}, max {stats.get('max')}"
            )
        lines.append("")

    _block_md("5) Physical ocean snapshot (from file)", pkg.get("physical_summary") or {})
    _block_md("6) Biogeochemistry snapshot (from file)", pkg.get("biogeochem_summary") or {})

    lines.append("### 7) Suggested live demo path")
    lines.append(
        "1. Skim this briefing (you are here).\n"
        "2. Open **Data quality** if someone asks *prove it* — same facts as a table.\n"
        "3. Open **Time series** and pick 2–3 variables with high fill %.\n"
        "4. Mention that **Optional LLM** only rewrites text; it is not the source of truth."
    )
    lines.append("")
    note = pkg.get("note_for_agents")
    if note:
        lines.append("### Note on methodology")
        lines.append(f"_{note}_")
        lines.append("")

    lines.append("---")
    lines.append(
        "*This page is a **template filled in Python** from the JSON summary. "
        "No vector DB / RAG is required because the “knowledge” is already this table-level audit.*"
    )
    return "\n".join(lines)
