"""Shared Plotly layout defaults for Streamlit dashboards."""

PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,22,38,0.92)",
    font=dict(family="Segoe UI, system-ui, sans-serif", color="#c8d7f0", size=13),
    title_font=dict(size=16, color="#f0f6ff"),
    margin=dict(l=52, r=24, t=48, b=44),
    colorway=["#3dffe8", "#7aa8ff", "#ffb86b", "#ff7edb", "#b4f397", "#ffd93d"],
)


def apply_plotly(fig):
    fig.update_layout(**PLOTLY_BASE)
    return fig
