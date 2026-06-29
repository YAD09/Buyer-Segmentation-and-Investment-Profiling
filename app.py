from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.segmentation import filter_segments, format_currency, load_raw_data, run_segmentation


COUNTRY_ISO3 = {
    "Australia": "AUS",
    "Belgium": "BEL",
    "Canada": "CAN",
    "Denmark": "DNK",
    "France": "FRA",
    "Germany": "DEU",
    "Mexico": "MEX",
    "Russia": "RUS",
    "UK": "GBR",
    "USA": "USA",
}

LIGHT_THEME = {
    "mode_label": "Light",
    "bg": "#f5f7fb",
    "panel": "#ffffff",
    "panel_alt": "#eef5f4",
    "text": "#10201f",
    "muted": "#5f6f6d",
    "border": "#d9e5e3",
    "accent": "#0f766e",
    "accent_2": "#2563eb",
    "accent_3": "#c2410c",
    "shadow": "0 16px 36px rgba(15, 23, 42, 0.10)",
    "plot_template": "plotly_white",
    "colors": ["#0f766e", "#2563eb", "#c2410c", "#7c3aed", "#0e7490", "#be123c"],
}

DARK_THEME = {
    "mode_label": "Dark",
    "bg": "#0b1120",
    "panel": "#111827",
    "panel_alt": "#172033",
    "text": "#eef6f5",
    "muted": "#a7b7b5",
    "border": "#263547",
    "accent": "#2dd4bf",
    "accent_2": "#60a5fa",
    "accent_3": "#fb923c",
    "shadow": "0 18px 42px rgba(0, 0, 0, 0.35)",
    "plot_template": "plotly_dark",
    "colors": ["#2dd4bf", "#60a5fa", "#fb923c", "#a78bfa", "#22d3ee", "#f472b6"],
}


st.set_page_config(
    page_title="Buyer Segmentation and Investment Profiling",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_and_segment(clients_path: str, properties_path: str, n_clusters: int):
    clients, properties = load_raw_data(clients_path, properties_path)
    result = run_segmentation(clients, properties, n_clusters=n_clusters)
    return result.data, result.cluster_profiles, result.model_metrics


def inject_css(theme: dict[str, str]) -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --app-bg: {theme["bg"]};
            --panel: {theme["panel"]};
            --panel-alt: {theme["panel_alt"]};
            --text: {theme["text"]};
            --muted: {theme["muted"]};
            --border: {theme["border"]};
            --accent: {theme["accent"]};
            --accent-2: {theme["accent_2"]};
            --accent-3: {theme["accent_3"]};
            --shadow: {theme["shadow"]};
        }}

        .stApp {{
            background:
                radial-gradient(circle at top left, color-mix(in srgb, var(--accent) 20%, transparent), transparent 28rem),
                linear-gradient(135deg, var(--app-bg) 0%, var(--app-bg) 48%, color-mix(in srgb, var(--accent-2) 10%, var(--app-bg)) 100%);
            color: var(--text);
        }}

        header[data-testid="stHeader"] {{
            background: var(--app-bg);
            border-bottom: 1px solid var(--border);
        }}

        [data-testid="stToolbar"],
        [data-testid="stToolbar"] div,
        [data-testid="stToolbar"] button {{
            color: var(--text);
        }}

        [data-testid="stDecoration"] {{
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, var(--panel) 0%, var(--panel-alt) 100%);
            border-right: 1px solid var(--border);
        }}

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {{
            color: var(--text);
        }}

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: var(--text);
        }}

        [data-testid="stSidebar"] .stMarkdown {{
            color: var(--text);
        }}

        [data-testid="stTextInput"] div[data-baseweb="input"],
        [data-testid="stTextInput"] input {{
            background-color: var(--panel);
            color: var(--text);
            border-color: var(--border);
            box-shadow: none;
        }}

        [data-testid="stTextInput"] input:focus {{
            border-color: var(--accent);
            box-shadow: 0 0 0 1px var(--accent);
        }}

        div[data-baseweb="select"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] div {{
            background-color: var(--panel);
            color: var(--text);
            border-color: var(--border);
        }}

        div[data-baseweb="select"] svg {{
            fill: var(--text);
        }}

        div[data-baseweb="popover"],
        div[data-baseweb="popover"] ul,
        div[data-baseweb="menu"],
        ul[role="listbox"] {{
            background-color: var(--panel);
            color: var(--text);
            border: 1px solid var(--border);
        }}

        li[role="option"],
        div[role="option"] {{
            background-color: var(--panel);
            color: var(--text);
        }}

        li[role="option"]:hover,
        div[role="option"]:hover {{
            background-color: var(--panel-alt);
        }}

        [data-testid="stSlider"] [data-baseweb="slider"] div {{
            color: var(--text);
        }}

        [data-testid="stSlider"] [role="slider"] {{
            background-color: var(--accent);
            border-color: var(--accent);
        }}

        .block-container {{
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1420px;
        }}

        .hero {{
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.35rem 1.5rem;
            margin-bottom: 1.1rem;
            background:
                linear-gradient(135deg, color-mix(in srgb, var(--panel) 92%, var(--accent) 8%), var(--panel));
            box-shadow: var(--shadow);
        }}

        .hero-eyebrow {{
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }}

        .hero h1 {{
            color: var(--text);
            font-size: clamp(2rem, 4vw, 3.2rem);
            line-height: 1.05;
            margin: 0;
            letter-spacing: 0;
        }}

        .hero p {{
            color: var(--muted);
            max-width: 880px;
            margin: 0.75rem 0 0;
            font-size: 1rem;
        }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
            margin: 0.4rem 0 1rem;
        }}

        .metric-card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            box-shadow: var(--shadow);
            min-width: 0;
        }}

        .metric-label {{
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }}

        .metric-value {{
            color: var(--text);
            font-size: clamp(1.45rem, 2.2vw, 2.05rem);
            line-height: 1.1;
            overflow-wrap: anywhere;
        }}

        @media (max-width: 1100px) {{
            .metric-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
        }}

        @media (max-width: 620px) {{
            .metric-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.4rem;
            border-bottom: 1px solid var(--border);
        }}

        .stTabs [data-baseweb="tab"] {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-bottom: 0;
            border-radius: 8px 8px 0 0;
            color: var(--muted);
            font-weight: 700;
        }}

        .stTabs [aria-selected="true"] {{
            color: var(--text);
            background: linear-gradient(180deg, color-mix(in srgb, var(--accent) 16%, var(--panel)), var(--panel));
        }}

        div[data-testid="stPlotlyChart"] {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.65rem;
            box-shadow: var(--shadow);
        }}

        .js-plotly-plot .plotly .modebar-btn svg {{
            fill: var(--muted);
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: var(--shadow);
        }}

        .table-wrap {{
            overflow-x: auto;
            border: 1px solid var(--border);
            border-radius: 8px;
            box-shadow: var(--shadow);
            background: var(--panel);
            margin-bottom: 1rem;
        }}

        .insight-table {{
            width: 100%;
            border-collapse: collapse;
            min-width: 980px;
            color: var(--text);
            background: var(--panel);
        }}

        .insight-table th {{
            background: var(--panel-alt);
            color: var(--muted);
            font-size: 0.78rem;
            text-align: left;
            padding: 0.75rem 0.7rem;
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
        }}

        .insight-table td {{
            color: var(--text);
            padding: 0.78rem 0.7rem;
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
        }}

        .insight-table tr:last-child td {{
            border-bottom: 0;
        }}

        .insight-table td.numeric,
        .insight-table th.numeric {{
            text-align: right;
        }}

        .stDownloadButton button,
        .stButton button {{
            border-radius: 8px;
            border: 1px solid color-mix(in srgb, var(--accent) 45%, var(--border));
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            color: #ffffff;
            font-weight: 800;
        }}

        h2, h3 {{
            color: var(--text);
            letter-spacing: 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(theme: dict[str, str]) -> None:
    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-eyebrow">Parcl Real Estate Market Intelligence</div>
            <h1>Buyer Segmentation and Investment Profiling</h1>
            <p>
                K-Means and hierarchical clustering trained on the provided client and property CSV files.
                Current view is using {theme["mode_label"].lower()} mode with interactive buyer, geography,
                finance, and segment diagnostics.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig: go.Figure, theme: dict[str, str]) -> go.Figure:
    fig.update_layout(
        template=theme["plot_template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=theme["text"],
        title_font=dict(size=18, color=theme["text"]),
        colorway=theme["colors"],
        margin=dict(l=20, r=20, t=58, b=35),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=theme["text"])),
    )
    fig.update_xaxes(gridcolor=theme["border"], zerolinecolor=theme["border"])
    fig.update_yaxes(gridcolor=theme["border"], zerolinecolor=theme["border"])
    return fig


def option_list(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).unique().tolist())


def sidebar_filters(data: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    st.sidebar.header("Filters")
    countries = st.sidebar.multiselect("Country", option_list(data["country"]))
    regions = st.sidebar.multiselect("Region", option_list(data["region"]))
    purposes = st.sidebar.multiselect("Acquisition purpose", option_list(data["acquisition_purpose"]))
    client_types = st.sidebar.multiselect("Client type", option_list(data["client_type"]))
    return countries, regions, purposes, client_types


def metric_row(data: pd.DataFrame) -> None:
    total_buyers = len(data)
    total_value = data["total_investment"].sum()
    loan_rate = data["loan_applied_flag"].mean() if total_buyers else 0
    investment_rate = data["investment_buyer"].mean() if total_buyers else 0

    cards = [
        ("Buyers", f"{total_buyers:,}"),
        ("Linked investment", f"${total_value / 1_000_000_000:.2f}B"),
        ("Loan rate", f"{loan_rate:.1%}"),
        ("Investment-purpose buyers", f"{investment_rate:.1%}"),
    ]
    html = ['<div class="metric-grid">']
    for label, value in cards:
        html.append(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def show_overview(data: pd.DataFrame, metrics: pd.DataFrame, theme: dict[str, str]) -> None:
    left, right = st.columns([1.1, 1])
    with left:
        counts = data["segment_name"].value_counts().reset_index()
        counts.columns = ["Segment", "Buyers"]
        fig = px.bar(
            counts,
            x="Segment",
            y="Buyers",
            color="Segment",
            title="Buyer Segment Distribution",
            text_auto=True,
            color_discrete_sequence=theme["colors"],
        )
        fig.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(style_chart(fig, theme), width="stretch")

    with right:
        fig = px.line(
            metrics,
            x="k",
            y="inertia",
            markers=True,
            title="Elbow Method: K-Means Inertia",
            color_discrete_sequence=[theme["accent"]],
        )
        fig.update_layout(xaxis_title="Number of clusters", yaxis_title="Inertia")
        st.plotly_chart(style_chart(fig, theme), width="stretch")

    fig = px.line(
        metrics,
        x="k",
        y="silhouette_score",
        markers=True,
        title="Silhouette Score by Cluster Count",
        color_discrete_sequence=[theme["accent_2"]],
    )
    fig.update_layout(xaxis_title="Number of clusters", yaxis_title="Silhouette score")
    st.plotly_chart(style_chart(fig, theme), width="stretch")


def show_investor_behavior(data: pd.DataFrame, theme: dict[str, str]) -> None:
    investment = (
        data.groupby("segment_name")
        .agg(
            buyers=("client_id", "count"),
            avg_total_investment=("total_investment", "mean"),
            avg_sale_price=("avg_sale_price", "mean"),
            investment_rate=("investment_buyer", "mean"),
            loan_rate=("loan_applied_flag", "mean"),
        )
        .reset_index()
    )

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            investment,
            x="segment_name",
            y="avg_total_investment",
            color="segment_name",
            title="Average Linked Investment by Segment",
            text_auto=".2s",
            color_discrete_sequence=theme["colors"],
        )
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Average investment")
        st.plotly_chart(style_chart(fig, theme), width="stretch")

    with right:
        long_rates = investment.melt(
            id_vars="segment_name",
            value_vars=["investment_rate", "loan_rate"],
            var_name="Metric",
            value_name="Rate",
        )
        long_rates["Metric"] = long_rates["Metric"].replace(
            {"investment_rate": "Investment purpose", "loan_rate": "Loan applied"}
        )
        fig = px.bar(
            long_rates,
            x="segment_name",
            y="Rate",
            color="Metric",
            barmode="group",
            title="Investment and Financing Behavior",
            color_discrete_sequence=[theme["accent"], theme["accent_3"]],
        )
        fig.update_layout(xaxis_title=None, yaxis_tickformat=".0%")
        st.plotly_chart(style_chart(fig, theme), width="stretch")

    purpose_mix = data.groupby(["segment_name", "acquisition_purpose"]).size().reset_index(name="buyers")
    fig = px.sunburst(
        purpose_mix,
        path=["segment_name", "acquisition_purpose"],
        values="buyers",
        title="Acquisition Purpose Mix by Segment",
        color_discrete_sequence=theme["colors"],
    )
    st.plotly_chart(style_chart(fig, theme), width="stretch")


def show_geography(data: pd.DataFrame, theme: dict[str, str]) -> None:
    country_summary = (
        data.groupby("country")
        .agg(buyers=("client_id", "count"), investment_value=("total_investment", "sum"))
        .reset_index()
    )
    country_summary["iso_alpha"] = country_summary["country"].map(COUNTRY_ISO3)
    fig = px.choropleth(
        country_summary,
        locations="iso_alpha",
        color="buyers",
        hover_name="country",
        hover_data={"iso_alpha": False, "investment_value": ":,.0f", "buyers": True},
        color_continuous_scale="Tealgrn" if theme["mode_label"] == "Light" else "Aggrnyl",
        title="Buyer Concentration by Country",
    )
    st.plotly_chart(style_chart(fig, theme), width="stretch")

    region_segment = data.groupby(["region", "segment_name"]).size().reset_index(name="buyers")
    top_regions = (
        region_segment.groupby("region")["buyers"].sum().sort_values(ascending=False).head(20).index
    )
    region_segment = region_segment[region_segment["region"].isin(top_regions)]
    fig = px.bar(
        region_segment,
        x="region",
        y="buyers",
        color="segment_name",
        title="Top Regions by Buyer Segment",
        color_discrete_sequence=theme["colors"],
    )
    fig.update_layout(xaxis_title=None, yaxis_title="Buyers")
    st.plotly_chart(style_chart(fig, theme), width="stretch")


def show_segment_insights(data: pd.DataFrame, profiles: pd.DataFrame, theme: dict[str, str]) -> None:
    visible_profiles = profiles[profiles["segment_name"].isin(data["segment_name"].unique())].copy()
    columns = [
        ("segment_name", "Segment", ""),
        ("buyers", "Buyers", "numeric"),
        ("avg_age", "Avg age", "numeric"),
        ("avg_satisfaction", "Avg satisfaction", "numeric"),
        ("loan_rate", "Loan rate", "numeric"),
        ("investment_rate", "Investment rate", "numeric"),
        ("corporate_rate", "Corporate rate", "numeric"),
        ("international_rate", "International rate", "numeric"),
        ("avg_property_count", "Avg properties", "numeric"),
        ("avg_total_investment", "Avg investment", "numeric"),
        ("avg_sale_price", "Avg sale price", "numeric"),
    ]

    def format_profile_value(column: str, value: object) -> str:
        if column in {"loan_rate", "investment_rate", "corporate_rate", "international_rate"}:
            return f"{float(value):.1%}"
        if column in {"avg_total_investment", "avg_sale_price"}:
            return format_currency(float(value))
        if column in {"avg_age", "avg_satisfaction", "avg_property_count"}:
            return f"{float(value):.2f}" if column == "avg_property_count" else f"{float(value):.1f}"
        if column == "buyers":
            return f"{int(value):,}"
        return str(value)

    header_cells = "".join(
        f'<th class="{css_class}">{html.escape(label)}</th>' for _, label, css_class in columns
    )
    body_rows = []
    for _, row in visible_profiles.iterrows():
        cells = []
        for column, _, css_class in columns:
            formatted = html.escape(format_profile_value(column, row[column]))
            cells.append(f'<td class="{css_class}">{formatted}</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    st.markdown(
        f"""
        <div class="table-wrap">
            <table class="insight-table">
                <thead><tr>{header_cells}</tr></thead>
                <tbody>{''.join(body_rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_segment = st.selectbox("Segment detail", option_list(data["segment_name"]))
    segment_data = data[data["segment_name"] == selected_segment]
    left, right = st.columns(2)
    with left:
        fig = px.histogram(
            segment_data,
            x="age",
            nbins=20,
            title=f"Age Distribution: {selected_segment}",
            color_discrete_sequence=[theme["accent"]],
        )
        st.plotly_chart(style_chart(fig, theme), width="stretch")
    with right:
        fig = px.box(
            segment_data,
            x="segment_name",
            y="total_investment",
            title=f"Investment Range: {selected_segment}",
            color_discrete_sequence=[theme["accent_2"]],
        )
        fig.update_layout(xaxis_title=None, yaxis_title="Total investment")
        st.plotly_chart(style_chart(fig, theme), width="stretch")

    st.download_button(
        "Download filtered buyer segments",
        data.to_csv(index=False).encode("utf-8"),
        file_name="filtered_buyer_segments.csv",
        mime="text/csv",
    )


def main() -> None:
    default_clients = Path("data/clients.csv")
    default_properties = Path("data/properties.csv")

    dark_mode = st.sidebar.toggle("Dark mode", value=False)
    theme = DARK_THEME if dark_mode else LIGHT_THEME
    inject_css(theme)
    render_hero(theme)

    st.sidebar.header("Data")
    clients_path = st.sidebar.text_input("Clients CSV", str(default_clients))
    properties_path = st.sidebar.text_input("Properties CSV", str(default_properties))
    n_clusters = st.sidebar.slider("K-Means clusters", min_value=2, max_value=8, value=4)

    data, profiles, metrics = load_and_segment(clients_path, properties_path, n_clusters)
    countries, regions, purposes, client_types = sidebar_filters(data)
    filtered = filter_segments(data, countries, regions, purposes, client_types)

    if filtered.empty:
        st.warning("No buyers match the selected filters.")
        return

    metric_row(filtered)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Buyer Segmentation Overview",
            "Investor Behavior Dashboard",
            "Geographic Buyer Analysis",
            "Segment Insights Panel",
        ]
    )

    with tab1:
        show_overview(filtered, metrics, theme)
    with tab2:
        show_investor_behavior(filtered, theme)
    with tab3:
        show_geography(filtered, theme)
    with tab4:
        show_segment_insights(filtered, profiles, theme)


if __name__ == "__main__":
    main()
