"""
Customer Churn Prediction — Interactive Dashboard
Built with Streamlit + Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color Palette ────────────────────────────────────────────────────────────
COLORS = {
    "Won": "#2ECC71",
    "Churned": "#E74C3C",
    "Open": "#F39C12",
    "bg_dark": "#0E1117",
    "card_bg": "#1A1D23",
    "card_border": "#2D3139",
    "text_primary": "#FAFAFA",
    "text_secondary": "#8B949E",
    "accent": "#3498DB",
    "accent2": "#9B59B6",
    "gradient_start": "#667eea",
    "gradient_end": "#764ba2",
}

OUTCOME_COLOR_MAP = {
    "Won": COLORS["Won"],
    "Churned": COLORS["Churned"],
    "Open": COLORS["Open"],
}

OUTCOME_ORDER = ["Won", "Churned", "Open"]

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1A1D23 0%, #22262E 100%);
        border: 1px solid #2D3139;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        border-color: #3498DB;
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 8px 0;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .metric-subtitle {
        font-size: 0.75rem;
        color: #6B7280;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #FAFAFA;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #3498DB;
        display: inline-block;
    }

    /* Page title */
    .page-title {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: #8B949E;
        margin-bottom: 24px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #161B22 100%);
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: #8B949E;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
    }

    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #2D3139, transparent);
        margin: 24px 0;
    }

    /* Model highlight */
    .model-best {
        background: linear-gradient(135deg, rgba(46,204,113,0.15) 0%, rgba(52,152,219,0.15) 100%);
        border: 1px solid rgba(46,204,113,0.3);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
    }

    /* Insight box */
    .insight-box {
        background: linear-gradient(135deg, rgba(52,152,219,0.1) 0%, rgba(155,89,182,0.1) 100%);
        border-left: 3px solid #3498DB;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #B0B8C4;
    }

    /* Streamlit elements styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }

    /* Plotly chart containers */
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")


@st.cache_data(ttl=3600)
def load_master():
    path = os.path.join(DATA_DIR, "master_churn_dataset.csv")
    df = pd.read_csv(path, low_memory=False)
    # Parse dates safely
    for col in ["Renewal_Month", "Prospect_Renewal_Date", "Closed_Date", "DateTime_Out"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # Derived helpers
    df["Is_Churned"] = (df["Prospect_Outcome"] == "Churned").astype(int)
    return df


@st.cache_data(ttl=3600)
def load_model_ready():
    path = os.path.join(DATA_DIR, "model_ready_dataset.csv")
    return pd.read_csv(path, low_memory=False)


def get_model_metrics():
    """Hardcoded from notebooks/Model-training/model_comparison_summary.md"""
    return pd.DataFrame([
        {"Model": "XGBoost", "Variant": "Tuned", "Accuracy": 0.9565, "Precision": 0.8024, "Recall": 0.8074, "F1": 0.8049, "ROC_AUC": 0.9806},
        {"Model": "LightGBM", "Variant": "Tuned", "Accuracy": 0.9549, "Precision": 0.7895, "Recall": 0.8110, "F1": 0.8001, "ROC_AUC": 0.9808},
        {"Model": "Gradient Boosting", "Variant": "Tuned", "Accuracy": 0.9532, "Precision": 0.7754, "Recall": 0.8149, "F1": 0.7947, "ROC_AUC": 0.9777},
        {"Model": "Decision Tree", "Variant": "SMOTE", "Accuracy": 0.9432, "Precision": 0.7106, "Recall": 0.8256, "F1": 0.7638, "ROC_AUC": None},
        {"Model": "AdaBoost", "Variant": "Tuned", "Accuracy": 0.9353, "Precision": 0.6640, "Recall": 0.8477, "F1": 0.7447, "ROC_AUC": 0.9711},
        {"Model": "Random Forest", "Variant": "No SMOTE", "Accuracy": 0.9421, "Precision": 0.9037, "Recall": 0.5371, "F1": 0.6738, "ROC_AUC": None},
        {"Model": "Logistic Regression", "Variant": "Baseline", "Accuracy": 0.8909, "Precision": 0.5056, "Recall": 0.8571, "F1": 0.6360, "ROC_AUC": 0.9529},
        {"Model": "KNN", "Variant": "Tuned", "Accuracy": 0.9187, "Precision": 0.6740, "Recall": 0.5221, "F1": 0.5884, "ROC_AUC": 0.8845},
        {"Model": "Naive Bayes", "Variant": "Baseline", "Accuracy": 0.8743, "Precision": 0.4595, "Recall": 0.7356, "F1": 0.5656, "ROC_AUC": 0.9028},
        {"Model": "SVM", "Variant": "Tuned", "Accuracy": 0.8508, "Precision": 0.4132, "Recall": 0.8118, "F1": 0.5477, "ROC_AUC": 0.9157},
    ])


# ── Plotly Theme Helper ──────────────────────────────────────────────────────
def apply_chart_theme(fig, height=400):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#FAFAFA"),
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    )
    return fig


# ── Helper: KPI Card HTML ────────────────────────────────────────────────────
def kpi_card(label, value, color="#FAFAFA", subtitle=""):
    subtitle_html = f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color};">{value}</div>
        {subtitle_html}
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size:2.5rem;">📊</div>
        <div style="font-size:1.1rem; font-weight:800; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Churn Prediction
        </div>
        <div style="font-size:0.75rem; color:#8B949E; margin-top:4px;">Interactive Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    page = st.selectbox(
        "NAVIGATE",
        [
            "📊 Executive Summary",
            "👤 Customer Profile",
            "📞 Interaction Analysis",
            "🤖 Model Performance",
            "🔍 Risk Drilldown",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Load data
    master = load_master()
    model_ready = load_model_ready()

    # Global filters
    st.markdown('<div class="metric-label" style="margin-bottom:8px;">FILTERS</div>', unsafe_allow_html=True)

    years = sorted(master["Renewal_Year"].dropna().unique().astype(int))
    selected_years = st.multiselect("Renewal Year", years, default=years)

    bands = sorted(master["Band"].dropna().unique())
    selected_bands = st.multiselect("Band", bands, default=bands)

    tenure_groups = sorted(master["Tenure_Group"].dropna().unique())
    selected_tenure = st.multiselect("Tenure Group", tenure_groups, default=tenure_groups)

    # Apply filters
    mask = (
        master["Renewal_Year"].isin(selected_years)
        & master["Band"].isin(selected_bands)
        & master["Tenure_Group"].isin(selected_tenure)
    )
    df = master[mask].copy()

    # Also filter model_ready (same row indices)
    df_mr = model_ready[mask].copy()

    # Binary subset (exclude Open)
    df_binary = df[df["Prospect_Outcome"].isin(["Won", "Churned"])].copy()

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center; font-size:0.75rem; color:#6B7280;">
        Showing <strong style="color:#3498DB;">{len(df):,}</strong> of {len(master):,} records
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Summary":
    st.markdown('<div class="page-title">Executive Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">High-level overview of customer churn health across all renewal periods</div>', unsafe_allow_html=True)

    # KPI row
    total = len(df)
    won = len(df[df["Prospect_Outcome"] == "Won"])
    churned = len(df[df["Prospect_Outcome"] == "Churned"])
    open_count = len(df[df["Prospect_Outcome"] == "Open"])
    binary_total = won + churned
    churn_rate = churned / binary_total if binary_total > 0 else 0
    retention_rate = 1 - churn_rate

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(kpi_card("Total Customers", f"{total:,}", COLORS["accent"]), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Won", f"{won:,}", COLORS["Won"], "Retained"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Churned", f"{churned:,}", COLORS["Churned"], "Lost"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Open", f"{open_count:,}", COLORS["Open"], "Pending"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("Churn Rate", f"{churn_rate:.1%}", COLORS["Churned"], "Won+Churned basis"), unsafe_allow_html=True)
    with c6:
        st.markdown(kpi_card("Retention Rate", f"{retention_rate:.1%}", COLORS["Won"], "Won+Churned basis"), unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Charts row 1
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="section-header">Outcome Distribution</div>', unsafe_allow_html=True)
        outcome_counts = df["Prospect_Outcome"].value_counts().reindex(OUTCOME_ORDER).fillna(0)
        fig_donut = go.Figure(data=[go.Pie(
            labels=outcome_counts.index,
            values=outcome_counts.values,
            hole=0.65,
            marker=dict(colors=[OUTCOME_COLOR_MAP[o] for o in outcome_counts.index]),
            textinfo="label+percent",
            textfont=dict(size=13, family="Inter"),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        )])
        fig_donut.update_layout(showlegend=False)
        fig_donut.add_annotation(
            text=f"<b>{total:,}</b><br><span style='font-size:11px;color:#8B949E;'>Total</span>",
            showarrow=False, font=dict(size=22, color="#FAFAFA", family="Inter"),
        )
        apply_chart_theme(fig_donut, height=380)
        st.plotly_chart(fig_donut, width="stretch")

    with col2:
        st.markdown('<div class="section-header">Churn Trend by Renewal Year</div>', unsafe_allow_html=True)
        yearly = df.groupby(["Renewal_Year", "Prospect_Outcome"]).size().reset_index(name="Count")
        fig_trend = px.bar(
            yearly, x="Renewal_Year", y="Count", color="Prospect_Outcome",
            color_discrete_map=OUTCOME_COLOR_MAP,
            category_orders={"Prospect_Outcome": OUTCOME_ORDER},
            barmode="stack",
        )
        fig_trend.update_traces(marker_line_width=0)
        fig_trend.update_xaxes(title="", dtick=1)
        fig_trend.update_yaxes(title="Number of Customers")
        apply_chart_theme(fig_trend, 380)
        st.plotly_chart(fig_trend, width="stretch")

    # Charts row 2
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Churn Rate Over Years</div>', unsafe_allow_html=True)
        yr_churn = df_binary.groupby("Renewal_Year").agg(
            churned=("Is_Churned", "sum"),
            total=("Is_Churned", "count"),
        ).reset_index()
        yr_churn["Churn_Rate"] = yr_churn["churned"] / yr_churn["total"]
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=yr_churn["Renewal_Year"], y=yr_churn["Churn_Rate"],
            mode="lines+markers",
            line=dict(color=COLORS["Churned"], width=3),
            marker=dict(size=10, color=COLORS["Churned"], line=dict(width=2, color="#FAFAFA")),
            hovertemplate="Year: %{x}<br>Churn Rate: %{y:.1%}<extra></extra>",
            fill="tozeroy",
            fillcolor="rgba(231,76,60,0.1)",
        ))
        fig_line.update_xaxes(title="", dtick=1)
        fig_line.update_yaxes(title="Churn Rate", tickformat=".0%")
        apply_chart_theme(fig_line, 350)
        st.plotly_chart(fig_line, width="stretch")

    with col4:
        st.markdown('<div class="section-header">Revenue Distribution by Outcome</div>', unsafe_allow_html=True)
        fig_box = px.box(
            df_binary, x="Prospect_Outcome", y="Total_Net_Paid",
            color="Prospect_Outcome",
            color_discrete_map=OUTCOME_COLOR_MAP,
            category_orders={"Prospect_Outcome": ["Won", "Churned"]},
        )
        fig_box.update_xaxes(title="")
        fig_box.update_yaxes(title="Total Net Paid (£)")
        fig_box.update_layout(showlegend=False)
        apply_chart_theme(fig_box, 350)
        st.plotly_chart(fig_box, width="stretch")

    # Insight box
    avg_rev_won = df[df["Prospect_Outcome"] == "Won"]["Total_Net_Paid"].mean()
    avg_rev_churn = df[df["Prospect_Outcome"] == "Churned"]["Total_Net_Paid"].mean()
    st.markdown(f"""
    <div class="insight-box">
        💡 <strong>Key Insight:</strong> The average revenue from retained customers (£{avg_rev_won:,.0f}) is
        <strong>£{avg_rev_won - avg_rev_churn:,.0f}</strong> higher than churned customers (£{avg_rev_churn:,.0f}).
        The binary churn rate stands at <strong>{churn_rate:.1%}</strong> across the selected filters.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: CUSTOMER PROFILE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Customer Profile":
    st.markdown('<div class="page-title">Customer Profile & Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Understand which customer segments are most likely to churn</div>', unsafe_allow_html=True)

    # Top KPIs
    c1, c2, c3, c4 = st.columns(4)
    avg_tenure = df_binary["Tenure_Years"].mean()
    avg_connections = df_binary["#_of_Connection"].mean()
    top_churn_band = (
        df_binary.groupby("Band")["Is_Churned"].mean().sort_values(ascending=False).index[0]
        if len(df_binary) > 0 else "N/A"
    )
    top_churn_band_rate = (
        df_binary.groupby("Band")["Is_Churned"].mean().sort_values(ascending=False).iloc[0]
        if len(df_binary) > 0 else 0
    )

    with c1:
        st.markdown(kpi_card("Avg Tenure", f"{avg_tenure:.1f} yrs", COLORS["accent"]), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Avg Connections", f"{avg_connections:.1f}", COLORS["accent2"]), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Highest Churn Band", top_churn_band, COLORS["Churned"]), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Its Churn Rate", f"{top_churn_band_rate:.1%}", COLORS["Churned"]), unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Churn Rate by Band</div>', unsafe_allow_html=True)
        band_churn = df_binary.groupby("Band").agg(
            churn_rate=("Is_Churned", "mean"),
            count=("Is_Churned", "count"),
        ).reset_index().sort_values("churn_rate", ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=band_churn["Band"], x=band_churn["churn_rate"],
            orientation="h",
            marker=dict(
                color=band_churn["churn_rate"],
                colorscale=[[0, COLORS["Won"]], [1, COLORS["Churned"]]],
                line=dict(width=0),
            ),
            text=band_churn["churn_rate"].apply(lambda x: f"{x:.1%}"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Churn Rate: %{x:.1%}<br>Count: %{customdata:,}<extra></extra>",
            customdata=band_churn["count"],
        ))
        fig.update_xaxes(title="Churn Rate", tickformat=".0%")
        fig.update_yaxes(title="")
        fig.update_layout(showlegend=False)
        apply_chart_theme(fig, 400)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown('<div class="section-header">Churn Rate by Tenure Group</div>', unsafe_allow_html=True)
        tenure_churn = df_binary.groupby("Tenure_Group").agg(
            churn_rate=("Is_Churned", "mean"),
            count=("Is_Churned", "count"),
        ).reset_index().sort_values("churn_rate", ascending=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=tenure_churn["Tenure_Group"], x=tenure_churn["churn_rate"],
            orientation="h",
            marker=dict(
                color=tenure_churn["churn_rate"],
                colorscale=[[0, COLORS["Won"]], [1, COLORS["Churned"]]],
                line=dict(width=0),
            ),
            text=tenure_churn["churn_rate"].apply(lambda x: f"{x:.1%}"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Churn Rate: %{x:.1%}<br>Count: %{customdata:,}<extra></extra>",
            customdata=tenure_churn["count"],
        ))
        fig2.update_xaxes(title="Churn Rate", tickformat=".0%")
        fig2.update_yaxes(title="")
        fig2.update_layout(showlegend=False)
        apply_chart_theme(fig2, 400)
        st.plotly_chart(fig2, width="stretch")

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Outcome by Payment Method</div>', unsafe_allow_html=True)
        pay_data = df.groupby(["Payment_Method", "Prospect_Outcome"]).size().reset_index(name="Count")
        fig3 = px.bar(
            pay_data, x="Payment_Method", y="Count", color="Prospect_Outcome",
            color_discrete_map=OUTCOME_COLOR_MAP,
            category_orders={"Prospect_Outcome": OUTCOME_ORDER},
            barmode="group",
        )
        fig3.update_xaxes(title="")
        fig3.update_yaxes(title="Count")
        fig3.update_traces(marker_line_width=0)
        apply_chart_theme(fig3, 380)
        st.plotly_chart(fig3, width="stretch")

    with col4:
        st.markdown('<div class="section-header">Churn Rate by Connection Group</div>', unsafe_allow_html=True)
        conn_churn = df_binary.groupby("Connection_Group").agg(
            churn_rate=("Is_Churned", "mean"),
            count=("Is_Churned", "count"),
        ).reset_index().sort_values("Connection_Group")
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=conn_churn["Connection_Group"], y=conn_churn["churn_rate"],
            marker=dict(
                color=conn_churn["churn_rate"],
                colorscale=[[0, COLORS["Won"]], [1, COLORS["Churned"]]],
                line=dict(width=0),
            ),
            text=conn_churn["churn_rate"].apply(lambda x: f"{x:.1%}"),
            textposition="outside",
            hovertemplate="Connection Group: %{x}<br>Churn Rate: %{y:.1%}<br>Count: %{customdata:,}<extra></extra>",
            customdata=conn_churn["count"],
        ))
        fig4.update_xaxes(title="Connection Group")
        fig4.update_yaxes(title="Churn Rate", tickformat=".0%")
        fig4.update_layout(showlegend=False)
        apply_chart_theme(fig4, 380)
        st.plotly_chart(fig4, width="stretch")

    # Row 3: Membership status & Audit status
    col5, col6 = st.columns(2)

    with col5:
        st.markdown('<div class="section-header">Churn by Membership Status</div>', unsafe_allow_html=True)
        mem_churn = df_binary.groupby("Proforma_Membership_Status").agg(
            churn_rate=("Is_Churned", "mean"),
            count=("Is_Churned", "count"),
        ).reset_index().sort_values("churn_rate", ascending=True)
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            y=mem_churn["Proforma_Membership_Status"], x=mem_churn["churn_rate"],
            orientation="h",
            marker=dict(
                color=mem_churn["churn_rate"],
                colorscale=[[0, COLORS["Won"]], [1, COLORS["Churned"]]],
                line=dict(width=0),
            ),
            text=mem_churn["churn_rate"].apply(lambda x: f"{x:.1%}"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Churn Rate: %{x:.1%}<br>N: %{customdata:,}<extra></extra>",
            customdata=mem_churn["count"],
        ))
        fig5.update_xaxes(title="Churn Rate", tickformat=".0%")
        fig5.update_yaxes(title="")
        fig5.update_layout(showlegend=False)
        apply_chart_theme(fig5, 380)
        st.plotly_chart(fig5, width="stretch")

    with col6:
        st.markdown('<div class="section-header">Tenure Distribution by Outcome</div>', unsafe_allow_html=True)
        fig6 = px.histogram(
            df_binary, x="Tenure_Years", color="Prospect_Outcome",
            color_discrete_map=OUTCOME_COLOR_MAP,
            barmode="overlay", nbins=30, opacity=0.7,
            category_orders={"Prospect_Outcome": ["Won", "Churned"]},
        )
        fig6.update_xaxes(title="Tenure (Years)")
        fig6.update_yaxes(title="Count")
        apply_chart_theme(fig6, 380)
        st.plotly_chart(fig6, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: INTERACTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📞 Interaction Analysis":
    st.markdown('<div class="page-title">Interaction Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">How customer-care calls, emails, and renewal calls relate to churn</div>', unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    avg_em_churn = df_binary[df_binary["Prospect_Outcome"] == "Churned"]["em_email_count"].mean()
    avg_em_won = df_binary[df_binary["Prospect_Outcome"] == "Won"]["em_email_count"].mean()
    avg_cc_churn = df_binary[df_binary["Prospect_Outcome"] == "Churned"]["cc_call_count"].mean()
    avg_ren_churn = df_binary[df_binary["Prospect_Outcome"] == "Churned"]["ren_call_count"].mean()

    with c1:
        st.markdown(kpi_card("Avg Emails (Churned)", f"{avg_em_churn:.1f}", COLORS["Churned"], f"Won: {avg_em_won:.1f}"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Avg CC Calls (Churned)", f"{avg_cc_churn:.1f}", COLORS["Churned"]), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Avg Renewal Calls (Churned)", f"{avg_ren_churn:.1f}", COLORS["Churned"]), unsafe_allow_html=True)
    with c4:
        leave_rate = df_binary[df_binary["Prospect_Outcome"] == "Churned"]["em_crm_contractor_suggested_leave"].mean()
        st.markdown(kpi_card("Leave Suggestion Rate", f"{leave_rate:.1%}", "#FF6B6B", "Churned customers"), unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Avg Interaction Counts by Outcome</div>', unsafe_allow_html=True)
        interaction_cols = ["em_email_count", "cc_call_count", "ren_call_count"]
        int_means = df_binary.groupby("Prospect_Outcome")[interaction_cols].mean().T
        int_means.index = ["Emails", "CC Calls", "Renewal Calls"]
        fig = go.Figure()
        for outcome in ["Won", "Churned"]:
            fig.add_trace(go.Bar(
                name=outcome,
                x=int_means.index,
                y=int_means[outcome],
                marker_color=OUTCOME_COLOR_MAP[outcome],
                text=int_means[outcome].apply(lambda x: f"{x:.2f}"),
                textposition="outside",
            ))
        fig.update_layout(barmode="group", showlegend=True)
        fig.update_yaxes(title="Average Count")
        fig.update_xaxes(title="")
        apply_chart_theme(fig, 380)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown('<div class="section-header">Email Volume vs Churn Rate by Year</div>', unsafe_allow_html=True)
        yr_int = df_binary.groupby("Renewal_Year").agg(
            avg_emails=("em_email_count", "mean"),
            churn_rate=("Is_Churned", "mean"),
        ).reset_index()
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(
            x=yr_int["Renewal_Year"], y=yr_int["avg_emails"],
            name="Avg Emails", marker_color=COLORS["accent"], opacity=0.7,
        ), secondary_y=False)
        fig2.add_trace(go.Scatter(
            x=yr_int["Renewal_Year"], y=yr_int["churn_rate"],
            name="Churn Rate", mode="lines+markers",
            line=dict(color=COLORS["Churned"], width=3),
            marker=dict(size=9),
        ), secondary_y=True)
        fig2.update_yaxes(title_text="Avg Emails", secondary_y=False)
        fig2.update_yaxes(title_text="Churn Rate", tickformat=".0%", secondary_y=True)
        fig2.update_xaxes(dtick=1)
        apply_chart_theme(fig2, 380)
        st.plotly_chart(fig2, width="stretch")

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Email Sentiment vs Outcome</div>', unsafe_allow_html=True)
        sent_data = df[df["em_sentiment_mode"] != "No Interaction"].groupby(
            ["em_sentiment_mode", "Prospect_Outcome"]
        ).size().reset_index(name="Count")
        fig3 = px.bar(
            sent_data, x="em_sentiment_mode", y="Count", color="Prospect_Outcome",
            color_discrete_map=OUTCOME_COLOR_MAP,
            category_orders={"Prospect_Outcome": OUTCOME_ORDER},
            barmode="group",
        )
        fig3.update_xaxes(title="Email Sentiment Mode")
        fig3.update_yaxes(title="Count")
        fig3.update_traces(marker_line_width=0)
        apply_chart_theme(fig3, 380)
        st.plotly_chart(fig3, width="stretch")

    with col4:
        st.markdown('<div class="section-header">CC Call Sentiment vs Outcome</div>', unsafe_allow_html=True)
        cc_sent = df[df["cc_sentiment_mode"] != "No Interaction"].groupby(
            ["cc_sentiment_mode", "Prospect_Outcome"]
        ).size().reset_index(name="Count")
        fig4 = px.bar(
            cc_sent, x="cc_sentiment_mode", y="Count", color="Prospect_Outcome",
            color_discrete_map=OUTCOME_COLOR_MAP,
            category_orders={"Prospect_Outcome": OUTCOME_ORDER},
            barmode="group",
        )
        fig4.update_xaxes(title="CC Sentiment Mode")
        fig4.update_yaxes(title="Count")
        fig4.update_traces(marker_line_width=0)
        apply_chart_theme(fig4, 380)
        st.plotly_chart(fig4, width="stretch")

    # Row 3: Scatter plot
    st.markdown('<div class="section-header">Interaction Intensity Scatter</div>', unsafe_allow_html=True)
    # Sample for performance
    scatter_df = df_binary.sample(n=min(5000, len(df_binary)), random_state=42)
    fig5 = px.scatter(
        scatter_df,
        x="cc_call_count", y="ren_call_count",
        color="Prospect_Outcome",
        color_discrete_map=OUTCOME_COLOR_MAP,
        size="em_email_count",
        size_max=20,
        opacity=0.5,
        category_orders={"Prospect_Outcome": ["Won", "Churned"]},
        hover_data=["Total_Net_Paid", "Tenure_Years"],
    )
    fig5.update_xaxes(title="Customer Care Calls")
    fig5.update_yaxes(title="Renewal Calls")
    apply_chart_theme(fig5, 450)
    st.plotly_chart(fig5, width="stretch")

    st.markdown(f"""
    <div class="insight-box">
        💡 <strong>Key Insight:</strong> Churned customers have on average <strong>{avg_em_churn:.1f}</strong> emails
        vs <strong>{avg_em_won:.1f}</strong> for retained. The <em>Leave Suggestion</em> flag rate among churned
        customers is <strong>{leave_rate:.1%}</strong> — one of the strongest single predictors identified in hypothesis testing.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown('<div class="page-title">Model Performance Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">10 classification models evaluated — XGBoost is the recommended production candidate</div>', unsafe_allow_html=True)

    metrics_df = get_model_metrics()

    # Best model highlight
    st.markdown("""
    <div class="model-best">
        <span style="font-size:1.4rem;">🏆</span>
        <strong style="font-size:1.1rem; color:#2ECC71;"> XGBoost</strong>
        <span style="color:#8B949E;"> — Best overall F1 (0.8049) with balanced precision (0.8024) and recall (0.8074)</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Best F1 Score", "0.8049", COLORS["Won"], "XGBoost"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Best ROC AUC", "0.9808", COLORS["accent"], "LightGBM"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Best Precision", "0.9037", COLORS["accent2"], "Random Forest"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Best Recall", "0.8571", COLORS["Open"], "Logistic Regression"), unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">F1 Score Ranking</div>', unsafe_allow_html=True)
        sorted_df = metrics_df.sort_values("F1", ascending=True)
        colors_f1 = []
        for _, row in sorted_df.iterrows():
            if row["F1"] >= 0.79:
                colors_f1.append(COLORS["Won"])
            elif row["F1"] >= 0.65:
                colors_f1.append(COLORS["Open"])
            else:
                colors_f1.append(COLORS["Churned"])

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=sorted_df["Model"], x=sorted_df["F1"],
            orientation="h",
            marker=dict(color=colors_f1, line=dict(width=0)),
            text=sorted_df["F1"].apply(lambda x: f"{x:.4f}"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>F1: %{x:.4f}<extra></extra>",
        ))
        fig.update_xaxes(title="F1 Score", range=[0, 0.9])
        fig.update_yaxes(title="")
        fig.update_layout(showlegend=False)
        apply_chart_theme(fig, 450)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown('<div class="section-header">Precision vs Recall Tradeoff</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=metrics_df["Precision"], y=metrics_df["Recall"],
            mode="markers+text",
            text=metrics_df["Model"],
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(
                size=metrics_df["F1"] * 60,
                color=metrics_df["F1"],
                colorscale=[[0, COLORS["Churned"]], [0.5, COLORS["Open"]], [1, COLORS["Won"]]],
                showscale=True,
                colorbar=dict(title="F1", tickformat=".2f"),
                line=dict(width=1, color="#FAFAFA"),
            ),
            hovertemplate="<b>%{text}</b><br>Precision: %{x:.4f}<br>Recall: %{y:.4f}<extra></extra>",
        ))
        fig2.update_xaxes(title="Precision", range=[0.35, 0.95])
        fig2.update_yaxes(title="Recall", range=[0.45, 0.90])
        fig2.update_layout(showlegend=False)
        apply_chart_theme(fig2, 450)
        st.plotly_chart(fig2, width="stretch")

    # Row 2: Multi-metric comparison
    st.markdown('<div class="section-header">Multi-Metric Comparison</div>', unsafe_allow_html=True)
    metric_cols = ["Accuracy", "Precision", "Recall", "F1"]
    sorted_models = metrics_df.sort_values("F1", ascending=False)
    fig3 = go.Figure()
    palette = [COLORS["accent"], COLORS["Won"], COLORS["Churned"], COLORS["Open"]]
    for i, metric in enumerate(metric_cols):
        fig3.add_trace(go.Bar(
            name=metric,
            x=sorted_models["Model"],
            y=sorted_models[metric],
            marker_color=palette[i],
            text=sorted_models[metric].apply(lambda x: f"{x:.2f}"),
            textposition="outside",
            textfont=dict(size=9),
        ))
    fig3.update_layout(
        barmode="group",
        xaxis_tickangle=-30,
    )
    fig3.update_yaxes(title="Score", range=[0, 1.05])
    fig3.update_xaxes(title="")
    apply_chart_theme(fig3, 450)
    st.plotly_chart(fig3, width="stretch")

    # ROC AUC chart
    st.markdown('<div class="section-header">ROC AUC Comparison</div>', unsafe_allow_html=True)
    roc_df = metrics_df.dropna(subset=["ROC_AUC"]).sort_values("ROC_AUC", ascending=True)
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        y=roc_df["Model"], x=roc_df["ROC_AUC"],
        orientation="h",
        marker=dict(
            color=roc_df["ROC_AUC"],
            colorscale=[[0, "#E74C3C"], [0.5, "#F39C12"], [1, "#2ECC71"]],
            line=dict(width=0),
        ),
        text=roc_df["ROC_AUC"].apply(lambda x: f"{x:.4f}"),
        textposition="outside",
    ))
    fig4.update_xaxes(title="ROC AUC", range=[0.85, 1.0])
    fig4.update_yaxes(title="")
    fig4.update_layout(showlegend=False)
    apply_chart_theme(fig4, 380)
    st.plotly_chart(fig4, width="stretch")

    # Full metrics table
    st.markdown('<div class="section-header">Full Metrics Table</div>', unsafe_allow_html=True)
    display_df = metrics_df.copy()
    display_df = display_df.sort_values("F1", ascending=False).reset_index(drop=True)
    display_df.index = display_df.index + 1
    display_df.index.name = "Rank"
    st.dataframe(
        display_df.style.format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}",
            "Recall": "{:.4f}", "F1": "{:.4f}", "ROC_AUC": "{:.4f}",
        }).background_gradient(subset=["F1"], cmap="RdYlGn"),
        width="stretch",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: RISK DRILLDOWN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Risk Drilldown":
    st.markdown('<div class="page-title">Risk Drilldown — Engineered Features</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Explore how composite features and churn risk signals differ between retained and churned customers</div>', unsafe_allow_html=True)

    df_binary_mr = df_mr[df_mr["Prospect_Outcome"].isin(["Won", "Churned"])].copy()

    # Composite feature groups
    composite_features = {
        "Email Risk Signals": [
            "em_churn_risk_signals", "em_dissatisfaction_index",
            "em_engagement_signals", "em_accreditation_health",
            "em_crm_contractor_suggested_leave",
        ],
        "CC Call Indices": [
            "cc_dissatisfaction_index", "cc_platform_issues_index",
            "cc_pricing_index", "cc_engagement_index", "cc_sentiment_score_avg",
        ],
        "Renewal Friction": [
            "ren_complaint_index", "ren_price_sensitivity",
            "ren_competitor_threat", "ren_has_churn_reason",
            "ren_friction_score_mean",
        ],
    }

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    churn_risk_churned = df_binary_mr[df_binary_mr["Prospect_Outcome"] == "Churned"]["em_churn_risk_signals"].mean()
    churn_risk_won = df_binary_mr[df_binary_mr["Prospect_Outcome"] == "Won"]["em_churn_risk_signals"].mean()
    complaint_churned = df_binary_mr[df_binary_mr["Prospect_Outcome"] == "Churned"]["ren_complaint_index"].mean()
    competitor_churned = df_binary_mr[df_binary_mr["Prospect_Outcome"] == "Churned"]["ren_competitor_threat"].mean()

    with c1:
        st.markdown(kpi_card("Email Risk (Churned)", f"{churn_risk_churned:.3f}", COLORS["Churned"], f"Won: {churn_risk_won:.3f}"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Complaint Index (Churned)", f"{complaint_churned:.3f}", COLORS["Churned"]), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Competitor Threat (Churned)", f"{competitor_churned:.3f}", "#FF6B6B"), unsafe_allow_html=True)
    with c4:
        leave_pct = df_binary_mr[df_binary_mr["Prospect_Outcome"] == "Churned"]["em_crm_contractor_suggested_leave"].mean()
        st.markdown(kpi_card("Leave Suggestion (Churned)", f"{leave_pct:.1%}", COLORS["Churned"]), unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Tabs for each feature group
    tabs = st.tabs(list(composite_features.keys()))

    for tab, (group_name, features) in zip(tabs, composite_features.items()):
        with tab:
            # Compute means
            means = df_binary_mr.groupby("Prospect_Outcome")[features].mean().T
            means.columns.name = None
            means = means.reset_index().rename(columns={"index": "Feature"})

            # Clean feature names for display
            means["Feature_Clean"] = means["Feature"].str.replace("em_", "Email: ", regex=False)\
                .str.replace("cc_", "CC: ", regex=False)\
                .str.replace("ren_", "Renewal: ", regex=False)\
                .str.replace("_", " ", regex=False).str.title()

            col1, col2 = st.columns([2, 1])

            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="Won",
                    y=means["Feature_Clean"],
                    x=means["Won"],
                    orientation="h",
                    marker_color=COLORS["Won"],
                ))
                fig.add_trace(go.Bar(
                    name="Churned",
                    y=means["Feature_Clean"],
                    x=means["Churned"],
                    orientation="h",
                    marker_color=COLORS["Churned"],
                ))
                fig.update_layout(
                    barmode="group",
                    title=f"Average {group_name} — Won vs Churned",
                )
                fig.update_xaxes(title="Average Value")
                fig.update_yaxes(title="")
                apply_chart_theme(fig, 380)
                st.plotly_chart(fig, width="stretch")

            with col2:
                # Lift table
                means["Lift"] = means.apply(
                    lambda r: (r["Churned"] - r["Won"]) / r["Won"] if r["Won"] != 0 else float("inf"),
                    axis=1,
                )
                display = means[["Feature_Clean", "Won", "Churned", "Lift"]].copy()
                display.columns = ["Feature", "Won Avg", "Churned Avg", "Lift"]
                st.markdown(f"**{group_name} — Lift Analysis**")
                st.dataframe(
                    display.style.format({
                        "Won Avg": "{:.4f}", "Churned Avg": "{:.4f}", "Lift": "{:+.1%}",
                    }).background_gradient(subset=["Lift"], cmap="RdYlGn_r"),
                    width="stretch",
                    hide_index=True,
                )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Scatter: Two strongest features
    st.markdown('<div class="section-header">Risk Signal Scatter — Email Risk vs Renewal Complaints</div>', unsafe_allow_html=True)
    scatter_sample = df_binary_mr.sample(n=min(5000, len(df_binary_mr)), random_state=42)
    fig_scatter = px.scatter(
        scatter_sample,
        x="em_churn_risk_signals",
        y="ren_complaint_index",
        color="Prospect_Outcome",
        color_discrete_map=OUTCOME_COLOR_MAP,
        opacity=0.4,
        size="total_interaction_count",
        size_max=18,
        category_orders={"Prospect_Outcome": ["Won", "Churned"]},
        hover_data=["ren_competitor_threat", "cc_dissatisfaction_index"],
    )
    fig_scatter.update_xaxes(title="Email Churn Risk Signals")
    fig_scatter.update_yaxes(title="Renewal Complaint Index")
    apply_chart_theme(fig_scatter, 480)
    st.plotly_chart(fig_scatter, width="stretch")

    # Radar chart for top features
    st.markdown('<div class="section-header">Feature Profile — Churned vs Won (Normalized)</div>', unsafe_allow_html=True)
    radar_features = [
        "em_churn_risk_signals", "em_dissatisfaction_index",
        "cc_dissatisfaction_index", "cc_pricing_index",
        "ren_complaint_index", "ren_price_sensitivity", "ren_competitor_threat",
    ]
    radar_labels = [
        "Email Risk", "Email Dissatisfaction",
        "CC Dissatisfaction", "CC Pricing",
        "Renewal Complaints", "Price Sensitivity", "Competitor Threat",
    ]

    won_vals = df_binary_mr[df_binary_mr["Prospect_Outcome"] == "Won"][radar_features].mean()
    churned_vals = df_binary_mr[df_binary_mr["Prospect_Outcome"] == "Churned"][radar_features].mean()

    # Normalize to 0-1 range
    all_vals = pd.concat([won_vals, churned_vals])
    max_val = all_vals.max() if all_vals.max() != 0 else 1
    won_norm = (won_vals / max_val).tolist()
    churned_norm = (churned_vals / max_val).tolist()

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=won_norm + [won_norm[0]],
        theta=radar_labels + [radar_labels[0]],
        fill="toself",
        name="Won",
        line_color=COLORS["Won"],
        fillcolor="rgba(46,204,113,0.15)",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=churned_norm + [churned_norm[0]],
        theta=radar_labels + [radar_labels[0]],
        fill="toself",
        name="Churned",
        line_color=COLORS["Churned"],
        fillcolor="rgba(231,76,60,0.15)",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        ),
    )
    apply_chart_theme(fig_radar, 500)
    st.plotly_chart(fig_radar, width="stretch")

    st.markdown("""
    <div class="insight-box">
        💡 <strong>Key Insight:</strong> The radar chart clearly shows that churned customers have elevated signals
        across <strong>all risk dimensions</strong>, with the largest gaps in <em>Email Churn Risk Signals</em>,
        <em>Renewal Complaints</em>, and <em>Competitor Threat</em>. These composite features were engineered
        specifically to capture these multi-dimensional risk patterns.
    </div>
    """, unsafe_allow_html=True)
