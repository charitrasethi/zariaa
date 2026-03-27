import streamlit as st

st.set_page_config(
    page_title="Zaria Fashion Analytics",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #f8f9fa; }
.metric-card { background:#fff; border:1px solid #e9ecef; border-radius:10px;
               padding:16px; text-align:center; }
.metric-val  { font-size:28px; font-weight:700; color:#1D9E75; }
.metric-lbl  { font-size:13px; color:#6c757d; margin-top:4px; }
.section-hdr { font-size:18px; font-weight:600; color:#2C2C2A;
               border-left:4px solid #1D9E75; padding-left:10px; margin:20px 0 10px; }
.insight-box { background:#e8f5f0; border-left:4px solid #1D9E75;
               padding:12px 16px; border-radius:6px; margin:8px 0; font-size:13px; }
.warning-box { background:#fff8e1; border-left:4px solid #FFC107;
               padding:12px 16px; border-radius:6px; margin:8px 0; font-size:13px; }
</style>
""", unsafe_allow_html=True)

pages = {
    "🏠 Executive Overview":    "page_overview",
    "📊 Descriptive Analysis":  "page_descriptive",
    "🔍 Diagnostic Analysis":   "page_diagnostic",
    "👥 Customer Clustering":   "page_clustering",
    "🔗 Association Rules":     "page_arm",
    "🎯 Classification":        "page_classification",
    "💰 Regression & CLV":      "page_regression",
    "🆕 New Customer Predictor":"page_predictor",
}

st.sidebar.image("https://via.placeholder.com/200x60/1D9E75/FFFFFF?text=ZARIA+FASHION",
                 use_column_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
selected = st.sidebar.radio("", list(pages.keys()), label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.caption("1,200 survey respondents · 25 columns · Pan-India")
st.sidebar.markdown("---")
st.sidebar.markdown("**Algorithms**")
st.sidebar.caption("K-Means · Apriori · Random Forest · XGBoost · Ridge Regression")

import importlib
mod = importlib.import_module(pages[selected])
mod.render()
