import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from data_loader import load_data, BINARY_COLS

@st.cache_data
def get_data():
    return load_data()

@st.cache_data
def run_arm(min_support=0.07, min_confidence=0.50, region_filter="All"):
    df = get_data()
    if region_filter != "All":
        df = df[df["region"] == region_filter]

    # Build basket: binary product cols + fabric (one-hot) + color (one-hot)
    basket = df[BINARY_COLS].copy()

    # Add fabric as binary flags
    for fab in df["fabric_preference"].unique():
        basket[f"fabric_{fab}"] = (df["fabric_preference"] == fab).astype(int)

    # Add color as binary flags
    for col in df["color_preference"].unique():
        basket[f"color_{col}"] = (df["color_preference"] == col).astype(int)

    basket = basket.astype(bool)
    basket.columns = [c.replace("owns_","").replace("_"," ").title()
                      .replace("Fabric ","🧵 ").replace("Color ","🎨 ") for c in basket.columns]

    freq_items = apriori(basket, min_support=min_support, use_colnames=True)
    if freq_items.empty:
        return pd.DataFrame(), basket

    rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules["lift"] >= 1.1].sort_values("lift", ascending=False)
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules = rules[["antecedents_str","consequents_str","support","confidence","lift"]].round(3)
    return rules, basket

def render():
    st.title("🔗 Association Rule Mining")
    st.caption("What do customers own/prefer together? Apriori algorithm — drives bundle strategy.")

    df = get_data()

    # Controls
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        min_sup = st.slider("Min Support", 0.03, 0.30, 0.07, 0.01)
    with c2:
        min_conf = st.slider("Min Confidence", 0.30, 0.90, 0.50, 0.05)
    with c3:
        min_lift = st.slider("Min Lift", 1.0, 3.0, 1.1, 0.1)
    with c4:
        region_f = st.selectbox("Region filter", ["All"] + sorted(df["region"].unique().tolist()))

    with st.spinner("Mining association rules…"):
        rules, basket = run_arm(min_sup, min_conf, region_f)

    if rules.empty:
        st.warning("No rules found with these settings. Try lowering min support or confidence.")
        return

    rules_filtered = rules[rules["lift"] >= min_lift].copy()
    st.success(f"Found **{len(rules_filtered)}** rules (support ≥ {min_sup}, confidence ≥ {min_conf}, lift ≥ {min_lift})")

    # ── Top Rules Table ──────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Top Association Rules</div>", unsafe_allow_html=True)
    display_rules = rules_filtered.head(30).copy()
    display_rules.columns = ["If customer has →","→ They also have","Support","Confidence","Lift"]
    display_rules["Support"]    = display_rules["Support"].map("{:.3f}".format)
    display_rules["Confidence"] = display_rules["Confidence"].map("{:.3f}".format)
    display_rules["Lift"]       = display_rules["Lift"].map("{:.2f}".format)

    st.dataframe(display_rules, use_container_width=True, height=400)
    st.download_button("Download Rules CSV",
                       rules_filtered.to_csv(index=False).encode(),
                       "zaria_arm_rules.csv","text/csv")

    # ── Support vs Confidence scatter ────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Support vs Confidence vs Lift</div>", unsafe_allow_html=True)
    plot_rules = rules_filtered.copy()
    plot_rules["rule_label"] = plot_rules["antecedents_str"] + " → " + plot_rules["consequents_str"]
    fig_sc = px.scatter(plot_rules, x="support", y="confidence",
                        size="lift", color="lift",
                        color_continuous_scale="Greens",
                        hover_data=["rule_label","lift"],
                        title="Support vs Confidence (bubble size = Lift)",
                        labels={"support":"Support","confidence":"Confidence","lift":"Lift"})
    fig_sc.update_layout(height=420, margin=dict(t=50,b=10,l=10,r=10))
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Lift Distribution ────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Lift Distribution</div>", unsafe_allow_html=True)
    c_a, c_b = st.columns(2)
    with c_a:
        fig_lift = px.histogram(rules_filtered, x="lift", nbins=20,
                                color_discrete_sequence=["#1D9E75"],
                                title="Distribution of Lift Values",
                                labels={"lift":"Lift","count":"Number of Rules"})
        fig_lift.update_layout(height=320, margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig_lift, use_container_width=True)

    with c_b:
        fig_conf = px.histogram(rules_filtered, x="confidence", nbins=20,
                                color_discrete_sequence=["#378ADD"],
                                title="Distribution of Confidence Values",
                                labels={"confidence":"Confidence","count":"Number of Rules"})
        fig_conf.update_layout(height=320, margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig_conf, use_container_width=True)

    # ── Network Graph ────────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Association Network Graph</div>", unsafe_allow_html=True)
    st.caption("Node size = frequency · Edge thickness = confidence · Edge colour = lift")

    top_rules_net = rules_filtered.head(25)
    G = nx.DiGraph()
    for _, row in top_rules_net.iterrows():
        G.add_edge(row["antecedents_str"], row["consequents_str"],
                   weight=row["confidence"], lift=row["lift"])

    pos = nx.spring_layout(G, seed=42, k=2)
    edge_x, edge_y, edge_colors = [], [], []
    for u, v, data in G.edges(data=True):
        x0,y0 = pos[u]; x1,y1 = pos[v]
        edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
        edge_colors.append(data.get("lift",1))

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_size = [max(15, G.degree(n)*8) for n in G.nodes()]
    node_text = list(G.nodes())

    fig_net = go.Figure()
    fig_net.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                 line=dict(width=1.5, color="#aaaaaa"), hoverinfo="none"))
    fig_net.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                 marker=dict(size=node_size, color="#1D9E75",
                                             line=dict(color="#fff",width=1.5)),
                                 text=node_text, textposition="top center",
                                 textfont=dict(size=9), hoverinfo="text"))
    fig_net.update_layout(showlegend=False, height=480,
                          xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                          yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                          margin=dict(t=10,b=10,l=10,r=10),
                          title="Product Association Network (top 25 rules)")
    st.plotly_chart(fig_net, use_container_width=True)

    # ── Bundle Recommendations ───────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>🛍️ Bundle Recommendations from Rules</div>", unsafe_allow_html=True)
    high_lift = rules_filtered[rules_filtered["lift"] >= rules_filtered["lift"].quantile(0.75)]

    if not high_lift.empty:
        st.markdown("Rules with **lift in top 25%** → direct bundle opportunities:")
        for _, r in high_lift.head(6).iterrows():
            lift_val = float(r["lift"])
            conf_val = float(r["confidence"])
            st.markdown(
                f"📦 **Bundle:** _{r['antecedents_str']}_ + _{r['consequents_str']}_ "
                f"| Confidence: `{conf_val:.0%}` | Lift: `{lift_val:.2f}x` "
                f"→ *{lift_val:.0f}x more likely to be bought together than by chance*"
            )
    else:
        st.info("Adjust thresholds to see bundle recommendations.")

    # ── Regional Rule Comparison ─────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Regional Rule Comparison</div>", unsafe_allow_html=True)
    regions = df["region"].unique().tolist()
    reg_rule_counts = {}
    for reg in regions:
        r2, _ = run_arm(min_sup, min_conf, reg)
        reg_rule_counts[reg] = len(r2[r2["lift"] >= min_lift]) if not r2.empty else 0

    rc_df = pd.DataFrame(list(reg_rule_counts.items()), columns=["Region","Rule Count"])
    rc_df = rc_df.sort_values("Rule Count", ascending=False)
    fig_rc = px.bar(rc_df, x="Region", y="Rule Count",
                    color="Rule Count", color_continuous_scale=["#9FE1CB","#085041"],
                    text="Rule Count", title="Number of Valid Rules per Region")
    fig_rc.update_traces(textposition="outside")
    fig_rc.update_layout(height=320, coloraxis_showscale=False,
                         margin=dict(t=50,b=10,l=10,r=10), xaxis_title="")
    st.plotly_chart(fig_rc, use_container_width=True)
    st.markdown("<div class='insight-box'>📌 Regions with more rules = richer product co-purchase patterns = better opportunities for targeted bundles in that geography.</div>", unsafe_allow_html=True)
