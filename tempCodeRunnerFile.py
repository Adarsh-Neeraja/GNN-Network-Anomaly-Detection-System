"""
================================================================================
 GNN NETWORK ANOMALY DETECTION — STREAMLIT COMMAND CENTER (app.py)
================================================================================
 Run with:   streamlit run app.py
 Requires model.py to have been executed first (produces threat_scores.csv).
================================================================================
"""

import os

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

from model import (
    load_raw_data,
    identify_key_columns,
    resolve_ip_columns,
    CSV_PATH,
)

# --- NEW PATH FIX ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCORES_PATH = os.path.join(BASE_DIR, "threat_scores.csv")
MAX_GRAPH_EDGES = 400  # cap for a readable / performant topology render
# --------------------

st.set_page_config(page_title="GNN Network Anomaly Detection", layout="wide")


# ------------------------------------------------------------------------------
# DATA LOADING (cached)
# ------------------------------------------------------------------------------
@st.cache_data
def load_scores():
    if not os.path.exists(SCORES_PATH):
        return None
    return pd.read_csv(SCORES_PATH)


@st.cache_data
def load_topology_frame():
    if not os.path.exists(CSV_PATH):
        return None
    df = load_raw_data(CSV_PATH)
    src_col, dst_col, _ = identify_key_columns(df)
    df, src_col, dst_col = resolve_ip_columns(df, src_col, dst_col)
    return df[[src_col, dst_col]].rename(columns={src_col: "src", dst_col: "dst"})


# ------------------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        "This dashboard visualizes network traffic as a graph and uses a "
        "**Graph Convolutional Network (GCN)** to flag devices (IP addresses) "
        "likely involved in malicious activity, trained on flow features "
        "derived from CIC-IDS-2017."
    )
    st.markdown("---")
    threshold = st.slider(
        "Threat Threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.01
    )
    st.caption("Devices scoring above this threshold are flagged as high-risk.")


# ------------------------------------------------------------------------------
# TITLE
# ------------------------------------------------------------------------------
st.title("🛡️ GNN Network Anomaly Detection")
st.caption("Graph-based anomaly detection command center — powered by PyTorch Geometric")

scores_df = load_scores()

if scores_df is None:
    st.error(
        f"'{SCORES_PATH}' was not found. Run `python model.py` first to train the "
        "model and generate anomaly scores."
    )
    st.stop()

scores_df["Anomaly_Score"] = pd.to_numeric(scores_df["Anomaly_Score"], errors="coerce").fillna(0.0)


# ------------------------------------------------------------------------------
# KPI ROW
# ------------------------------------------------------------------------------
total_devices = len(scores_df)
high_risk = int((scores_df["Anomaly_Score"] >= threshold).sum())
avg_score = scores_df["Anomaly_Score"].mean() if total_devices > 0 else 0.0
high_risk_pct = (high_risk / total_devices * 100) if total_devices > 0 else 0.0

k1, k2, k3 = st.columns(3)
k1.metric("Total Devices Analyzed", f"{total_devices:,}")
k2.metric(
    "High-Risk Devices",
    f"{high_risk:,}",
    delta=f"{high_risk_pct:.1f}% of network",
    delta_color="inverse",
)
k3.metric("Average Network Threat Score", f"{avg_score:.3f}")

st.markdown("---")


# ------------------------------------------------------------------------------
# MIDDLE ROW — TOPOLOGY + DISTRIBUTION
# ------------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Network Topology")
    topo_df = load_topology_frame()

    if topo_df is None or topo_df.empty:
        st.info(f"'{CSV_PATH}' not found — cannot render topology.")
    else:
        display_edges = topo_df.head(MAX_GRAPH_EDGES)
        G = nx.from_pandas_edgelist(display_edges, source="src", target="dst")

        risk_lookup = dict(zip(scores_df["IP_Address"], scores_df["Anomaly_Score"]))
        node_colors = [
            "#d62728" if risk_lookup.get(n, 0.0) >= threshold else "#2ca02c"
            for n in G.nodes()
        ]

        if G.number_of_nodes() == 0:
            st.info("No edges available to render.")
        else:
            fig, ax = plt.subplots(figsize=(6, 5))
            pos = nx.spring_layout(G, seed=42, k=0.4)
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=120, ax=ax, alpha=0.9)
            nx.draw_networkx_edges(G, pos, alpha=0.25, ax=ax)
            ax.set_axis_off()
            st.pyplot(fig)
            st.caption(
                f"Showing first {len(display_edges):,} flows "
                f"({G.number_of_nodes():,} devices). 🟢 Normal · 🔴 Suspicious"
            )

with col2:
    st.subheader("Anomaly Score Distribution")
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.hist(scores_df["Anomaly_Score"], bins=30, color="#1f77b4", edgecolor="white")
    ax2.axvline(
        threshold, color="#d62728", linestyle="--", linewidth=2,
        label=f"Threshold ({threshold:.2f})",
    )
    ax2.set_xlabel("Anomaly Score")
    ax2.set_ylabel("Number of Devices")
    ax2.legend()
    st.pyplot(fig2)

st.markdown("---")


# ------------------------------------------------------------------------------
# BOTTOM ROW — ACTIONABLE THREAT TABLE
# ------------------------------------------------------------------------------
st.subheader("🚨 Actionable Threat Table")

flagged = scores_df[scores_df["Anomaly_Score"] >= threshold].sort_values(
    "Anomaly_Score", ascending=False
)

if flagged.empty:
    st.success("No devices currently exceed the threat threshold.")
else:
    styled = flagged.style.format({"Anomaly_Score": "{:.3f}"}).background_gradient(
        subset=["Anomaly_Score"], cmap="Reds"
    )
    st.dataframe(styled, use_container_width=True)
    st.caption(f"{len(flagged):,} device(s) flagged above threshold {threshold:.2f}.")
