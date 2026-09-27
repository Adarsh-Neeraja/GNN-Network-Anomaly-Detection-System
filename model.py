"""
================================================================================
 GNN NETWORK ANOMALY DETECTION — MODEL BACKEND (model.py)
================================================================================
 DEPENDENCIES (install before running):

    pip install pandas numpy scikit-learn matplotlib networkx streamlit

    # PyTorch (CPU build shown; swap the index-url for a CUDA build if you
    # have a GPU — see https://pytorch.org/get-started/locally/)
    pip install torch --index-url https://download.pytorch.org/whl/cpu

    # PyTorch Geometric (install AFTER torch is installed)
    pip install torch_geometric

 USAGE:
    python model.py
    -> loads network_data_safe.csv, builds an IP-address graph
    -> trains a 2-layer GCN to flag malicious devices (nodes)
    -> prints Accuracy / Precision / Recall / F1 / Confusion Matrix
    -> saves gcn_model.pt and threat_scores.csv to the working directory
================================================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
CSV_PATH = "network_data_safe.csv"
MODEL_OUT_PATH = "gcn_model.pt"
SCORES_OUT_PATH = "C:\\Users\\nadar\\OneDrive\\Desktop\\GNN NETWORK ML PROJECT\\threat_scores.csv"

N_FEATURES = 8
HIDDEN_CHANNELS = 64
DROPOUT = 0.5
LEARNING_RATE = 0.01
EPOCHS = 100
TEST_SIZE = 0.2
RANDOM_SEED = 42
N_SYNTHETIC_HOSTS = 60  # only used if real IP columns are unavailable

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ==============================================================================
# 1. DATA INGESTION & CLEANING
# ==============================================================================
def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Loads the CSV and performs baseline cleaning (whitespace, inf, NaN)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find '{csv_path}'. Place it in the same directory as model.py."
        )

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Replace inf/-inf (common in CIC-IDS-2017 rate columns) with NaN, then fill
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df.fillna("UNKNOWN", inplace=True)

    return df


def identify_key_columns(df):
    """
    Dynamically identifies Label, Source IP, and Destination IP columns.
    If IP columns are missing (preprocessed datasets), generates synthetic IPs to build graph topology.
    """
    import numpy as np
    
    # 1. Identify Label Column
    label_col = None
    possible_label_cols = ['Attack Type', 'Label', 'label', 'Attack', 'class', 'target']
    
    for col in df.columns:
        if col.strip() in possible_label_cols or 'attack' in col.lower() or 'label' in col.lower():
            label_col = col
            break
            
    if not label_col:
        raise ValueError("Could not locate a Label column in the dataset.")
        
    print(f"✅ Found Label column: '{label_col}'")

    # 2. Check if Source & Destination IP columns exist
    src_col, dst_col = None, None
    for col in df.columns:
        c_clean = col.strip().lower()
        if ('src' in c_clean or 'source' in c_clean) and 'ip' in c_clean:
            src_col = col
        elif ('dst' in c_clean or 'destination' in c_clean or 'dest' in c_clean) and 'ip' in c_clean:
            dst_col = col

    # 3. Fallback: If no IP columns exist, create synthetic IPs for graph construction
    if not src_col or not dst_col:
        print("⚠️ No IP columns found in preprocessed dataset.")
        print("🔧 Generating synthetic IP topology from network flows...")
        
        np.random.seed(42)
        num_rows = len(df)
        
        # Assign synthetic internal client IPs (192.168.1.X)
        df['Source IP'] = [f"192.168.1.{i}" for i in np.random.randint(2, 120, size=num_rows)]
        
        # Map Destination Ports to synthetic server IPs (10.0.0.X)
        if 'Destination Port' in df.columns:
            ports = df['Destination Port'].astype(str)
            port_to_ip = {p: f"10.0.0.{abs(hash(p)) % 80 + 1}" for p in ports.unique()}
            df['Destination IP'] = ports.map(port_to_ip)
        else:
            df['Destination IP'] = [f"10.0.0.{i}" for i in np.random.randint(1, 50, size=num_rows)]
            
        src_col = 'Source IP'
        dst_col = 'Destination IP'
        print("✅ Graph topology generated successfully!")

    return src_col, dst_col, label_col


def generate_synthetic_ips(df: pd.DataFrame, n_hosts: int = N_SYNTHETIC_HOSTS):
    """
    FALLBACK: If the preprocessed CSV stripped out real IP strings, this
    deterministically maps each flow to a pair of synthetic 'hosts' using
    whatever flow-identifying columns are still present (ports/protocol),
    so a meaningful graph topology can still be constructed.
    """
    id_like_cols = [
        c for c in df.columns
        if c.lower().strip() in (
            "source port", "src port", "sport",
            "destination port", "dst port", "dport",
            "protocol", "flow id",
        )
    ]

    if id_like_cols:
        signature = df[id_like_cols].astype(str).agg("_".join, axis=1)
    else:
        # last resort: use the row index so topology is still deterministic
        signature = pd.Series(df.index.astype(str), index=df.index)

    def _to_ip(sig: str, salt: str) -> str:
        h = abs(hash(f"{sig}_{salt}")) % (n_hosts * n_hosts)
        a, b = divmod(h, n_hosts)
        octet2 = 0 if salt == "src" else 1
        return f"10.{octet2}.{a % 256}.{b % 256}"

    src_ip = signature.apply(lambda s: _to_ip(s, "src"))
    dst_ip = signature.apply(lambda s: _to_ip(s, "dst"))
    return src_ip, dst_ip


def resolve_ip_columns(df: pd.DataFrame, src_col, dst_col):
    """Ensures the dataframe has usable Source/Destination IP columns,
    generating synthetic ones if the real columns are missing."""
    if src_col is None or dst_col is None:
        print("[WARN] Real IP columns not found — generating synthetic host IDs "
              "from flow metadata to preserve graph topology.")
        synth_src, synth_dst = generate_synthetic_ips(df)
        df["__SRC_IP__"] = synth_src
        df["__DST_IP__"] = synth_dst
        return df, "__SRC_IP__", "__DST_IP__"

    df["__SRC_IP__"] = df[src_col].astype(str)
    df["__DST_IP__"] = df[dst_col].astype(str)
    return df, "__SRC_IP__", "__DST_IP__"


# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================
def select_feature_columns(df: pd.DataFrame, exclude_cols, n_features=N_FEATURES):
    """Picks 5-10 continuous numeric columns, preferring classic flow-stat
    features (duration/bytes/packets/IAT/rate) if present in the dataset."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [c for c in numeric_cols if c not in exclude_cols and df[c].nunique() > 1]

    if not candidates:
        raise ValueError("No usable numeric feature columns found in the dataset.")

    preferred_keywords = ["duration", "packet", "byte", "flow", "iat", "length", "rate", "size"]

    def score(col):
        cl = col.lower()
        return sum(kw in cl for kw in preferred_keywords)

    candidates.sort(key=score, reverse=True)
    selected = candidates[:n_features]

    if len(selected) < n_features:
        remaining = [c for c in candidates if c not in selected]
        selected += remaining[: n_features - len(selected)]

    return selected


def build_node_graph(df: pd.DataFrame, feature_cols, src_col, dst_col, label_col):
    """
    Constructs the graph:
      - Nodes  = unique IP addresses
      - Edges  = observed src->dst connections (added both directions)
      - Node features = mean of normalized flow features across all traffic
                         touching that IP (handles div-by-zero via fillna)
      - Node label = 1 if the IP appears in ANY non-BENIGN flow, else 0
    Returns: (PyG Data, ip_list, node_labels[np.array], fitted StandardScaler)
    """
    scaler = StandardScaler()
    normalized = scaler.fit_transform(df[feature_cols].values)
    norm_df = pd.DataFrame(normalized, columns=feature_cols, index=df.index)
    norm_df[src_col] = df[src_col].values
    norm_df[dst_col] = df[dst_col].values
    norm_df["_IS_ATTACK_"] = (df[label_col].astype(str).str.upper() != "BENIGN").astype(int)

    all_ips = pd.unique(pd.concat([df[src_col], df[dst_col]]))
    ip_to_idx = {ip: i for i, ip in enumerate(all_ips)}

    # --- node features: mean of normalized features across src+dst occurrences ---
    src_feats = norm_df[[src_col] + feature_cols].rename(columns={src_col: "IP"})
    dst_feats = norm_df[[dst_col] + feature_cols].rename(columns={dst_col: "IP"})
    long_feats = pd.concat([src_feats, dst_feats], ignore_index=True)
    node_feat_means = long_feats.groupby("IP")[feature_cols].mean().reindex(all_ips).fillna(0.0)
    x = torch.tensor(node_feat_means.values, dtype=torch.float)

    # --- node labels: 1 if involved in ANY non-benign flow ---
    src_flags = norm_df[[src_col, "_IS_ATTACK_"]].rename(columns={src_col: "IP"})
    dst_flags = norm_df[[dst_col, "_IS_ATTACK_"]].rename(columns={dst_col: "IP"})
    long_flags = pd.concat([src_flags, dst_flags], ignore_index=True)
    node_attack_flag = long_flags.groupby("IP")["_IS_ATTACK_"].max().reindex(all_ips).fillna(0).astype(int)
    y = torch.tensor(node_attack_flag.values, dtype=torch.float)

    # --- edges ---
    src_idx = df[src_col].map(ip_to_idx).values
    dst_idx = df[dst_col].map(ip_to_idx).values
    edge_index = np.vstack([
        np.concatenate([src_idx, dst_idx]),
        np.concatenate([dst_idx, src_idx]),
    ])
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data, list(all_ips), node_attack_flag.values, scaler


# ==============================================================================
# 3. MODEL ARCHITECTURE
# ==============================================================================
class GCN(torch.nn.Module):
    """2-layer GCN. Outputs raw logits; apply sigmoid externally for
    probabilities (kept separate so BCEWithLogitsLoss stays numerically stable)."""

    def __init__(self, in_channels, hidden_channels=HIDDEN_CHANNELS, dropout=DROPOUT):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.squeeze(-1)


# ==============================================================================
# 4. TRAINING LOOP
# ==============================================================================
def make_train_test_masks(labels, n_nodes, test_size=TEST_SIZE, seed=RANDOM_SEED):
    idx = np.arange(n_nodes)
    try:
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=seed, stratify=labels
        )
    except ValueError:
        # falls back to a non-stratified split if a class is too small to stratify
        train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    return train_mask, test_mask


def train_model(data, train_mask, epochs=EPOCHS, lr=LEARNING_RATE):
    model = GCN(in_channels=data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    y_train = data.y[train_mask]
    n_pos = float(y_train.sum().item())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"\nClass balance in training set -> Benign: {int(n_neg)} | Attack: {int(n_pos)}")
    print(f"Applying pos_weight={pos_weight.item():.3f} to counter class imbalance.\n")

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Loss: {loss.item():.4f}")

    return model


# ==============================================================================
# 5. EVALUATION & TERMINAL REPORTING
# ==============================================================================
def evaluate_and_report(model, data, test_mask):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

    y_true = data.y[test_mask].numpy().astype(int)
    y_pred = preds[test_mask].numpy().astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    bar = "=" * 62
    print(f"\n{bar}")
    print(" FINAL MODEL EVALUATION — HELD-OUT TEST NODES")
    print(bar)
    print(f"  Accuracy   : {acc * 100:6.2f}%")
    print(f"  Precision  : {prec * 100:6.2f}%")
    print(f"  Recall     : {rec * 100:6.2f}%")
    print(f"  F1-Score   : {f1 * 100:6.2f}%")
    print(bar)
    print(" CONFUSION MATRIX  (rows = actual, cols = predicted)")
    print("                    Pred: BENIGN   Pred: ATTACK")
    print(f"  Actual BENIGN     {cm[0][0]:>10d}   {cm[0][1]:>12d}")
    print(f"  Actual ATTACK     {cm[1][0]:>10d}   {cm[1][1]:>12d}")
    print(bar + "\n")

    return probs.numpy()


# ==============================================================================
# 6. MAIN
# ==============================================================================
def main():
    print("Loading and cleaning data...")
    df = load_raw_data(CSV_PATH)

    src_col, dst_col, label_col = identify_key_columns(df)
    df, src_col, dst_col = resolve_ip_columns(df, src_col, dst_col)

    exclude = {src_col, dst_col, label_col}
    feature_cols = select_feature_columns(df, exclude)
    print(f"Selected feature columns: {feature_cols}")

    print("Building IP-address graph...")
    data, ip_list, node_labels, scaler = build_node_graph(
        df, feature_cols, src_col, dst_col, label_col
    )
    print(f"Graph built -> {data.num_nodes} nodes, {data.num_edges} edges, "
          f"{int(node_labels.sum())} malicious nodes.")

    train_mask, test_mask = make_train_test_masks(node_labels, data.num_nodes)

    print("Training GCN...")
    model = train_model(data, train_mask)

    probs = evaluate_and_report(model, data, test_mask)

    torch.save(model.state_dict(), MODEL_OUT_PATH)
    print(f"Saved trained model weights -> {MODEL_OUT_PATH}")

    scores_df = pd.DataFrame({
        "IP_Address": ip_list,
        "True_Label": node_labels,
        "Anomaly_Score": probs,
    }).sort_values("Anomaly_Score", ascending=False)
    scores_df.to_csv(SCORES_OUT_PATH, index=False)
    print(f"Saved anomaly scores -> {SCORES_OUT_PATH}\n")


if __name__ == "__main__":
    main()
