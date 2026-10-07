# GNN-Network-Anomaly-Detection-System
An AI-powered cybersecurity MVP that detects malicious network activity by modeling network traffic as a mathematical graph. Instead of analyzing isolated packets, this tool uses a Graph Convolutional Network (GCN) to identify attackers based on abnormal "neighborhood" communication patterns.

##Project Structure

*   **`shrink_data.py`**: The data prep script. Safely samples the massive 2GB CIC-IDS-2017 dataset down to a lightweight 12,500-row file (`network_data_safe.csv`) for local testing without crashing your machine.
   
*   **`model.py`**: The AI backend. It loads the CSV, generates synthetic IP topology, builds the graph, trains the GCN, and exports the final device risk grades to `threat_scores.csv`.
  
*   **`app.py`**: The frontend command center. A live Streamlit dashboard that visualizes the network topology, allows dynamic "smoke detector" threat threshold adjustments, and ranks high-risk devices in an actionable table.

##How to Run

**1. Prepare the Data** *(Run this once to generate the lightweight dataset)*:
```bash
python shrink_data.py
```

**2. Train the Model & Generate Scores** *(Run this to let the AI grade the network)*:
```bash
python model.py
```

**3. Launch the Dashboard** *(Run this to view the interactive UI in your browser)*:
```bash
python -m streamlit run app.py
```
