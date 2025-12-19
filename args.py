# args.py
import argparse
import os

parser = argparse.ArgumentParser(description="FedSEA: One-Shot Federated Graph Learning")

# ==============================================================================
# 1. Environment & Paths
# ==============================================================================
current_path = os.path.abspath(__file__)
dataset_path = os.path.join(os.path.dirname(current_path), 'datasets')
root_dir = os.path.join(dataset_path, 'raw_data')

if not os.path.exists(root_dir):
    os.makedirs(root_dir, exist_ok=True)

log_path = os.path.join(os.path.dirname(current_path), 'logs')
if not os.path.exists(log_path):
    os.makedirs(log_path, exist_ok=True)

env_group = parser.add_argument_group('Environment')
env_group.add_argument("--dataset", type=str, default="cora", help="Dataset name: cora, ogbn-arxiv, etc.")
env_group.add_argument("--dataset_dir", type=str, default=root_dir, help="Root path for datasets")
env_group.add_argument("--logs_dir", type=str, default=log_path, help="Path to save logs")
env_group.add_argument("--device_id", type=int, default=0, help="GPU device ID")
env_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

# ==============================================================================
# 2. Data Partitioning (Non-IID Settings)
# ==============================================================================
data_group = parser.add_argument_group('Data Partitioning')
data_group.add_argument("--task", type=str, default="node_classification")
data_group.add_argument("--num_clients", type=int, default=10, help="Number of clients")
data_group.add_argument("--dirichlet_alpha", type=float, default=0.3, help="Dirichlet Alpha (smaller = more Non-IID)")
data_group.add_argument("--least_samples", type=int, default=10, help="Min samples per client")
data_group.add_argument("--dataset_split_metric", type=str, default="transductive", choices=["transductive"])
data_group.add_argument("--train_val_test_split", type=float, nargs='+', default=[0.6, 0.2, 0.2])
data_group.add_argument("--dirichlet_try_cnt", type=int, default=100)

# ==============================================================================
# 3. Federated Learning Settings
# ==============================================================================
fl_group = parser.add_argument_group('Federated Learning')
fl_group.add_argument("--fed_algorithm", type=str, default="FedSEA")
fl_group.add_argument("--num_rounds", type=int, default=100, help="Total communication rounds")
fl_group.add_argument("--cl_sample_rate", type=float, default=1.0, help="Client sampling rate")
fl_group.add_argument("--T_L", type=int, default=1, help="Local epochs (Skipped in One-Shot FedSEA)")

# ==============================================================================
# 4. Global Model (Backbone GNN)
# ==============================================================================
model_group = parser.add_argument_group('Backbone Model')
model_group.add_argument("--model", type=str, default="GCN", help="GNN Backbone: GCN, GAT, SAGE")
model_group.add_argument("--hidden_dim", type=int, default=256)
model_group.add_argument("--num_layers", type=int, default=2)
model_group.add_argument("--dropout", type=float, default=0.5)
model_group.add_argument("--server_lr", type=float, default=0.005, help="Learning rate for Global GNN")
model_group.add_argument("--weight_decay", type=float, default=0.0005)
model_group.add_argument('--patience', type=int, default=30, help='Patience rounds for early stopping based on Val Acc')

# ==============================================================================
# 5. FedSEA Generator (Diffusion Model)
# ==============================================================================
sea_group = parser.add_argument_group('FedSEA Generator')
sea_group.add_argument("--gen_mode", type=str, default="server", help="Generator location: server or federated")
sea_group.add_argument("--gen_train_steps", type=int, default=100, help="Generator training steps per round")
sea_group.add_argument("--gen_lr", type=float, default=0.0001, help="Learning rate for Generator")
sea_group.add_argument("--gen_hidden", type=int, default=256, help="Hidden dim for diffusion network")
sea_group.add_argument("--noise_dim", type=int, default=64, help="Dimension of input noise z")
sea_group.add_argument("--diff_steps", type=int, default=50, help="Number of denoising steps (T)")
sea_group.add_argument("--gen_knn", type=int, default=5, help="k for KNN structure generation")

# ==============================================================================
# 6. FedSEA Loss Weights & Hyperparams
# ==============================================================================
loss_group = parser.add_argument_group('FedSEA Losses')
loss_group.add_argument("--T_G", type=int, default=50, help="Global GNN training steps per round")
loss_group.add_argument("--tau", type=float, default=1.0, help="Temperature for SWA weights (Entropy-based)")
loss_group.add_argument("--gen_num_samples", type=int, default=1, help="Number of graph samples per client (Ensemble size)")

# The Two Main Loss Components
loss_group.add_argument("--w_proto", type=float, default=20.0, help="Weight for Prototype Consistency Loss")
loss_group.add_argument("--w_ot", type=float, default=0.01, help="Weight for FGW-OT Regularization Loss")

# ==============================================================================
# 7. Regularization (EWC/Retention)
# ==============================================================================
reg_group = parser.add_argument_group('Regularization')
reg_group.add_argument("--use_retention", type=int, default=0, help="Enable EWC-like retention")
reg_group.add_argument("--lambda_r", type=float, default=0.0, help="Weight for retention loss")

# Parse
args = parser.parse_args()