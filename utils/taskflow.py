# utils/taskflow.py
import torch
import numpy as np
import time
from backbone import get_model
from datasets.dataset_loader import load_dataset
from datasets.dataset_ds_loader import partition_by_node  # The optimized vectorized slicer
from algorithm import get_server, get_client
from datasets.partition import dirichlet_partitioner
from utils.logger import DefaultLogger


class TaskFlow:
    """
    Orchestrates the entire Federated Learning pipeline:
    1. Load Global Dataset
    2. Partition Data (Non-IID)
    3. Initialize Clients (with Local Subgraphs)
    4. Initialize Server (with Global Graph)
    5. Start Training Loop
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")

        # Initialize Logger
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        exp_name = f"{timestamp}-{args.task}-{args.dataset}-{args.fed_algorithm}"
        logger = DefaultLogger(exp_name, args.logs_dir)

        # ---------------------------------------------------------
        # 1. Load Global Dataset
        # ---------------------------------------------------------
        if args.task == "node_classification":
            self.dataset = load_dataset(args.train_val_test_split, args.dataset_dir, args.dataset)

            # Auto-detect dimensions
            input_dim = self.dataset.num_node_features
            num_classes = len(np.unique(self.dataset.y.numpy()))
            args.num_classes = num_classes  # Inject into args for model init

            # Scale hyperparameters if needed (Legacy logic preserved)
            if hasattr(args, 'lambda_r'): args.lambda_r *= getattr(args, 'r_scale', 1.0)
            if hasattr(args, 'lambda_n'): args.lambda_n *= getattr(args, 'n_scale', 1.0)
            if hasattr(args, 'lambda_f'): args.lambda_f *= getattr(args, 'f_scale', 1.0)

            # ---------------------------------------------------------
            # 2. Initialize Global Model (Backbone)
            # ---------------------------------------------------------
            server_model = get_model(
                args.model,
                input_dim,
                args.hidden_dim,
                num_classes,
                args.num_layers,
                args.dropout
            )
            server_model.to(self.device)

            # ---------------------------------------------------------
            # 3. Partition Data (Transductive Only)
            # ---------------------------------------------------------
            if args.dataset_split_metric == "transductive":
                # A. Get Node Indices per Client (using Dirichlet)
                print("[TaskFlow] Partitioning graph...")
                clients_node_indices = dirichlet_partitioner(
                    self.dataset,
                    args.num_clients,
                    alpha=args.dirichlet_alpha,
                    least_samples=getattr(args, 'least_samples', 10)
                )

                # B. Extract Subgraphs (Vectorized)
                # This uses the new fast partition_by_node from dataset_ds_loader.py
                clients_data = partition_by_node(self.dataset, clients_node_indices)

                # ---------------------------------------------------------
                # 4. Initialize Clients
                # ---------------------------------------------------------
                clients = []
                for cid in range(args.num_clients):
                    client_data = clients_data[cid]

                    # For FedSEA, clients don't train local GNNs, so model can be None
                    if args.fed_algorithm == "FedSEA":
                        client_model = None
                    else:
                        # For FedAvg/GHOST, clients need a model copy
                        client_model = get_model(
                            args.model, input_dim, args.hidden_dim, num_classes,
                            args.num_layers, args.dropout
                        )
                        client_model.to(self.device)

                    # Instantiate Client
                    client = get_client(args.fed_algorithm, args, client_model, client_data)
                    client.client_id = cid  # Assign ID for logging
                    clients.append(client)

                # ---------------------------------------------------------
                # 5. Initialize Server
                # ---------------------------------------------------------
                # Server gets the FULL dataset for global evaluation (Standard Transductive Setting)
                server_data = self.dataset

                self.server = get_server(
                    args.fed_algorithm, args, clients, server_model, server_data, logger
                )

            else:
                raise NotImplementedError("FedSEA currently supports Transductive setting only.")

    def run(self):
        print("[TaskFlow] Starting Server Run Loop...")
        self.server.run()