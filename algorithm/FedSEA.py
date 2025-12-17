# algorithm/FedSEA.py
import numpy as np
import torch
import copy
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from algorithm.Base import BaseServer, BaseClient
from utils.stats import graph_structural_entropy, compute_class_prototypes
from utils.graph_ops import fuse_graphs, softmax_weights
from backbone.GraphDiffusionGenerator import GraphDiffusionGenerator
from utils.training_utils import calculate_generator_loss
from utils.utils import compute_criterion  # Optional parameter retention


class FedSEAServer(BaseServer):
    """
    FedSEA Server:
    1. Collects stats (Prototypes, Entropy) from clients.
    2. Trains a server-side Diffusion Generator.
    3. Generates synthetic subgraphs.
    4. Fuses them into a global mega-graph (Union).
    5. Trains the global GNN on this union graph.
    """

    def __init__(self, args, clients, model, data, logger):
        super(FedSEAServer, self).__init__(args, clients, model, data, logger)
        self.args = args
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.feature_dim = self.data.x.shape[-1]
        self.num_classes = args.num_classes
        self.noise_dim = args.noise_dim

        # Optimizer for the global GNN
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=getattr(args, "server_lr", 0.01),
            weight_decay=getattr(args, "weight_decay", 0.0)
        )

        # Hyperparameters
        self.lambda_r = getattr(args, "lambda_r", 0.0)  # Optional param retention
        self.tau = getattr(args, "tau", 1.0)  # Temp for entropy weights
        self.T_G = getattr(args, "T_G", 1)  # Global epochs per round

        # Initialize Server-side Generator
        # gen_mode can be 'server' (default) or 'federated'
        self.gen_mode = getattr(args, "gen_mode", "server")

        self.fedsea_generator = GraphDiffusionGenerator(
            self.device,
            self.noise_dim,
            self.feature_dim,
            self.num_classes,
            # No degree bins anymore
            num_timesteps=getattr(args, "diff_steps", 100),
            hidden_dim=getattr(args, "gen_hidden", 256),
            args=args
        ).to(self.device)

        self.gen_optimizer = torch.optim.Adam(
            self.fedsea_generator.parameters(),
            lr=getattr(args, "gen_lr", 1e-3)
        )

        self.round_idx = 0
        self.best_acc = 0.0
        self.best_online_acc = 0.0

    def aggregate(self):
        print('---------------------------')
        print(f'FedSEA aggregate (Round {self.round_idx + 1}): Collect stats & Train Generator')

        device = self.device
        sampled = list(self.sampled_clients)

        # ======================================================
        # 1. Collect Statistics from Clients
        # Only keeping: Entropy (S_k) and Prototypes
        # ======================================================
        stats_list = []
        entropies = []
        client_protos = []

        for cid in sampled:
            stats = self.clients[cid].get_stats()
            # Entropy for adaptive weighting
            S_k = float(stats.get("S_k", 0.0))
            entropies.append(S_k)
            stats_list.append(stats)

            # Prototypes for alignment
            prot = stats.get("prototypes", None)
            if prot is not None and isinstance(prot, torch.Tensor):
                prot = prot.detach().cpu().numpy()
            client_protos.append(prot)

        if len(stats_list) == 0:
            return

        # ======================================================
        # 2. Calculate Adaptive Weights (Alpha) based on Entropy
        # ======================================================
        alphas = softmax_weights(entropies, tau=self.tau)
        alphas = [float(a) for a in alphas]
        print(f"FedSEA: Entropies: {[f'{e:.4f}' for e in entropies]}")
        print(f"FedSEA: Alphas:    {[f'{a:.4f}' for a in alphas]}")

        # ======================================================
        # 3. Calculate Global Prototypes
        # Step A: Weighted Average (SWA) - Baseline Target
        # ======================================================
        global_protos_swa = np.zeros((self.num_classes, self.feature_dim), dtype=np.float32)
        valid_protos_count = 0
        for w, p in zip(alphas, client_protos):
            if p is not None:
                global_protos_swa += w * np.array(p)
                valid_protos_count += 1

        # ======================================================
        # Step B: FGW-OT Alignment (Optional, controlled by w_ot)
        # ======================================================
        global_protos_ot = None
        w_ot = getattr(self.args, "w_ot", 0.0)

        if w_ot > 0 and valid_protos_count > 1:
            try:
                from utils.ot_align import fgw_prototype_alignment
                global_protos_ot = fgw_prototype_alignment(
                    client_protos, alphas,
                    self.num_classes, self.feature_dim,
                    max_iter=5, beta=0.5
                )
                print(f"  [FedSEA] FGW-OT Alignment finished. (w_ot={w_ot})")
            except Exception as e:
                print(f"  [Warning] FGW calculation failed: {e}")
                global_protos_ot = None

        # ======================================================
        # 4. Inject Conditions into Stats
        # ======================================================
        if valid_protos_count > 0:
            # We use SWA prototypes as the conditioning input for generation
            # (OT prototypes are only used for Loss regularization)
            aligned_proto_tensor = torch.tensor(global_protos_swa, dtype=torch.float32)
            for stats in stats_list:
                stats["prototypes"] = aligned_proto_tensor

        # Prepare OT target tensor for Loss
        ot_target_tensor = None
        if global_protos_ot is not None:
            ot_target_tensor = torch.tensor(global_protos_ot, dtype=torch.float32, device=device)

        # ======================================================
        # 5. Train Generator on Server
        # ======================================================
        gen_train_steps = getattr(self.args, "gen_train_steps", 100)

        if self.gen_mode == "server" or gen_train_steps > 0:
            self.fedsea_generator.train()

            for step in range(gen_train_steps):
                # Randomly sample a client's stats context
                idx = np.random.randint(0, len(stats_list))
                stats = stats_list[idx]

                self.gen_optimizer.zero_grad()

                # Calculate Loss (Proto Consistency + OT Regularization)
                loss_gen, _, _ = calculate_generator_loss(
                    self.fedsea_generator,
                    stats,
                    device,
                    w_proto=getattr(self.args, "w_proto", 1.0),
                    w_ot=w_ot,
                    ot_target_protos=ot_target_tensor
                )

                loss_gen.backward()
                self.gen_optimizer.step()

        # ======================================================
        # 6. Generate Pseudo-Graphs
        # ======================================================
        pseudo_graphs = []
        for stats in stats_list:
            # Ensure num_nodes is set (fallback to 20 if missing)
            if "num_nodes" not in stats:
                stats["num_nodes"] = int(stats.get("n_nodes", 20))

            with torch.no_grad():
                # generator.generate() now includes KNN internally
                Gk = self.fedsea_generator.generate(stats)

            Gk = Gk.to(device)
            pseudo_graphs.append(Gk)

        # ======================================================
        # 7. Fuse Graphs (Concatenation)
        # ======================================================
        # Ensure using the concatenation version from utils/graph_ops
        G_global = fuse_graphs(pseudo_graphs, alphas, device=device)

        # Memory Cleanup
        del pseudo_graphs
        torch.cuda.empty_cache()

        # Detach to stop gradients flowing back to generator
        G_global.x = G_global.x.detach()
        if hasattr(G_global, 'edge_index'):
            G_global.edge_index = G_global.edge_index.detach()

        print(f"FedSEA: Fused global union graph -> {G_global}")

        # ======================================================
        # 8. Pseudo Labeling (via SWA Prototypes)
        # ======================================================
        P_labeling = torch.tensor(global_protos_swa, device=device).detach()
        Xg = G_global.x
        # Cosine similarity assignment
        sims = torch.matmul(F.normalize(Xg, p=2, dim=1), F.normalize(P_labeling, p=2, dim=1).t())
        y_pseudo = sims.argmax(dim=1)

        # ======================================================
        # 9. Optional: Parameter Retention (EWC-like)
        # ======================================================
        use_retention = getattr(self, "use_retention", False)
        criterion = None
        if use_retention:
            criterion, _ = compute_criterion(self.model, G_global.x, G_global.edge_index)

        # ======================================================
        # 10. Train Global Model (GNN)
        # [Crucial] OOM Protection for Large Graphs (e.g. Arxiv)
        # ======================================================
        train_device = device
        # Threshold: 100k nodes (Arxiv is ~170k, Cora ~20k)
        if G_global.x.size(0) > 100000:
            print(f"  [FedSEA] Graph size ({G_global.x.size(0)}) large. Switching training to CPU to avoid OOM.")
            train_device = 'cpu'
            self.model = self.model.to('cpu')
            G_global = G_global.to('cpu')
            y_pseudo = y_pseudo.to('cpu')

            # Move retention criterion to CPU if needed
            if use_retention and criterion:
                for k in criterion:
                    if isinstance(criterion[k], torch.Tensor):
                        criterion[k] = criterion[k].to('cpu')

        print(f"FedSEA: Training global model for {self.T_G} steps on {train_device}")

        self.model.train()
        for t in range(self.T_G):
            self.optimizer.zero_grad()

            _, out = self.model(G_global)
            # Handle PyG output (tuple or tensor)
            logits = out if not isinstance(out, (list, tuple)) else out[-1]

            loss = F.cross_entropy(logits, y_pseudo)

            # Add Retention Loss
            if use_retention and criterion is not None:
                loss_ret = 0.0
                for name, param in self.model.named_parameters():
                    val = criterion.get(name, None)
                    if val is not None:
                        loss_ret += (abs(val) * (param ** 2)).sum()
                loss += self.lambda_r * loss_ret

            loss.backward()
            self.optimizer.step()

        # [Restore] Move model back to GPU for evaluation
        if train_device == 'cpu':
            self.model = self.model.to(device)
            torch.cuda.empty_cache()

        # ======================================================
        # 11. Evaluation & Saving
        # ======================================================
        # Online Eval (Current Model)
        online_acc, _, online_f1 = self.global_evaluate()

        if online_acc > self.best_online_acc:
            self.best_online_acc = online_acc
            self._save_checkpoint(f"{self.args.dataset}_best_gnn_online_seed{self.args.seed}.pth", self.model)
            print(f"✅ [Save] New Best Online Model (Acc: {online_acc:.4f})")

        print(
            f"Round {self.round_idx + 1} Online Acc: {online_acc:.4f} (Best: {self.best_online_acc:.4f}) | F1: {online_f1:.4f}")

        # EMA Eval (Moving Average Model)
        if not hasattr(self, 'ema_state_dict'):
            self.ema_state_dict = copy.deepcopy(self.model.state_dict())
        else:
            beta = 0.95
            current_state = self.model.state_dict()
            for key in current_state:
                self.ema_state_dict[key] = beta * self.ema_state_dict[key] + (1 - beta) * current_state[key]

        # Backup current, load EMA, eval, restore
        backup_state = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.ema_state_dict)

        test_acc, _, test_f1 = self.global_evaluate()

        if test_acc > self.best_acc:
            self.best_acc = test_acc
            self._save_checkpoint(f"{self.args.dataset}_best_gnn_ema_seed{self.args.seed}.pth", self.model)
            self._save_checkpoint(f"{self.args.dataset}_best_gen_seed{self.args.seed}.pth", self.fedsea_generator)
            print(f"🏆 Round {self.round_idx + 1}: New Best EMA Acc: {self.best_acc:.4f} (Saved)")

        print(f"Round {self.round_idx + 1} EMA Acc: {test_acc:.4f} (Best: {self.best_acc:.4f}) | F1: {test_f1:.4f}")

        self.model.load_state_dict(backup_state)
        self.round_idx += 1
        print("FedSEA aggregation finished.")

    def _save_checkpoint(self, filename, model_obj):
        import os
        root_dir = getattr(self.args, 'log_dir', '.')
        save_dir = os.path.join(root_dir, "checkpoints")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model_obj.state_dict(), os.path.join(save_dir, filename))


class FedSEAClient(BaseClient):
    """
    FedSEA Client:
    - Calculates local statistics (Entropy, Prototypes).
    - No local generator training needed in the default 'server' mode.
    """

    def __init__(self, args, model, data):
        super(FedSEAClient, self).__init__(args, model, data)
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.feature_dim = self.data.x.shape[-1]
        self.num_classes = args.num_classes

        # If gen_mode is 'federated', we can init a local generator here (Optional)
        # For current 'server' mode, this is skipped.

    def get_stats(self):
        """
        Compute and return local statistics.
        Removed: degree_histogram, phi.
        Kept: S_k, prototypes.
        """
        # Determine training mask
        if hasattr(self.data, 'train_mask') and self.data.train_mask is not None:
            train_mask = self.data.train_mask.cpu().numpy()
        else:
            # Fallback for full-graph usage (use with caution)
            train_mask = np.ones(self.data.x.shape[0], dtype=bool)

        # 1. Prepare Data
        num_nodes = int(self.data.x.shape[0])
        # Convert to dense adj for entropy calculation
        A = to_dense_adj(self.data.edge_index, max_num_nodes=num_nodes).squeeze(0)
        if isinstance(A, torch.Tensor):
            A_np = A.cpu().numpy()
        else:
            A_np = np.array(A)

        feats = self.data.x.cpu().numpy()
        labels = self.data.y.cpu().numpy()

        # 2. Compute Structural Entropy (S_k)
        # Based on full topology (A_np) to capture structural complexity
        S_k = graph_structural_entropy(A_np)

        # 3. Compute Prototypes
        # MUST use only training data to prevent leakage
        train_feats = feats[train_mask]
        train_labels = labels[train_mask]
        prototypes = compute_class_prototypes(train_feats, train_labels, self.num_classes)

        return {
            "S_k": float(S_k),
            "prototypes": prototypes,
            "num_nodes": num_nodes
        }