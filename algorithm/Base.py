# algorithm/base.py
import torch
import torch.nn.functional as F
import random
from sklearn.metrics import f1_score


class BaseServer:
    """
    Base class for Federated Learning Server.
    Handles client sampling, communication, aggregation, and global evaluation.
    """

    def __init__(self, args, clients, model, data, logger):
        self.logger = logger
        self.sampled_clients = None
        self.clients = clients
        self.model = model  # Global model (GNN)
        self.cl_sample_rate = args.cl_sample_rate
        self.num_rounds = args.num_rounds
        self.T_L = args.T_L  # Local training epochs
        self.data = data
        # Determine the device of the model
        self.device = next(model.parameters()).device
        self.num_total_samples = sum([client.num_samples for client in self.clients])

    def run(self):
        """
        Main training loop for Federated Learning.
        """
        for round_idx in range(self.num_rounds):
            print(f"Round {round_idx + 1}:")
            self.logger.write_round(round_idx + 1)

            # 1. Client Sampling
            self.sample()

            # 2. Communication (Download global model to clients)
            # For FedSEA, clients are frozen, so this might be skipped or no-op.
            self.communicate()

            # 3. Local Training
            print("cid : ", end='')
            for cid in self.sampled_clients:
                client = self.clients[cid]

                # Check if client has a trainable model (FedSEA clients might not)
                if client.model is not None:
                    print('---------------------------')
                    print(f'{cid} training proxy model...')
                    for epoch in range(self.T_L):
                        client.train()
                else:
                    # Silent pass for data-free methods like FedSEA
                    pass

            # 4. Aggregation
            # FedSEA will override this method with its own logic (Generator training & Fusion).
            # Base implementation performs standard FedAvg.
            self.aggregate()

            # 5. Global Evaluation
            self.global_evaluate()

    def communicate(self):
        """
        Broadcast global model parameters to sampled clients.
        """
        for cid in self.sampled_clients:
            client = self.clients[cid]
            # Only update if client has a local model structure
            if client.model is not None:
                for client_param, server_param in zip(client.model.parameters(), self.model.parameters()):
                    client_param.data.copy_(server_param.data)

    def sample(self):
        """
        Randomly sample a subset of clients for the current round.
        """
        num_sample_clients = int(len(self.clients) * self.cl_sample_rate)
        # Ensure at least 1 client is sampled if rate > 0
        num_sample_clients = max(1, num_sample_clients)
        self.sampled_clients = random.sample(range(len(self.clients)), num_sample_clients)

    def aggregate(self):
        """
        Standard FedAvg Aggregation Strategy.
        Aggregates parameters from sampled clients weighted by their dataset size.
        Note: FedSEA overrides this method.
        """
        total_samples = sum([self.clients[cid].num_samples for cid in self.sampled_clients])

        for i, cid in enumerate(self.sampled_clients):
            client = self.clients[cid]

            # Skip clients without models
            if client.model is None:
                continue

            w = client.num_samples / total_samples

            for client_param, global_param in zip(client.model.parameters(), self.model.parameters()):
                if i == 0:
                    global_param.data.copy_(w * client_param)
                else:
                    global_param.data += w * client_param

    def global_evaluate(self):
        """
        Evaluate the global model on the server-side test data.
        Returns Accuracy, Loss, and Macro-F1 Score.
        Includes automatic CPU offloading for large graphs to prevent OOM.
        """
        self.model.eval()

        # Determine evaluation device (CPU for large graphs, GPU otherwise)
        eval_device = self.device
        if hasattr(self.data, 'x') and self.data.x.shape[0] > 100000:
            # Force CPU evaluation for large-scale graphs (e.g., Arxiv)
            eval_device = 'cpu'
            self.model.to('cpu')

        # Temporarily move data to the evaluation device
        if hasattr(self.data, 'to'):
            eval_data = self.data.to(eval_device)
        else:
            eval_data = self.data

        with torch.no_grad():
            # Forward pass
            out = self.model(eval_data)

            # Handle different GNN output formats
            if isinstance(out, (tuple, list)):
                logits = out[-1]
            else:
                logits = out

            # Initialize metrics
            loss_val = 0.0
            acc = 0.0
            macro_f1 = 0.0

            # Determine mask (Test Mask or Full Data)
            mask = getattr(eval_data, 'test_mask', None)

            if mask is not None and mask.sum().item() > 0:
                # Transductive Setting
                loss = F.cross_entropy(logits[mask], eval_data.y[mask])
                pred = logits[mask].max(dim=1)[1]
                acc = pred.eq(eval_data.y[mask]).sum().item() / mask.sum().item()

                # Calculate F1 (CPU required for sklearn)
                y_true = eval_data.y[mask].detach().cpu().numpy()
                y_pred = pred.detach().cpu().numpy()
            else:
                # Inductive / Full Graph Setting
                loss = F.cross_entropy(logits, eval_data.y)
                pred = logits.max(dim=1)[1]
                acc = pred.eq(eval_data.y).sum().item() / eval_data.y.shape[0]

                y_true = eval_data.y.detach().cpu().numpy()
                y_pred = pred.detach().cpu().numpy()

            macro_f1 = f1_score(y_true, y_pred, average='macro')

            # Format loss for logging
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss

            print(f"test_loss : {loss_val:.4f}")
            self.logger.write_test_loss(loss_val)

            print(f"test_acc  : {acc:.4f}")
            self.logger.write_test_acc(acc)

            print(f"macro_f1  : {macro_f1:.4f}")
            if hasattr(self.logger, 'write_test_f1'):
                self.logger.write_test_f1(macro_f1)

        # Restore model to original device if it was moved to CPU
        if eval_device == 'cpu':
            self.model.to(self.device)
            # eval_data will be automatically garbage collected

        return acc, loss_val, macro_f1


class BaseClient:
    """
    Base class for a Federated Learning Client.
    """

    def __init__(self, args, model, data):
        self.model = model
        self.data = data
        self.loss_fn = F.cross_entropy
        self.args = args
        self.num_samples = len(data.x) if data.x is not None else 0

        # Initialize optimizer only if model exists (FedSEA clients might be model-less)
        if self.model is not None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
        else:
            self.optimizer = None

    def train(self):
        """
        Perform local training for one epoch.
        """
        if self.model is None:
            return 0.0

        # Ensure data and model are on the same device
        device = next(self.model.parameters()).device
        train_data = self.data.to(device)

        self.model.train()
        self.optimizer.zero_grad()

        _, out = self.model(train_data)

        if hasattr(train_data, 'train_mask') and train_data.train_mask is not None:
            loss = self.loss_fn(out[train_data.train_mask], train_data.y[train_data.train_mask])
        else:
            loss = self.loss_fn(out, train_data.y)

        loss.backward()
        self.optimizer.step()

        return loss.item()