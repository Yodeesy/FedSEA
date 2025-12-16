# utils/training_utils.py
import copy
import torch
import torch.nn.functional as F
import numpy as np


def average_state_dicts(state_dicts):
    """
    Aggregates a list of state_dicts by averaging their parameters.
    Used for standard FedAvg logic.
    """
    if len(state_dicts) == 0:
        return None

    avg_state = copy.deepcopy(state_dicts[0])
    keys = list(avg_state.keys())

    # Initialize with 0
    for k in keys:
        if avg_state[k].dtype.is_floating_point:
            avg_state[k] = avg_state[k].float() * 0.0
        else:
            # Skip non-float buffers (e.g. num_batches_tracked) if any
            continue

    # Accumulate
    num_states = len(state_dicts)
    for sd in state_dicts:
        for k in keys:
            if avg_state[k].dtype.is_floating_point:
                avg_state[k] += sd[k].float() / num_states
            else:
                # For integer buffers (like LongTensor), usually we take the first one or majority
                # Here we just keep the first one's value logic for simplicity
                pass

    return avg_state


def calculate_generator_loss(generator, stats, device,
                             w_proto=1.0, w_ot=0.0, ot_target_protos=None):
    """
    Computes the loss for training the Graph Diffusion Generator.

    Components:
    1. Proto Loss: Instance-level consistency (MSE between features and assigned prototypes).
    2. OT Loss: Global-level geometric consistency (MSE between batch centers and OT targets).

    Note: Topology/Degree losses are removed as they are non-differentiable
          and implicitly handled by the diffusion backbone + KNN decoder.
    """

    # 1. Prepare Data
    # We only need to generate features (X), no need to build the graph (A) here.
    # This significantly speeds up training compared to calling generate().
    num_nodes = int(stats.get("num_nodes", 20))

    # Randomly sample labels for this batch
    y_gen = torch.randint(0, generator.num_classes, (num_nodes,), device=device)

    # Build conditions (Prototypes + Entropy)
    cond = generator.build_cond_tensor(stats, device, assigned_labels=y_gen)

    # Sample noise
    z = torch.randn((num_nodes, generator.noise_dim), device=device)

    # Forward pass (Denoising) -> Generate Features X_hat
    X_hat = generator.forward(z, cond)

    # Initialize Loss
    loss = torch.tensor(0.0, device=device)
    log_dict = {}

    # -----------------------------------------------------------
    # 2. Prototype Consistency Loss (L_proto)
    # -----------------------------------------------------------
    if w_proto > 0:
        # Retrieve input SWA prototypes (Condition)
        input_protos = stats["prototypes"].to(device)

        # Target: Each generated node should differ little from its class prototype
        target_P = input_protos[y_gen]

        loss_proto = F.mse_loss(X_hat, target_P)
        loss = loss + w_proto * loss_proto
        log_dict['proto'] = loss_proto.item()

    # -----------------------------------------------------------
    # 3. FGW-OT Regularization Loss (L_ot)
    # -----------------------------------------------------------
    if w_ot > 0 and ot_target_protos is not None:
        # Ensure target is a tensor
        if not isinstance(ot_target_protos, torch.Tensor):
            target_ot = torch.tensor(ot_target_protos, device=device, dtype=torch.float32)
        else:
            target_ot = ot_target_protos.to(device)

        # Calculate actual class centers in the current generated batch
        batch_centers = torch.zeros_like(target_ot)

        for c in range(generator.num_classes):
            mask = (y_gen == c)
            if mask.sum() > 0:
                batch_centers[c] = X_hat[mask].mean(dim=0)
            else:
                # If a class is missing in this random batch, fill with target
                # so MSE loss for this class becomes 0 (no gradient impact)
                batch_centers[c] = target_ot[c].detach()

        loss_ot_val = F.mse_loss(batch_centers, target_ot)
        loss = loss + w_ot * loss_ot_val
        log_dict['ot'] = loss_ot_val.item()

    return loss, None, log_dict