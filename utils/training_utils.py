# utils/training_utils.py
import copy
import torch
import torch.nn.functional as F
import numpy as np


def average_state_dicts(state_dicts):
    """
    Aggregates a list of state_dicts by averaging their parameters.
    """
    if len(state_dicts) == 0:
        return None

    avg_state = copy.deepcopy(state_dicts[0])
    keys = list(avg_state.keys())

    for k in keys:
        if avg_state[k].dtype.is_floating_point:
            avg_state[k] = avg_state[k].float() * 0.0
        else:
            continue

    num_states = len(state_dicts)
    for sd in state_dicts:
        for k in keys:
            if avg_state[k].dtype.is_floating_point:
                avg_state[k] += sd[k].float() / num_states

    return avg_state


def calculate_generator_loss(generator, stats, device,
                             w_proto=1.0, w_ot=0.0, ot_target_protos=None,
                             batch_size=4096):
    """
    Computes the loss for training the Graph Diffusion Generator.
    Uses Mini-batching to prevent OOM on large graphs.
    """

    # 1. Determine safe batch size
    # We MUST cap the generation size.
    full_num_nodes = int(stats.get("num_nodes", 20))

    # Take the smaller of the two: actual nodes OR batch limit
    curr_batch_size = min(full_num_nodes, batch_size)

    # Update stats temporarily to reflect batch size (for condition building)
    # This ensures the condition tensor matches the noise tensor size
    batch_stats = stats.copy()
    batch_stats["num_nodes"] = curr_batch_size

    # 2. Randomly sample labels for this batch
    y_gen = torch.randint(0, generator.num_classes, (curr_batch_size,), device=device)

    # 3. Build conditions (Prototypes + Entropy)
    # The generator will internally use batch_stats["num_nodes"] to shape the condition
    cond = generator.build_cond_tensor(batch_stats, device, assigned_labels=y_gen)

    # 4. Sample noise
    z = torch.randn((curr_batch_size, generator.noise_dim), device=device)

    # 5. Forward pass (Denoising) -> Generate Features X_hat
    X_hat = generator.forward(z, cond)

    # Initialize Loss
    loss = torch.tensor(0.0, device=device)
    log_dict = {}

    # -----------------------------------------------------------
    # 6. Prototype Consistency Loss (L_proto)
    # -----------------------------------------------------------
    if w_proto > 0:
        input_protos = stats["prototypes"].to(device)
        target_P = input_protos[y_gen]

        loss_proto = F.mse_loss(X_hat, target_P)
        loss = loss + w_proto * loss_proto
        log_dict['proto'] = loss_proto.item()

    # -----------------------------------------------------------
    # 7. FGW-OT Regularization Loss (L_ot)
    # -----------------------------------------------------------
    if w_ot > 0 and ot_target_protos is not None:
        if not isinstance(ot_target_protos, torch.Tensor):
            target_ot = torch.tensor(ot_target_protos, device=device, dtype=torch.float32)
        else:
            target_ot = ot_target_protos.to(device)

        # Calculate actual class centers in the current mini-batch
        batch_centers = torch.zeros_like(target_ot)

        for c in range(generator.num_classes):
            mask = (y_gen == c)
            if mask.sum() > 0:
                batch_centers[c] = X_hat[mask].mean(dim=0)
            else:
                # If a class is missing in this random batch, use target to ignore loss
                batch_centers[c] = target_ot[c].detach()

        loss_ot_val = F.mse_loss(batch_centers, target_ot)
        loss = loss + w_ot * loss_ot_val
        log_dict['ot'] = loss_ot_val.item()

    return loss, None, log_dict