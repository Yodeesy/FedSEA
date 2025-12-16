# utils/GW_utils.py
"""
Solvers from the Python Optimal Transport (POT) library.
https://pythonot.github.io/

Adapted for the integration of (Fused) Gromov-Wasserstein in differentiable layers.
Note: This file appears to be a legacy component from GHOST/TFGW. 
FedSEA currently uses 'utils/ot_align.py' for prototype alignment.
"""

import numpy as np
import torch as th
from ot.optim import cg
from ot.utils import list_to_array
from ot.backend import get_backend
from ot.gromov import init_matrix, gwloss, gwggrad

def parallel_gromov_wasserstein2(C1, C2, p, q, loss_fun='square_loss', log=False, armijo=False, G0=None, **kwargs):
    """
    Computes the Gromov-Wasserstein discrepancy between (C1, p) and (C2, q).
    
    Optimization problem:
        GW = min_T sum_{i,j,k,l} L(C1_{i,k}, C2_{j,l}) * T_{i,j} * T_{k,l}
    
    Args:
        C1: Metric cost matrix in the source space (ns, ns)
        C2: Metric cost matrix in the target space (nt, nt)
        p: Distribution in the source space (ns,)
        q: Distribution in the target space (nt,)
        loss_fun: 'square_loss' or 'kl_loss'
        log: If True, returns log dictionary
        armijo: If True, uses Armijo line search
        G0: Initial transport plan (optional)
        
    Returns:
        gw_dist: The GW distance value
        (Optional) gradients: If backprop is needed
    """
    p, q = list_to_array(p, q)

    p0, q0, C10, C20 = p, q, C1, C2
    nx = get_backend(p0, q0, C10, C20)

    p = nx.to_numpy(p)
    q = nx.to_numpy(q)
    C1 = nx.to_numpy(C10)
    C2 = nx.to_numpy(C20)

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)
    
    if G0 is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = nx.to_numpy(G0)

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    # Solve using Conditional Gradient (CG)
    T, log_gw = cg(p, q, 0, 1, f, df, G0, log=True, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)

    # Compute Gradients for Backward Pass (if needed for custom Autograd)
    gp = nx.from_numpy(log_gw['u'] - log_gw['u'].mean())
    gq = nx.from_numpy(log_gw['v'] - log_gw['v'].mean())

    if loss_fun == 'square_loss':
        gC1 = nx.from_numpy(2 * C1 * (p[:, None] * p[None, :]) - 2 * T.dot(C2).dot(T.T))
        gC2 = nx.from_numpy(2 * C2 * (q[:, None] * q[None, :]) - 2 * T.T.dot(C1).dot(T))
    else:
        gC1 = None
        gC2 = None

    return nx.from_numpy(gwloss(constC, hC1, hC2, T), type_as=C10), gp, gq, gC1, gC2


def parallel_fused_gromov_wasserstein2_learnablealpha(C1, C2, F1, F2, M, p, q, 
                                                      loss_fun='square_loss', alpha=0.5, 
                                                      compute_gradients=True, learn_alpha=False, 
                                                      armijo=False, log=False, G0=None, **kwargs):
    """
    Computes the Fused Gromov-Wasserstein (FGW) distance with support for learnable alpha.
    
    Objective:
        min_T (1 - alpha) * <T, M>_F + alpha * sum L(C1, C2) * T * T
        
    Args:
        C1, C2: Structure matrices
        F1, F2: Feature matrices (used to compute gradients)
        M: Feature cost matrix (ns, nt)
        p, q: Marginal distributions
        alpha: Trade-off parameter (0 < alpha < 1)
        
    Returns:
        fgw_dist: The FGW distance
        gradients: Tuple of gradients w.r.t inputs
    """
    p, q = list_to_array(p, q)

    p0, q0, C10, C20, F10, F20, M0, alpha0 = p, q, C1, C2, F1, F2, M, alpha
    nx = get_backend(p0, q0, C10, C20, F10, F20, M0, alpha0)

    p = nx.to_numpy(p0)
    q = nx.to_numpy(q0)
    C1 = nx.to_numpy(C10)
    C2 = nx.to_numpy(C20)
    F1 = nx.to_numpy(F10)
    F2 = nx.to_numpy(F20)
    M = nx.to_numpy(M0)
    alpha = nx.to_numpy(alpha0)
    
    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    if G0 is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = nx.to_numpy(G0)

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    # Solve FGW
    T, log_fgw = cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, log=True, **kwargs)

    fgw_dist = nx.from_numpy(log_fgw['loss'][-1], type_as=C10)
    
    if not compute_gradients:
        return fgw_dist
    
    # Compute Gradients
    gp = nx.from_numpy(log_fgw['u'] - log_fgw['u'].mean())
    gq = nx.from_numpy(log_fgw['v'] - log_fgw['v'].mean())
    
    # Gradient w.r.t Features (F1, F2) derived from Transport Plan T
    gF1 = nx.from_numpy(2 * F1 * p[:, None] - 2 * T.dot(F2))
    gF2 = nx.from_numpy(2 * F2 * q[:, None] - 2 * (T.T).dot(F1))
    
    gC1, gC2, galpha = None, None, None

    if loss_fun == 'square_loss':
        gC1 = nx.from_numpy(2 * C1 * (p[:, None] * p[None, :]) - 2 * T.dot(C2).dot(T.T))
        gC2 = nx.from_numpy(2 * C2 * (q[:, None] * q[None, :]) - 2 * T.T.dot(C1).dot(T))
        
        if learn_alpha:
            gwloss_val = gwloss(constC, hC1, hC2, T)
            galpha = nx.from_numpy(gwloss_val - (M * T).sum(), type_as=C10)

    # Move to correct device
    device = fgw_dist.device
    gp = gp.to(device)
    gq = gq.to(device)
    if gC1 is not None: gC1 = gC1.to(device)
    if gC2 is not None: gC2 = gC2.to(device)
    gF1 = gF1.to(device)
    gF2 = gF2.to(device)
    
    return fgw_dist, gp, gq, alpha0 * gC1, alpha0 * gC2, (1. - alpha0) * gF1, (1. - alpha0) * gF2, galpha

def probability_simplex_projection(x):
    """
    Projects input vector x onto the probability simplex.
    """
    descending_idx = th.argsort(x, descending=True)
    u = x[descending_idx]
    rho = 0.
    lambda_ = 1.
    for i in range(u.shape[0]):
        value = u[i] + (1 - u[:(i+1)].sum()) / (i+1)
        if value > 0:
            rho += 1
            lambda_ -= u[i]
        else:
            break
    return th.max(x + lambda_/rho, th.zeros_like(x))