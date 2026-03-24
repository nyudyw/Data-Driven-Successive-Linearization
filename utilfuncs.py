import numpy as np
import pickle
import os

def get_v_ctrl_from_sol(sol, controlled_nodes: np.ndarray) -> np.ndarray:

    v_all = np.asarray(sol["v"], dtype=float).ravel()
    m = controlled_nodes.size

    if v_all.size == m:
        idx = controlled_nodes
    elif v_all.size == m + 1:
        idx = controlled_nodes + 1
    else:
        raise ValueError(
            f"[Linear] Unexpected v length {v_all.size} for m={m} controlled nodes."
        )

    return v_all[idx].reshape(1, -1)

def extract_v_ctrl(sol, controlled_nodes, cvc):

    v_all = np.asarray(sol['v'], dtype=float).ravel()
    if cvc.index_mode == "reduced":
        idx_full = np.asarray(controlled_nodes, dtype=int) + 1
    else:
        idx_full = np.asarray(controlled_nodes, dtype=int)
    return v_all[idx_full].reshape(1, -1)

def has_nan(matrix):
    return np.isnan(matrix).any()


def func_grad_u(v_curr, s_ref, J_v_u, u_curr, c_action: float = 0.0, R_u: np.ndarray = None):
    """
    ∇_u J(u) = J^T (v - s_ref) + 2 c_action R_u u   (R_u = I if None)
    """
    v_curr = np.asarray(v_curr, dtype=float).reshape(1, -1)
    s_ref  = np.asarray(s_ref,  dtype=float).reshape(1, -1)
    u_vec  = np.asarray(u_curr, dtype=float).reshape(-1, 1)
    J      = np.asarray(J_v_u,  dtype=float)

    e = (v_curr - s_ref).reshape(-1, 1)
    grad_from_state = J.T @ e

    if c_action != 0.0:
        if R_u is None:
            grad_from_action = (2.0 * float(c_action)) * u_vec
        else:
            R_u = np.asarray(R_u, dtype=float)
            if R_u.shape != (u_vec.shape[0], u_vec.shape[0]):
                raise ValueError(f"R_u must be {(u_vec.shape[0], u_vec.shape[0])}, got {R_u.shape}")
            grad_from_action = (2.0 * float(c_action)) * (R_u @ u_vec)
    else:
        grad_from_action = 0.0

    grad_u = grad_from_state + grad_from_action
    return grad_u.ravel()

def func_cost1(s, s_ref, u, c_action: float = 0.0, R_u: np.ndarray = None):
    """
    J(u) = 0.5 * ||s - s_ref||^2 + c_action * (u^T R_u u),  R_u=I if None
    """
    s     = np.asarray(s, dtype=float).reshape(1, -1)
    s_ref = np.asarray(s_ref, dtype=float).reshape(1, -1)
    u     = np.asarray(u, dtype=float).reshape(-1, 1)

    e = (s - s_ref).ravel()
    voltage_cost = 0.5 * float(e @ e)

    if R_u is None:
        action_cost = float(c_action) * float(u.ravel() @ u.ravel())
    else:
        R_u = np.asarray(R_u, dtype=float)
        if R_u.shape != (u.shape[0], u.shape[0]):
            raise ValueError(f"R_u must be {(u.shape[0], u.shape[0])}, got {R_u.shape}")
        action_cost = float(c_action) * float((u.T @ R_u @ u).item())

    return voltage_cost + action_cost



def load_kvec_from_pckl_simple(path, controlled_nodes):
    if not os.path.exists(path):
        raise FileNotFoundError(f"K file not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, (list, tuple)):
        arr = np.asarray(obj[0], dtype=float)
    else:
        arr = np.asarray(obj, dtype=float)
    kvec = np.squeeze(arr).ravel()  
    m = controlled_nodes.size
    if kvec.size != m:
        raise ValueError(
            f"K vector length {kvec.size} != #controlled {m}. "
            f"Check file {path} or controlled_nodes."
        )
    if not np.isfinite(kvec).all():
        raise ValueError("K vector contains NaN/inf.")
    return kvec.astype(float, copy=False)

def get_top_bottom_voltage_nodes(trajectory, controlled_nodes, k, top_n=5):
    traj_arr = np.asarray(trajectory)

    if traj_arr.ndim == 3:
        traj_arr = np.squeeze(traj_arr, axis=1)
    elif traj_arr.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected trajectory shape {traj_arr.shape}")

    T, m = traj_arr.shape
    if not (0 <= k < T):
        raise IndexError(f"k={k} is out of range for trajectory length {T}")

    v_k = traj_arr[k].astype(float).ravel()
    nodes = np.asarray(controlled_nodes, dtype=int).ravel()
    if nodes.size != m:
        raise ValueError(f"controlled_nodes length {nodes.size} != m={m}")

    idx_sorted = np.argsort(v_k)
    bottom_idx = idx_sorted[:top_n]
    top_idx = idx_sorted[-top_n:][::-1]

    result = {
        "k": k,
        "top_idx": top_idx,
        "top_nodes": nodes[top_idx],
        "top_values": v_k[top_idx],
        "bottom_idx": bottom_idx,
        "bottom_nodes": nodes[bottom_idx],
        "bottom_values": v_k[bottom_idx],
    }
    return result


def print_top_bottom_voltage_nodes(trajectory, controlled_nodes, k, top_n=5):
    res = get_top_bottom_voltage_nodes(trajectory, controlled_nodes, k, top_n=top_n)
    print(f"\nTime step k = {res['k']}")
    print("Top nodes (largest voltages):")
    for nid, val in zip(res["top_nodes"], res["top_values"]):
        print(f"  node {int(nid):>4d}: {val:.6f}")
    print("Bottom nodes (smallest voltages):")
    for nid, val in zip(res["bottom_nodes"], res["bottom_values"]):
        print(f"  node {int(nid):>4d}: {val:.6f}")


def tail_mean(costs_opt, tail_k: int) -> float:
    if costs_opt is None or len(costs_opt) == 0:
        return float("nan")
    arr = np.asarray(costs_opt, float).ravel()
    k = min(tail_k, arr.size)
    return float(np.mean(arr[-k:]))

def head_mean(costs_opt, head_k: int) -> float:
    if costs_opt is None or len(costs_opt) == 0:
        return float("nan")
    arr = np.asarray(costs_opt, float).ravel()
    k = min(head_k, arr.size)
    return float(np.mean(arr[:k]))


def tail_mean_last_k(x, k=5):
    x = np.asarray(x, float).ravel()
    if x.size == 0:
        return float("nan")
    k = min(k, x.size)
    return float(np.mean(x[-k:]))

def head_mean_first_k(x, k=5):
    x = np.asarray(x, float).ravel()
    if x.size == 0:
        return float("nan")
    k = min(k, x.size)
    return float(np.mean(x[:k]))