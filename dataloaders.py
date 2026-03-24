import numpy as np
from typing import Optional, Tuple, Dict, Sequence, Union


def build_netload_segments(
    data_pred: np.ndarray,
    control_nodes: np.ndarray,
    timehorizon: int,
    seed: Optional[int] = None,
    start_idx: Optional[int] = None,
    replace: Optional[bool] = None,
    return_augmented: bool = False,
):

    rng = np.random.default_rng(seed)
    data_pred = np.asarray(data_pred, dtype=float)
    num_nodes, time_steps = data_pred.shape
    control_nodes = np.asarray(control_nodes, dtype=int).ravel()
    num_control = control_nodes.size

    if timehorizon > time_steps:
        raise ValueError(f"timehorizon={timehorizon} longer than available time_steps={time_steps}")
    if timehorizon < 1:
        raise ValueError("timehorizon must be >= 1")

    if replace is None:
        replace = num_control > num_nodes

    chosen_rows = rng.choice(num_nodes, size=num_control, replace=replace)
    mapping: Dict[int, int] = {int(control_nodes[i]): int(chosen_rows[i]) for i in range(num_control)}

    if start_idx is None:
        start_idx = int(rng.integers(0, time_steps - timehorizon + 1))
    start_idx_used = int(start_idx)

    data_augmented = data_pred[chosen_rows, :]
    p_netload = data_augmented[:, start_idx_used:start_idx_used + timehorizon]

    if return_augmented:
        return p_netload, mapping, start_idx_used, chosen_rows, data_augmented
    return p_netload, mapping, start_idx_used


def compute_weighted_profiles(
    p_init: np.ndarray,
    q_init: np.ndarray,
    p_netload: np.ndarray,
    scale_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert baseline injections (p_init) and proxy load weights (p_netload) into a
    time-varying active power series p_ts.
    """
    p_init = np.asarray(p_init, dtype=float).reshape(-1, 1)
    q_init = np.asarray(q_init, dtype=float).reshape(-1, 1)
    p_netload = np.asarray(p_netload, dtype=float)

    m, timehorizon = p_netload.shape
    if p_init.shape[0] != m:
        raise ValueError(f"p_init has {p_init.shape[0]} rows but p_netload needs {m}")

    p_ts = p_init * (-(p_netload)) * float(scale_factor)
    return p_ts, p_init, q_init



def _broadcast_list(x, n, name: str):
    if isinstance(x, str):
        return [x] * n
    if isinstance(x, (list, tuple)):
        if len(x) != n:
            raise ValueError(f"{name} length {len(x)} does not match number of intervals {n}.")
        return list(x)
    return [x] * n


def _normalize_tspans(step_tspan, T: int):

    if step_tspan is None:
        return []

    if isinstance(step_tspan, (tuple, list)) and len(step_tspan) == 2 and np.isscalar(step_tspan[0]):
        spans = [tuple(map(int, step_tspan))]
    else:
        spans = [tuple(map(int, s)) for s in step_tspan]
    for (t0, t1) in spans:
        if not (0 <= t0 < t1 <= T):
            raise ValueError(f"Invalid tspan {(t0, t1)}; must satisfy 0 <= t0 < t1 <= {T}.")
    return spans

def _ensure_nodes(nodes, m: int):
    if nodes is None:
        return np.arange(m, dtype=int)
    arr = np.asarray(nodes, dtype=int).ravel()
    if np.any((arr < 0) | (arr >= m)):
        raise ValueError("step_nodes contains invalid indices.")
    return arr

def _signed_delta(delta: float, direction: str) -> float:
    direction = str(direction).lower()
    if direction in ("increase", "inc", "up", "+load", "more"):
        return -abs(delta)
    elif direction in ("decrease", "dec", "down", "-load", "less"):
        return +abs(delta)
    else:
        raise ValueError(f"Unknown step_direction '{direction}' (use 'increase' or 'decrease').")

def _apply_step_change(
    p_ts: np.ndarray,
    *,
    p_init: np.ndarray,
    step_nodes: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    step_tspan: Optional[Union[Tuple[int, int], Sequence[Tuple[int, int]]]] = None,
    step_delta: Optional[Union[float, Sequence[Optional[float]]]] = None,
    step_frac: Union[float, Sequence[float]] = 0.30,
    step_direction: Union[str, Sequence[str]] = "increase",
    clip_pu: Optional[Union[Tuple[float, float], Sequence[Optional[Tuple[float, float]]]]] = None,
) -> np.ndarray:
    P = np.asarray(p_ts, dtype=float).copy()
    m, T = P.shape
    p_init = np.asarray(p_init, dtype=float).reshape(-1, 1)
    if p_init.shape[0] != m:
        raise ValueError("p_init rows must match p_ts rows")

    spans = _normalize_tspans(step_tspan, T)
    if len(spans) == 0:
        return P

    step_nodes_list = _broadcast_list(step_nodes, len(spans), "step_nodes")
    step_delta_list = _broadcast_list(step_delta, len(spans), "step_delta")
    step_frac_list  = _broadcast_list(step_frac,  len(spans), "step_frac")
    step_dir_list   = _broadcast_list(step_direction, len(spans), "step_direction")
    clip_list       = _broadcast_list(clip_pu, len(spans), "clip_pu")

    baseline_mag = float(np.median(np.abs(p_init)))

    for i, (t0, t1) in enumerate(spans):
        nodes_i = _ensure_nodes(step_nodes_list[i], m)
        di = step_delta_list[i]
        if di is None:
            fi = float(step_frac_list[i])
            di = fi * baseline_mag
        di = float(di)
        signed_di = _signed_delta(di, step_dir_list[i])

        P[np.ix_(nodes_i, np.arange(t0, t1))] += signed_di

        clip_i = clip_list[i]
        if clip_i is not None:
            p_min, p_max = float(clip_i[0]), float(clip_i[1])
            P[np.ix_(nodes_i, np.arange(t0, t1))] = np.clip(
                P[np.ix_(nodes_i, np.arange(t0, t1))], p_min, p_max
            )

    return P


def prepare_inputs_with_measurements(
    data_pred: np.ndarray,
    control_nodes: np.ndarray,
    timehorizon: int,
    p_init: np.ndarray,
    q_init: np.ndarray,
    seed: Optional[int] = None,
    start_idx: Optional[int] = None,
    replace: Optional[bool] = None,
    scale_factor: float = 1.0,
    *,
    add_step: bool = False,
    step_nodes: Optional[np.ndarray] = None,
    step_tspan: Optional[Tuple[int, int]] = None,
    step_delta: Optional[float] = None,
    step_frac: float = 0.30,
    step_direction: str = "increase",
    clip_pu: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int], int]:
    """
      (1) builds a (node, time) weight matrix from `data_pred`,
      (2) converts it to a time-varying active power series `p_ts`, and
      (3) optionally injects a configurable step change (load jump) on selected nodes.

    Returns:
        p_ts: (n_bus-1, timehorizon)
        p_init: (n_bus-1, 1)
        q_init: (n_bus-1, 1)
        mapping: dict {control_node -> chosen_row_in_data_pred}
        start_idx_used: int
    """
    p_netload, mapping, start_idx_used = build_netload_segments(
        data_pred=np.asarray(data_pred, dtype=float),
        control_nodes=np.asarray(control_nodes, dtype=int),
        timehorizon=timehorizon,
        seed=seed,
        start_idx=start_idx,
        replace=replace,
    )

    p_ts, p_init, q_init = compute_weighted_profiles(
        p_init=np.asarray(p_init, dtype=float),
        q_init=np.asarray(q_init, dtype=float),
        p_netload=p_netload,
        scale_factor=scale_factor,
    )

    if add_step:
        p_ts = _apply_step_change(
            p_ts,
            p_init=p_init,
            step_nodes=step_nodes,
            step_tspan=step_tspan,
            step_delta=step_delta,
            step_frac=step_frac,
            step_direction=step_direction,
            clip_pu=clip_pu,
        )

    return p_ts, p_init, q_init, mapping, start_idx_used
