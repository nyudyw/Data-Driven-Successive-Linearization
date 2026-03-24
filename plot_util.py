import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import ScalarFormatter
from typing import Sequence, Tuple, Union, Dict, Any, Optional, List, Dict, Union, Type

def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1:"st", 2:"nd", 3:"rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

def report_step_change_costs(
    record_cost: Sequence[float],
    step_tspan: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
    num_steps: int = 10,
    start: str = "start",
    *,
    header: str = "Step-change cost report",
    return_results: bool = False
) -> Dict[str, Any]:

    rc = np.asarray(record_cost, dtype=float).ravel()
    n = len(rc)

    if isinstance(step_tspan, (tuple, list)) and len(step_tspan) == 2 and np.isscalar(step_tspan[0]):
        spans = [tuple(map(int, step_tspan))]
    else:
        spans = [tuple(map(int, s)) for s in step_tspan]

    for (t0, t1) in spans:
        if not (0 <= t0 < t1):
            raise ValueError(f"Invalid tspan {(t0, t1)}: must satisfy 0 <= t0 < t1.")

    print("=" * 60)
    print(header)
    print("=" * 60)

    results: Dict[str, Any] = {'spans': spans, 'segments': []}

    for k, (t0, t1) in enumerate(spans, start=1):
        start_idx = t0 if start == "start" else t1
        start_idx = int(np.clip(start_idx, 0, n))
        end_idx = int(np.clip(start_idx + int(num_steps), 0, n))

        seg = rc[start_idx:end_idx]
        label = f"{_ordinal(k)} step change"
        title = (
            f"{label}: window [t0, t1) = [{t0}, {t1}), "
            f"reporting steps [{start_idx}, {end_idx-1}] (n={len(seg)})"
        )

        print("\n" + title)
        print("-" * len(title))

        if len(seg) == 0:
            print("(no data in the requested range)")
            results['segments'].append({
                'label': label, 'tspan': (t0, t1),
                'start': start_idx, 'end': end_idx, 'costs': seg,
                'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan
            })
            continue

        print(f"{'t':>8} | {'cost':>14}")
        print("-" * 26)
        for t, c in enumerate(seg, start=start_idx):
            print(f"{t:>8d} | {float(c):>14.6g}")

        stats = {
            'mean': float(np.mean(seg)),
            'std':  float(np.std(seg)),
            'min':  float(np.min(seg)),
            'max':  float(np.max(seg)),
        }
        print("-" * 26)
        print(
            f"mean: {stats['mean']:.6g}    std: {stats['std']:.6g}    "
            f"min: {stats['min']:.6g}    max: {stats['max']:.6g}"
        )

        results['segments'].append({
            'label': label, 'tspan': (t0, t1),
            'start': start_idx, 'end': end_idx, 'costs': seg, **stats
        })

    print("=" * 60)
    return results if return_results else None

def add_zoom_inset_auto(
    ax, x, Y, x_zoom=(20, 30), size=0.38,
    loc_candidates=("upper right", "upper left", "lower left", "lower right"),
    *, inset_ticklabelsize: int = 15,
    inset_ylim=None,
):
    x = np.asarray(x); Y = np.asarray(Y)
    y_rep = np.median(Y, axis=1) if Y.ndim == 2 else Y

    ax.figure.canvas.draw()
    xL, xR = ax.get_xlim(); yB, yT = ax.get_ylim()
    xW, yH = (xR - xL), (yT - yB); box_w, box_h = xW * size, yH * size
    candidates = {
        "upper right":  ((xR - box_w, xR), (yT - box_h, yT)),
        "upper left":   ((xL, xL + box_w), (yT - box_h, yT)),
        "lower left":   ((xL, xL + box_w), (yB, yB + box_h)),
        "lower right":  ((xR - box_w, xR), (yB, yB + box_h)),
    }
    best_loc, best_score = None, None
    for loc in loc_candidates:
        (cx0, cx1), (cy0, cy1) = candidates[loc]
        mask = (x >= cx0) & (x <= cx1) & (y_rep >= cy0) & (y_rep <= cy1)
        score = int(np.count_nonzero(mask))
        if best_score is None or score < best_score:
            best_score, best_loc = score, loc

    iax = inset_axes(
        ax,
        width=f"{int(size*100)}%",
        height=f"{int(size*100)}%",
        loc=best_loc,
        borderpad=1.0
    )

    if Y.ndim == 2:
        iax.plot(x, Y, lw=1.2)
    else:
        iax.plot(x, Y, lw=1.6)

    x0, x1 = x_zoom
    iax.set_xlim(x0, x1)

    iax.xaxis.set_major_locator(MultipleLocator(5))


    if inset_ylim is not None:
        iax.set_ylim(inset_ylim[0], inset_ylim[1])
    else:
        zmask = (x >= x0) & (x <= x1)
        if np.any(zmask):
            if Y.ndim == 2:
                y_min = float(np.min(Y[zmask, :])); y_max = float(np.max(Y[zmask, :]))
            else:
                y_min = float(np.min(Y[zmask]));    y_max = float(np.max(Y[zmask]))
            pad = 0.08 * (y_max - y_min + 1e-12)
            iax.set_ylim(y_min - pad, y_max + pad)

    iax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    iax.tick_params(axis="both", labelsize=inset_ticklabelsize)
    yfmt = ScalarFormatter(useMathText=True)
    yfmt.set_scientific(True)
    yfmt.set_powerlimits((-3, 3))
    yfmt.set_useOffset(False)
    iax.yaxis.set_major_formatter(yfmt)

    offset_text = iax.yaxis.get_offset_text()
    offset_text.set_fontsize(inset_ticklabelsize)

    try:
        ax.indicate_inset_zoom(iax, edgecolor='0.25', alpha=0.9)
    except Exception:
        pass

    return iax, best_loc



def compute_ylim(arr_list, manual=None, *, pad_frac=0.06,
                 symmetric=False, min_span=1e-8):
    if manual is not None:
        y0, y1 = float(manual[0]), float(manual[1])
        if symmetric:
            m = max(abs(y0), abs(y1))
            return (-m, m)
        return (y0, y1)

    y_min, y_max = None, None
    for A in arr_list:
        A = np.asarray(A, dtype=float)
        if A.size == 0:
            continue
        a_min = float(np.nanmin(A))
        a_max = float(np.nanmax(A))
        y_min = a_min if y_min is None else min(y_min, a_min)
        y_max = a_max if y_max is None else max(y_max, a_max)

    if (y_min is None) or (y_max is None) or (
        not np.isfinite([y_min, y_max]).all()
    ):
        return (-1.0, 1.0)

    if abs(y_max - y_min) < min_span:
        mid = 0.5 * (y_max + y_min)
        y_min, y_max = mid - 0.5, mid + 0.5

    if symmetric:
        m = max(abs(y_min), abs(y_max))
        y_min, y_max = -m, m

    pad = pad_frac * (y_max - y_min)
    return (y_min - pad, y_max + pad)


def plot_pts_for_seed(
    *,
    p_ts_long: np.ndarray,
    p_init: np.ndarray,
    seed: int,
    start: int,
    step_tspan: List[Tuple[int, int]],
    num_nodes_to_plot: int = 9,
    rng_base: int = 42,
):
    p_ts_long = np.asarray(p_ts_long, dtype=float)
    p_init = np.asarray(p_init, dtype=float).reshape(-1, 1)

    m, L = p_ts_long.shape
    if p_init.shape[0] != m:
        raise ValueError(f"p_init rows ({p_init.shape[0]}) != p_ts_long rows ({m})")

    rng = np.random.default_rng(rng_base + seed)
    k = min(int(num_nodes_to_plot), m)
    sel = np.sort(rng.choice(np.arange(m), size=k, replace=False))

    t = np.arange(L)
    p_init_time = np.repeat(p_init, L, axis=1)

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    fig, ax = plt.subplots(figsize=(20.5, 7.5), dpi=120)

    cmap = plt.cm.get_cmap("tab10", len(sel))
    colors = [cmap(i) for i in range(len(sel))]

    for k_i, node_idx in enumerate(sel):
        ax.plot(t, p_ts_long[node_idx, :], lw=1.8, alpha=0.95,
                color=colors[k_i], label=f"dim {int(node_idx)} (p_ts)")
        ax.plot(t, p_init_time[node_idx, :], lw=1.4, alpha=0.95, ls="--",
                color=colors[k_i], label=f"dim {int(node_idx)} (p_init)")

    for (a, b) in step_tspan:
        a = max(0, min(int(a), L))
        b = max(0, min(int(b), L))
        if b > a:
            ax.axvspan(a, b, color="grey", alpha=0.10, linewidth=0)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Active power p (p.u.)")
    ax.set_title(f"$p_{{ts}}$ (solid) and $p_{{init}}$ (dashed) | seed={seed}, start={start}")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    ax.set_xticks(np.arange(0, L + 1, 5))

    handles, labels = ax.get_legend_handles_labels()
    kept = []
    seen = set()
    for h, lab in zip(handles, labels):
        try:
            dim = int(lab.split()[1])
        except Exception:
            dim = lab
        key = (dim, lab.endswith("(p_ts)"))
        if key not in seen:
            kept.append((h, lab))
            seen.add(key)

    ax.legend(*zip(*kept),
              loc="center left",
              bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0.0,
              frameon=True,
              ncol=1)

    plt.tight_layout(rect=(0, 0, 0.80, 1))
    plt.show()

    return sel



def tail_mean_1traj(cost_arr, start_idx, k=5):
    cost_arr = np.asarray(cost_arr, float).ravel()
    seg = cost_arr[start_idx:]   
    if seg.size == 0:
        return float("nan")
    k = min(k, seg.size)
    return float(np.mean(seg[-k:]))

def pct_lower(sc_val, other_val):
    if (not np.isfinite(sc_val)) or (not np.isfinite(other_val)) or other_val <= 0:
        return float("nan")
    return 100.0 * (1.0 - sc_val / other_val)