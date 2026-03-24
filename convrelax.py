import numpy as np
import cvxpy as cp
from typing import Dict, List, Optional, Tuple, Union
from configs.config_loader import ConfigLoader
from src.models.PF_models import PFModel

class VoltVarController:

    def __init__(self, node_num: int, v_ref: float = 1.0, q_lim: float = 0.5,
                 v_db: float = 0.02, capacity: float = 0.5):
        self.node = node_num
        self.v_ref = v_ref
        self.q_max = float(q_lim)
        self.q_min = -float(q_lim)
        self.v_db = v_db
        self.capacity = capacity
        self.v_pts = np.array([0.9, self.v_ref - self.v_db, self.v_ref,
                               self.v_ref + self.v_db, 1.1], dtype=float)
        self.q_pts = np.array([self.q_max, 0.0, 0.0, 0.0, self.q_min], dtype=float)

    def get_q(self, v: float, p: Optional[float] = None) -> float:
        q = float(np.interp(v, self.v_pts, self.q_pts))
        if p is not None:
            q_cap = float(np.sqrt(max(self.capacity**2 - p**2, 0.0)))
            q = float(np.clip(q, -q_cap, q_cap))
        return float(np.clip(q, self.q_min, self.q_max))
    
    def get_q_cvx(self, v, only_limits: bool = False):
        q_total = cp.Variable(name=f"q_node{self.node}")
        constraints: List[cp.Constraint] = []
        if not only_limits:
            for i in range(len(self.v_pts) - 1):
                v1, v2 = float(self.v_pts[i]), float(self.v_pts[i + 1])
                q1, q2 = float(self.q_pts[i]), float(self.q_pts[i + 1])
                slope = (q2 - q1) / (v2 - v1)
                constraints += [q_total >= q1 + slope * (v - v1)]
        constraints += [q_total >= self.q_min, q_total <= self.q_max]
        return q_total, constraints
    
    
class CentralizedVoltVarController:

    def __init__(self, model: PFModel, configLoader: Optional[ConfigLoader] = None):
        if configLoader is None:
            config_loader = ConfigLoader(None)
        else:
            config_loader = configLoader
        cfg = config_loader.get_controller_config()
        node_list     = cfg['node_list']
        v_ref_list    = cfg['v_ref']
        q_max_list    = cfg['q_max']
        v_db_list     = cfg['v_db']
        capacity_list = cfg['capacity']
        self.model: PFModel = model
        self.n_bus = int(model.n_bus)
        self.n_red = self.n_bus - 1
        nodes_arr = np.asarray(node_list, dtype=int).ravel()
        if nodes_arr.size == 0:
            raise ValueError("controller node_list is empty.")
        max_node = int(nodes_arr.max())
        min_node = int(nodes_arr.min())
        if (min_node >= 0) and (max_node <= self.n_red - 1) and (0 in nodes_arr):
            self.index_mode = "reduced"
            self.nodes = nodes_arr.tolist()
        elif (min_node >= 1) and (max_node <= self.n_bus - 1) and (0 not in nodes_arr):
            self.index_mode = "full"
            self.nodes = nodes_arr.tolist()
        else:
            raise ValueError(
                f"node_list inconsistent with both modes. "
                f"min={min_node}, max={max_node}, n_bus={self.n_bus}."
            )
        self.controllers: Dict[int, VoltVarController] = {}
        for i, node in enumerate(self.nodes):
            idx = min(i, len(v_ref_list) - 1)
            self.controllers[node] = VoltVarController(
                node_num=node,
                v_ref=float(v_ref_list[idx]),
                q_lim=float(q_max_list[idx]),
                v_db=float(v_db_list[idx]),
                capacity=float(capacity_list[idx]),
            )

    def get_q(
        self,
        v_dict: Dict[int, float],
        p: Union[np.ndarray, List[float]],
        q: Union[np.ndarray, List[float]],
        *,
        s_ref: Union[np.ndarray, List[float]],
        c_action: float = 0.0,
        R_u: Optional[np.ndarray] = None,
        q_abs_max: Optional[float] = None,
        solver: Optional[str] = None,
        verbose: bool = False,
        return_absolute: bool = False
    ) -> Dict[int, float]:
        p_red  = np.asarray(p, dtype=float).reshape(-1, 1, order="F")
        q_base = np.asarray(q, dtype=float).reshape(-1, 1, order="F")
        if p_red.shape[0] != self.n_red or q_base.shape[0] != self.n_red:
            raise ValueError(f"p/q must have length n_bus-1={self.n_red}, got {p_red.shape}, {q_base.shape}")
        m = len(self.nodes)
        s_ref = np.asarray(s_ref, dtype=float).reshape(-1)
        if s_ref.size != m:
            raise ValueError(f"s_ref length {s_ref.size} must equal number of controlled nodes {m}.")
        u_by_node: Dict[int, cp.Expression] = {}
        q_total_by_node: Dict[int, cp.Expression] = {}
        constraints: List[cp.Constraint] = []
        if self.index_mode == "reduced":
            base_ctrl_list = [q_base[k, 0] for k in self.nodes]
        else:
            base_ctrl_list = [q_base[fb - 1, 0] for fb in self.nodes]
        q_base_ctrl_vec = cp.vstack(base_ctrl_list)
        for i, node in enumerate(self.nodes):
            u_i = cp.Variable(name=f"u_node{node}")
            u_by_node[node] = u_i
            q_total_i = q_base_ctrl_vec[i, 0] + u_i
            q_total_by_node[node] = q_total_i
            q_min = self.controllers[node].q_min
            q_max = self.controllers[node].q_max
            constraints += [q_total_i >= q_min, q_total_i <= q_max]
            if q_abs_max is not None:
                qM = float(q_abs_max)
                constraints += [q_total_i <= qM, q_total_i >= -qM]
        q_injections_list: List[cp.Expression] = []
        if self.index_mode == "reduced":
            nodes_set = set(self.nodes)
            for k in range(self.n_red):
                if k in nodes_set:
                    q_injections_list.append(q_total_by_node[k])
                else:
                    q_injections_list.append(cp.Constant(q_base[k, 0]))
        else:
            nodes_set = set(self.nodes)
            for k in range(self.n_red):
                fb = k + 1
                if fb in nodes_set:
                    q_injections_list.append(q_total_by_node[fb])
                else:
                    q_injections_list.append(cp.Constant(q_base[k, 0]))
        q_injections = cp.vstack(q_injections_list)
        vdict = self.model.get_decision_variables(T=1)
        constraints += self.model.get_PF_constraints(
            p=p_red, q=q_injections, variable_dict=vdict
        )
        if self.index_mode == "reduced":
            v_ctrl_elems = [vdict['v'][k + 1, 0] for k in self.nodes]
        else:
            v_ctrl_elems = [vdict['v'][fb, 0] for fb in self.nodes]
        v_ctrl = cp.hstack(v_ctrl_elems)
        voltage_cost = 5 * cp.sum_squares(v_ctrl - s_ref)
        q_total_ctrl_vec = cp.vstack([q_total_by_node[n] for n in self.nodes])
        if c_action != 0.0:
            if R_u is None:
                action_cost = float(c_action) * cp.sum_squares(q_total_ctrl_vec)
            else:
                R_u = np.asarray(R_u, dtype=float)
                if R_u.shape != (m, m):
                    raise ValueError(f"R_u must be {(m, m)}, got {R_u.shape}.")
                action_cost = float(c_action) * cp.quad_form(q_total_ctrl_vec, R_u)
        else:
            action_cost = 0.0
        prob = cp.Problem(cp.Minimize(voltage_cost + action_cost), constraints)
        try:
            prob.solve(solver=(solver or cp.OSQP), verbose=verbose)
        except Exception:
            try:
                prob.solve(solver=cp.SCS, verbose=verbose)
            except Exception:
                try:
                    prob.solve(solver=cp.ECOS, verbose=verbose)
                except Exception:
                    prob.solve(solver=cp.MOSEK, verbose=verbose)
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Optimization not optimal: {prob.status}")
        out: Dict[int, float] = {}
        if return_absolute:
            if self.index_mode == "reduced":
                for k in self.nodes:
                    out[k] = float(q_total_by_node[k].value)
            else:
                for fb in self.nodes:
                    out[fb] = float(q_total_by_node[fb].value)
        else:
            if self.index_mode == "reduced":
                for k in self.nodes:
                    out[k] = float(u_by_node[k].value)
            else:
                for fb in self.nodes:
                    out[fb] = float(u_by_node[fb].value)
        return out
