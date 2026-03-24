import numpy as np
import cvxpy as cp
from configs.config_loader import ConfigLoader
from src.models.PF_models import PFModel


class VoltVarController:
    def __init__(self, node_num, v_ref=1.0, q_lim=0.5, v_db=0.02, capacity=0.5):
        """Initialize Volt/Var controller

        Args:
            node_num: node number this controller is connected to
            v_ref: reference voltage (p.u.)
            q_lim: reactive power limit (p.u.)
            v_db: voltage dead band (p.u.)
        """
        self.node = node_num
        self.v_ref = v_ref
        self.q_max = q_lim
        self.q_min = -q_lim
        self.v_db = v_db
        self.capacity = capacity

        self.v_pts = np.array([
            0.9,
            self.v_ref - self.v_db,
            self.v_ref,
            self.v_ref + self.v_db,
            1.1
        ])

        self.q_pts = np.array([
            self.q_max,
            0,
            0,
            0,
            self.q_min
        ])

    def get_q(self, v, p=None):
        """Get reactive power output based on voltage measurement

        Args:
            v: voltage measurement (p.u.)

        Returns:
            q: reactive power output (p.u.)
        """
        q = np.interp(v, self.v_pts, self.q_pts)

        if p is not None:
            q_lim = np.sqrt(self.capacity**2 - p**2)
            q = np.clip(q, -q_lim, q_lim)

        return q

    def get_q_cvx(self, v, only_limits=False):
        """Get reactive power constraints for optimization

        Args:
            v: voltage variable (cvxpy)
            only_limits: if True, only return the q limits constraints without piece-wise linear constraints

        Returns:
            q: cvxpy Variable for reactive power
            constraints: list of cvxpy constraints
        """
        constraints = []
        q = cp.Variable()

        if not only_limits:
            for i in range(len(self.v_pts) - 1):
                v1, v2 = self.v_pts[i:i + 2]
                q1, q2 = self.q_pts[i:i + 2]
                slope = (q2 - q1) / (v2 - v1)

                constraints += [q >= q1 + slope * (v - v1)]

        constraints += [q >= self.q_min, q <= self.q_max]

        return q, constraints


class DistributedVoltVarController:
    def __init__(self, configLoader=None):
        """Initialize distributed Volt/Var controller

        Args:
            node_list: list of nodes with controllers
            v_ref: reference voltage (p.u.)
            q_lim: reactive power limit (p.u.)
            v_db: voltage dead band (p.u.)
        """
        if configLoader is None:
            config_loader = ConfigLoader(None)
        else:
            config_loader = configLoader

        controller_config = config_loader.get_controller_config()
        node_list = controller_config['node_list']
        v_ref_list = controller_config['v_ref']
        q_max_list = controller_config['q_max']
        v_db_list = controller_config['v_db']
        capacity_list = controller_config['capacity']

        self.nodes = node_list
        self.controllers = {}

        for i, node in enumerate(node_list):
            num_setting = len(v_ref_list)
            idx = min(i, num_setting)
            self.controllers[node] = VoltVarController(
                node, v_ref_list[idx], q_max_list[idx], v_db_list[idx], capacity_list[idx]
            )

    def get_q(self, v_dict, p_dict=None):
        """Get reactive power output for all controlled nodes

        Args:
            v_dict: dictionary of voltage measurements {node(int): voltage}
            p_dict: dictionary of active power {node(int): voltage}

        Returns:
            q_dict: dictionary of reactive power outputs {node: q}
        """
        q_dict = {}
        for node in self.nodes:
            if node in v_dict:
                if p_dict is None:
                    q_dict[node] = self.controllers[node].get_q(v_dict[node])
                else:
                    q_dict[node] = self.controllers[node].get_q(v_dict[node], p_dict[node])
        return q_dict

    def get_q_cvx(self, v_dict):
        """Get reactive power constraints for optimization

        Args:
            v_dict: dictionary of voltage variables {node: v_var}

        Returns:
            q_dict: dictionary of reactive power variables
            constraints: list of constraints
        """
        q_dict = {}
        constraints = []

        for node in self.nodes:
            if node in v_dict:
                q, node_constraints = self.controllers[node].get_q_cvx(v_dict[node])
                q_dict[node] = q
                constraints.extend(node_constraints)

        return q_dict, constraints


class CentralizedVoltVarController:
    def __init__(self, model: PFModel, configLoader=None):
        """Initialize centralized Volt/Var controller.

        Centralized controller needs the full power flow model to construct constraints,
        therefore it is initialized with a PFModel object.

        Args:
            model: power flow model (PFModel)
            configLoader: configuration loader object (ConfigLoader)
        """

        if configLoader is None:
            config_loader = ConfigLoader(None)
        else:
            config_loader = configLoader

        controller_config = config_loader.get_controller_config()
        node_list = controller_config['node_list']
        v_ref_list = controller_config['v_ref']
        q_max_list = controller_config['q_max']
        v_db_list = controller_config['v_db']
        capacity_list = controller_config['capacity']

        self.nodes = node_list
        self.controllers = {}

        for i, node in enumerate(node_list):
            num_setting = len(v_ref_list)
            idx = min(i, num_setting)
            self.controllers[node] = VoltVarController(
                node, v_ref_list[idx], q_max_list[idx], v_db_list[idx], capacity_list[idx]
            )

        self.model = model

    def get_q(self, v_dict, p, q):
        """Get reactive power output for all controlled nodes

        Args:
            v_dict: dictionary of voltage measurements {node(int): voltage}
            p: active power injections at all buses except slack bus (num_buses-1,)
            q: reactive power injections at all buses except slack bus (num_buses-1,)
        """
        constraints = []

        q_injections = []

        for node in range(1, self.model.n_bus):
            if node in self.nodes:
                tmp_q, tmp_constraints = self.controllers[node].get_q_cvx(
                    v_dict[node], only_limits=True
                )
                q_injections.append(tmp_q + q[node - 1])
                constraints.extend(tmp_constraints)
            else:
                q_injections.append(cp.Constant(q[node - 1]))

        q_injections = cp.vstack(q_injections)

        power_flow_variable_dict = self.model.get_decision_variables(T=1)
        constraints.extend(
            self.model.get_PF_constraints(
                p=p.reshape(-1, 1), q=q_injections, variable_dict=power_flow_variable_dict
            )
        )


        node_voltage = [power_flow_variable_dict['v'][node, 0] for node in self.nodes]
        node_voltage = cp.hstack(node_voltage)

        objective_function = cp.Minimize(cp.norm2(node_voltage - 1.0))

        prob = cp.Problem(objective_function, constraints)
        prob.solve(solver=cp.MOSEK, verbose=False)

        if prob.status != cp.OPTIMAL:
            raise ValueError("Optimization problem not solved to optimality.")

        q_output = {}
        for node in self.nodes:
            q_output[node] = q_injections[node - 1, :].value - q[node - 1]

        return q_output



