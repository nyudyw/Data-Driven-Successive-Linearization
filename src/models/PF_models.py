import numpy as np
import cvxpy as cp
from numpy.linalg  import inv

class PFModel:
    def __init__(self,case):
        pass
    def runpf(self,p,q):
        pass
    def get_decision_variables(self,T=1):
        pass
    def set_time_horizon(self,T):
        self.T = T
    def get_PF_constraints(self,p,q,variable_dict):
        pass
    def get_line_constraints(self,variable_dict):
        pass
    def get_other_constraints(self,variable_dict):
        pass

class LinDistFlow(PFModel):
    def __init__(self, case: dict):
        """Initialize LinDistFlow model with case data
        
        Args:
            case: dict containing network data with following keys:
                'bus': bus data
                'branch': branch data containing r, x values
                'baseMVA': base MVA
        """
        super().__init__(case)
        
        self.case = case
        self.n_bus = len(case['bus'])
        self.n_branch = len(case['branch'])
        self.baseMVA = case['baseMVA']
        self.ref_node_num = np.where(case['bus'][:,1]==3)[0]
        
        self.from_bus = case['branch'][:, 0].astype(int) - 1
        self.to_bus = case['branch'][:, 1].astype(int) - 1
        
        self.r = case['branch'][:, 2]
        self.x = case['branch'][:, 3]
        
        self.T = 1
        
        self.C = np.zeros((self.n_bus, self.n_branch))
        for i in range(self.n_branch):
            self.C[self.from_bus[i], i] = 1
            self.C[self.to_bus[i], i] = -1
        self.C_ = np.delete(self.C,self.ref_node_num,0)
        self.C_inv = inv(self.C_)

        self.Dr = np.diag(self.r)
        self.Dx = np.diag(self.x)

        self.R = self.C_inv.T @ self.Dr @ self.C_inv
        self.X = self.C_inv.T @ self.Dx @ self.C_inv
            
        self.vmin = case['bus'][:, 12]
        self.vmax = case['bus'][:, 11]

        self.variables = {}

    def runpf(self, p, q):
        """Run power flow for given P, Q injections
        
        Args:
            p: real power injections (n_bus-1 x T), unit: p.u., direction: injection
            q: reactive power injections (n_bus-1 x T), unit: p.u., direction: injection
            
        Returns:
            dict containing:
                'v': voltage magnitudes (n_bus x T), magnitude of ref bus is 1
                'theta': voltage angle (n_bus x T), angle of ref bus is 0 
                'p': real power flows (n_bus x T)
                'q': reactive power flows (n_bus x T)
        """
        p = np.atleast_2d(p)
        q = np.atleast_2d(q)
        
        v = np.ones((self.n_bus, self.T))
        theta = np.zeros((self.n_bus,self.T))
        P = np.zeros((self.n_branch, self.T))
        Q = np.zeros((self.n_branch, self.T))
        p0 = np.zeros((1,self.T))
        q0 = np.zeros((1,self.T))
        
        for t in range(self.T):
            P[:, t:t+1] = -np.linalg.solve(self.C_, p)
            Q[:, t:t+1] = -np.linalg.solve(self.C_, q)
            
            v[1:,t:t+1] = 1+2*(self.R @ p + self.X @ q)
            delta_theta = np.maximum(self.C.T,0)@v[:,t:t+1]-(self.Dr-1j*self.Dx)@(P[:,t:t+1]+1j*Q[:,t:t+1])
            delta_theta = np.angle(delta_theta)
            theta[1:,t:t+1] = self.C_.T@delta_theta*(180/np.pi)

            p0[t:t+1] = self.C[0:1,:]@P[:,t:t+1]
            q0[t:t+1] = self.C[0:1,:]@Q[:,t:t+1]
        
        p = np.vstack((p0,p))
        q = np.vstack((q0,q))
    
        return {'v': np.sqrt(v), 'theta':theta, 'p': p, 'q': q}
    
    def get_decision_variables(self,T=1):
        """Initialize optimization variables
        Args:
            T: number of time steps, default is 1
        Output: 
            dict of cvxpy Variables
        """
        return {
            'P': cp.Variable((self.n_branch, T)),
            'Q': cp.Variable((self.n_branch, T)),
            'v': cp.Variable((self.n_bus, T)),
            'p0': cp.Variable((1,T)),
            'q0': cp.Variable((1,T))
        }
    
    def get_PF_constraints(self,p,q,variable_dict):
        """Get power flow constraints based on DistFlow model
        Args:
            p: real power injections (n_bus-1 x T), unit: p.u., direction: injection
            q: reactive power injections (n_bus-1 x T), unit: p.u., direction: injection
        Returns:
            list of power flow constraints
        """

        v = variable_dict['v']
        P = variable_dict['P']
        Q = variable_dict['Q']
        p0 = variable_dict['p0']
        q0 = variable_dict['q0']

        constraints = []
        
        constraints += [v[0, :] == 1]
        
        constraints += [self.C_ @ P == p]  
        constraints += [self.C_ @ Q == q]
        
        constraints += [self.C[0:1, :] @ P == p0]
        constraints += [self.C[0:1, :] @ Q == q0]
        
        for t in range(self.T):
            for i in range(self.n_branch):
                to_idx = self.to_bus[i]
                from_idx = self.from_bus[i]
                constraints += [
                    v[to_idx, t] == v[from_idx, t] - 
                    2 * (self.r[i] * P[i, t] + self.x[i] * Q[i, t])
                ]
        
        return constraints

    def get_line_constraints(self, variable_dict):
        """Get line flow constraints"""

        P = variable_dict['P']
        Q = variable_dict['Q']

        constraints = []
        
        if hasattr(self, 'rate_a'):
            for i in range(self.n_branch):
                if self.rate_a[i] > 0:
                    for t in range(self.T):
                        constraints += [
                            cp.square(P[i, t]) + cp.square(Q[i, t]) <= 
                            cp.square(self.rate_a[i])
                        ]
        
        return constraints

    def get_other_constraints(self, variable_dict):
        """Get voltage limits and other constraints"""

        v = variable_dict['v']

        constraints = []
        
        for i in range(self.n_bus):
            constraints += [v[i, :] >= self.vmin[i]]
            constraints += [v[i, :] <= self.vmax[i]]
        
        return constraints
    

    def set_horizon(self, T):
        """Set time horizon for multi-period analysis
        
        Args:
            T: number of time periods
        """
        self.T = T


class DistFlow(PFModel):
    def __init__(self, case):
        """Initialize DistFlow model with case data
        
        Args:
            case: dict containing network data with following keys:
                'bus': bus data
                'branch': branch data containing r, x values
                'baseMVA': base MVA
        """
        super().__init__(case)
        
        self.case = case
        self.n_bus = len(case['bus'])
        self.n_branch = len(case['branch'])
        self.baseMVA = case['baseMVA']
        self.ref_node_num = np.where(case['bus'][:,1]==3)[0]
        
        self.from_bus = case['branch'][:, 0].astype(int) - 1
        self.to_bus = case['branch'][:, 1].astype(int) - 1
        
        self.r = case['branch'][:, 2]
        self.x = case['branch'][:, 3]
        
        self.T = 1
        
        self.C = np.zeros((self.n_bus, self.n_branch))
        for i in range(self.n_branch):
            self.C[self.from_bus[i], i] = 1
            self.C[self.to_bus[i], i] = -1
        self.C_ = np.delete(self.C, self.ref_node_num, 0)
        
        self.Dr = np.diag(self.r)
        self.Dx = np.diag(self.x)
            
        self.vmin = case['bus'][:, 12]
        self.vmax = case['bus'][:, 11]

        self.variables = {}
    def solve_power_flow(self, s_inj, method='cvxpy'):
        """Solve power flow equations using specified method
        
        Args:
            s_inj: complex power injections (n_bus-1)
            method: 'cvxpy' or 'nl_solver'
            
        Returns:
            dict containing solution, solutions are numpy arrays
        """
        if method == 'cvxpy':
            return self._solve_cvxpy(s_inj)
        elif method == 'nl_solver':
            return self._solve_nonlinear(s_inj)
        else:
            raise ValueError("Method must be 'cvxpy' or 'nl_solver'")

    def _solve_cvxpy(self, s_inj):
        """Solve using CVXPY with convex relaxation"""
        n = self.n_bus
        m = self.n_branch
        
        v = cp.Variable(n)
        S = cp.Variable(m, complex=True)
        l = cp.Variable(m)

        s = cp.Variable(n, complex=True)

        constraints = []
        
        constraints += [v[0] == 1]
        constraints += [s[1:] == s_inj]
        
        for j in range(n):
            out_branches = [i for i in range(m) if self.from_bus[i] == j]
            in_branches = [i for i in range(m) if self.to_bus[i] == j]
            
            sum_out = sum(S[k] for k in out_branches)
            sum_in = sum(S[i] - (self.r[i] + 1j*self.x[i])*l[i] 
                        for i in in_branches)
            constraints += [sum_out == sum_in + s[j]]

        for i in range(m):
            j = self.from_bus[i]
            k = self.to_bus[i]
            z = self.r[i] + 1j*self.x[i]
            constraints += [
                v[j] - v[k] == 2*cp.real(cp.conj(z)*S[i]) - abs(z)**2*l[i]
            ]

        for i in range(m):
            j = self.from_bus[i]
            constraints += [

                cp.norm2(cp.hstack([2*cp.real(S[i]), 2*cp.imag(S[i]), l[i] - v[j]])) <= l[i] + v[j]
            ]


        objective = cp.Minimize(sum(self.r[i]*l[i] for i in range(m)))

        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.MOSEK)
        
        return {
            'v': np.sqrt(v.value),
            'S': S.value,
            'l': l.value,
            'status': prob.status,
            'cost': prob.value
        }

    def _solve_nonlinear(self, s_inj):
        """Solve using nonlinear solver"""
        from scipy.optimize import root
        
        def equations(x):
            n = self.n_bus
            m = self.n_branch
            
            v = x[:n]
            S_real = x[n:n+m]
            S_imag = x[n+m:n+2*m]
            l = x[n+2*m:]
            S = S_real + 1j*S_imag
            
            residual = [v[0]-1]
            
            for j in range(1, n):
                out_branches = [i for i in range(m) if self.from_bus[i] == j]
                in_branches = [i for i in range(m) if self.to_bus[i] == j]
                
                sum_out = sum(S[k] for k in out_branches)
                sum_in = sum(S[i] - (self.r[i] + 1j*self.x[i])*l[i] 
                            for i in in_branches)
                res = sum_out - sum_in - s_inj[j-1]
                residual.extend([res.real, res.imag])
            
            for i in range(m):
                j = self.from_bus[i]
                k = self.to_bus[i]
                z = self.r[i] + 1j*self.x[i]
                
                res_v = v[j] - v[k] - 2*np.real(np.conj(z)*S[i]) + abs(z)**2*l[i]
                res_l = v[j]*l[i] - abs(S[i])**2
                
                residual.extend([res_v, res_l])
            
            return residual
        
        n = self.n_bus
        m = self.n_branch
        x0 = np.ones(3*self.n_branch + self.n_bus)
        x0[n:] = 0
        
        solution = root(equations, x0, method='hybr')
        
        v = solution.x[:n]
        S = solution.x[n:n+m] + 1j*solution.x[n+m:n+2*m]
        l = solution.x[n+2*m:]
        
        return {
            'v': np.sqrt(v),
            'S': S,
            'l': l,
            'success': solution.success,
            'message': solution.message
        }

    def runpf(self, p, q, method='nl_solver'):
        """Run power flow for given P, Q injections
        
        Args:
            p: real power injections (n_bus-1 x T)
            q: reactive power injections (n_bus-1 x T)
            method: 'cvxpy' or 'nl_solver'
        
        Returns:
            dict containing:
                'v': voltage magnitudes (n_bus x T)
                'S': complex power flows (n_branch x T)
                'l': squared current magnitudes (n_branch x T)
        """
        p = np.atleast_2d(p)
        q = np.atleast_2d(q)
        
        s = p + 1j*q
        
        results = []
        for t in range(self.T):
            result = self.solve_power_flow(s[:, t], method=method)
            results.append(result)

        v = np.column_stack([r['v'] for r in results])
        S = np.column_stack([r['S'] for r in results])
        l = np.column_stack([r['l'] for r in results])
        
        return {
            'v': v,
            'S': S,
            'l': l
        }

    def get_decision_variables(self,T=1):
        """Initialize optimization variables
        Args: 
            T: number of time steps, default is 1
        Output: 
            dict of cvxpy Variables
        """

        n = self.n_bus
        m = self.n_branch

        return{
            'v': cp.Variable((n, T)),
            'S': cp.Variable((m, T), complex=True),
            'l': cp.Variable((m, T)),
            's': cp.Variable((n, T), complex=True)
        }

    def get_PF_constraints(self, p, q, variable_dict):
        """Get power flow constraints for non-linear DistFlow model
        Args:
            p: real power injections (n_bus-1 x T), can be np.ndarray or cp.Variable
            q: reactive power injections (n_bus-1 x T), can be np.ndarray or cp.Variable
        Returns:
            list of power flow constraints
        """

        if isinstance(p, np.ndarray):
            p = np.atleast_2d(p)
        elif isinstance(p, cp.Expression):
            p = cp.reshape(p, (self.n_bus-1, -1))
        else:
            raise ValueError("p must be either a numpy array or a cvxpy expression")

        if isinstance(q, np.ndarray):
            q = np.atleast_2d(q)
        elif isinstance(q, cp.Expression):
            q = cp.reshape(q, (self.n_bus-1, -1))
        else:
            raise ValueError("q must be either a numpy array or a cvxpy expression")

        n = self.n_bus  
        m = self.n_branch

        v_total = variable_dict['v']
        S_total = variable_dict['S']
        l_total = variable_dict['l']
        s_total = variable_dict['s']

        T = v_total.shape[1]

        constraints = []

        for t in range(T):
            v = v_total[:, t]
            S = S_total[:, t]
            l = l_total[:, t]
            s = s_total[:, t]

            s_inj = p[:, t] + 1j*q[:, t]
            
            constraints += [v[0] == 1]
            constraints += [s[1:] == s_inj]
            
            for j in range(n):
                out_branches = [i for i in range(m) if self.from_bus[i] == j]
                in_branches = [i for i in range(m) if self.to_bus[i] == j]
                
                sum_out = sum(S[k] for k in out_branches)
                sum_in = sum(S[i] - (self.r[i] + 1j*self.x[i])*l[i] 
                            for i in in_branches)
                constraints += [sum_out == sum_in + s[j]]

            for i in range(m):
                j = self.from_bus[i]
                k = self.to_bus[i]
                z = self.r[i] + 1j*self.x[i]
                constraints += [
                    v[j] - v[k] == 2*cp.real(cp.conj(z)*S[i]) - abs(z)**2*l[i]
                ]

            for i in range(m):
                j = self.from_bus[i]
                constraints += [
                    cp.norm2(cp.hstack([2*cp.real(S[i]), 2*cp.imag(S[i]), l[i] - v[j]])) <= l[i] + v[j]
                ]
    
        return constraints

    def get_line_constraints(self, variable_dict):
        """Get line flow constraints"""

        S = variable_dict['S']

        constraints = []
        
        if hasattr(self, 'rate_a'):
            for i in range(self.n_branch):
                if self.rate_a[i] > 0:
                    for t in range(self.T):
                        constraints += [
                            cp.square(cp.abs(S[i, t])) <= 
                            cp.square(self.rate_a[i])
                        ]
        
        return constraints

    def get_other_constraints(self, variable_dict):
        """Get voltage limits and other constraints"""

        v = variable_dict['v']

        constraints = []
        
        for i in range(self.n_bus):
            constraints += [v[i, :] >= self.vmin[i]**2]
            constraints += [v[i, :] <= self.vmax[i]**2]
        
        return constraints