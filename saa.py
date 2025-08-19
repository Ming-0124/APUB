
import numpy as np
import gurobipy as gp
from gurobipy import GRB

class SAA:
    def __init__(self, model, c, n_items, n_machines):
        self.model = model
        self.c = c
        self.n_items = n_items
        self.n_machines = n_machines


    def solve_nf(self, params_list):
        try:
            N = len(params_list['h'])  # Number of scenarios
            model = gp.Model("TwoStage_SAA")

            # first stage variables
            x = model.addVars(self.n_items, lb=0, ub=5000, name="x")
            # second stage variables
            y = {}
            for n in range(N):
                y[n] = model.addVars(3*self.n_machines, lb=0, name=f"y_{n}")
               
            # objective function: c'x + (1/N) * sum( q_n' y_n )
            model.setObjective(
                gp.quicksum(self.c[i] * x[i] for i in range(self.n_items)) + (1/N) * 
                gp.quicksum(gp.quicksum(params_list['q'][n][j] * y[n][j] for j in range(self.n_machines)) for n in range(N)),GRB.MINIMIZE)
            
            # second stage constrints: W_n y_n = h_n - T_n x
            for n in range(N):
                for i in range(self.n_machines):
                    model.addConstr(
                        gp.quicksum(params_list['W'][n][i,j] * y[n][j] for j in range(3*self.n_machines)) ==
                        -gp.quicksum(params_list['T'][i, k] * x[k] for k in range(self.n_items)),name=f"second_stage_{n}_{i}")
                
                model.addConstr(gp.quicksum(y[n][i] for i in range(self.n_machines, 2*self.n_machines)) == params_list['h'][n][-1], name="Cap_Constr")
           
            # solve the model
            model.setParam('OutputFlag', 0)
            model.optimize()

            if model.status == GRB.OPTIMAL:
                x_opt = np.array([x[i].X for i in range(self.n_items)])
                obj_val = model.ObjVal
                return x_opt, obj_val
            else:
                print(f"Optimization failed with status {model.status}")
                return None, None

        except gp.GurobiError as e:
            print(f"Gurobi error: {e}")
        except Exception as e:
            print(f"Other error: {e}")


    def saa_oos(self, x: np.ndarray, test_params):
        N, J, I = len(test_params['h']), self.n_machines, self.n_items
        q, W, h, T = test_params['q'], test_params["W"], test_params["h"], test_params["T"]

        x_use = np.asarray(x[:I], dtype=float)
        c_term = float(np.dot(self.c[:I], x_use))

        # For each scenario, solve the second-stage recourse LP with fixed x
        q_vals = []
        for n in range(N):
            m = gp.Model(f"Q_n_{n}")
            m.setParam('OutputFlag', 0)
            y = m.addVars(3*J, lb=0.0, name="y")

            m.addConstr(gp.quicksum(y[j] for j in range(J,2*J)) == float(h[n][-1]))
            for i in range(self.n_machines):
                m.addConstr(
                    gp.quicksum(W[n][i,j] * y[j] for j in range(3*J)) == -gp.quicksum(
                        T[i,j] * x_use[j] for j in range(I)), name=f"Sub_Constr_{i}")

            m.setObjective(gp.quicksum(float(q[n, j]) * y[j] for j in range(J)), GRB.MINIMIZE)
            m.optimize()
            if m.status != GRB.OPTIMAL:
                raise RuntimeError(f"OOS second-stage failed at scenario {n} with status {m.status}")
            q_vals.append(float(m.objVal))

        recourse_mean = float(np.mean(q_vals))
        return c_term + recourse_mean
    