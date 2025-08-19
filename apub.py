import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy.stats import multinomial
from params_generator import ParametersGenerator
import time



class APUB:
    def __init__(self, A, b, c, n_items, n_machines, model):
        self.A = A
        self.b = b
        self.c = c
        self.model = model
        self.n_items = n_items
        self.n_machines = n_machines
        return

    def initialize_master_problem(self):
        """initialize the master problem"""
        x = self.model.addVars(self.n_items, lb=0, ub=2000, name="x")  # decision variable x
        eta = self.model.addVar(lb=-3000, name="eta")

        # first-stage objective function
        self.model.setObjective(gp.quicksum(self.c[i] * x[i] for i in range(self.n_items)) + eta, GRB.MINIMIZE)

        # first stage constraint Ax = b
        for i in range(self.n_machines):
            self.model.addConstr(gp.quicksum(self.A[i, j] * x[j] for j in range(self.n_items)) == self.b[i],
                                 name=f"First_Stage_Constr_{i}")
            
        return self.model.getVars()

    def solve_master_problem(self):
        """solve master problem"""
        self.model.setParam('OutputFlag', 0)
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            return self.model.getVars()
        else:
            raise Exception(f"master problem failed, status code: {self.model.status}")

    def check_feasibility(self, x_vals, params_list):
        for params in params_list:
            h_n, T_n, W_n = params['h'], params['T'], params['W']
            feas_model = gp.Model("Feasibility_Check")
            y = feas_model.addVars(3*self.n_machines, lb=0, name="y")
            v_p = feas_model.addVars(self.n_machines, lb=0, name="v_p")
            v_m = feas_model.addVars(self.n_machines, lb=0, name="v_m")

            # 约束：Wy + v = h - Tx
            for i in range(self.n_machines):
                feas_model.addConstr(
                    gp.quicksum(W_n[i, j] * y[j] for j in range(2*self.n_machines)) + v_p[i] - v_m[i] == h_n[i] - gp.quicksum(
                        T_n[i, j] * x_vals[j].X for j in range(2*self.n_machines)),
                    name=f"Feas_Constr_{i}")

            feas_model.setObjective(sum(v_p[i] + v_m[i] for i in range(self.n_machines)), GRB.MINIMIZE)
            feas_model.update()
            feas_model.setParam('OutputFlag', 0)
            feas_model.optimize()

            if feas_model.ObjVal > 1e-6:  # 不可行时生成切割
                phi = [constr.Pi for constr in feas_model.getConstrs()]  # 对偶变量
                D_new = np.dot(phi, T_n)  # 切割系数 D_j
                d_new = np.dot(phi, h_n)  # 切割常数 d_j
                self.model.addConstr(gp.quicksum(D_new[j] * x_vals[j].X for j in range(self.n_items)) >= d_new)
                self.model.update()
                print('feasile cut added')
                return True
        return False

    def generate_optimality_cuts(self, x_vals, params, alpha, M_bootstrap, eta_hat):
        N = len(params["h"])  
        Q_values = []
        duals = []
        T_list = []
        h_list = []

        # 计算所有样本的第二阶段成本和对偶乘子
        for m in range(N):
            q_n, W_n, h_n, T_n = params['q'][m], params['W'][m], params['h'][m], params['T']
            T_list.append(T_n)
            h_list.append(h_n)
            
            model = gp.Model("Second_Stage")
            y = model.addVars(3*self.n_machines, lb=0, name="y") 
                
            for i in range(self.n_machines):
                model.addConstr(
                    gp.quicksum(W_n[i,j] * y[j] for j in range(3*self.n_machines)) == -gp.quicksum(
                        T_n[i,j] * x_vals[j].X for j in range(self.n_items)), name=f"Sub_Constr_{i}")
         
            model.addConstr(gp.quicksum(y[i] for i in range(self.n_machines, 2*self.n_machines)) == h_n[-1], name="Cap_Constr")

            model.setObjective(gp.quicksum(q_n[j] * y[j] for j in range(3*self.n_machines)), GRB.MINIMIZE)
            model.setParam('OutputFlag', 0)
            model.optimize()
            if model.status == GRB.OPTIMAL:
                Q_values.append(model.objVal)
                duals.append([con.Pi for con in model.getConstrs()])
            else:
                print(f"Second stage optimization failed for params {params}, status code: {model.status}")
                raise Exception(f"second stage problem failed, status code: {self.model.status}")

        duals_array = np.array(duals)
        T_array = np.array(T_list)
        h_array = np.array(h_list)
        Q_values = np.array(Q_values)

        # # 预计算关键矩阵乘积
        dot_T = np.einsum('nm,nmi->ni', duals_array, T_array)  # duals[n] @ T_list[n]
        dot_h = np.einsum('nm,nm->n', duals_array, h_array)  # duals[n] @ h_list[n]
        # print('dot_T shape:', dot_T.shape, 'dot_h shape:', dot_h.shape)
        # generate Bootstrap samples
        bootstrap_indices = np.random.choice(N, size=(M_bootstrap, N), replace=True)
        V_mn = np.apply_along_axis(lambda x: np.bincount(x, minlength=N), axis=1, arr=bootstrap_indices)
        # #print('V_mn shape: ', V_mn.shape)  # (M_bootstrap, N)

        # 向量化计算统计量
        r = np.dot(Q_values, V_mn.T) / N
        E_m = np.dot(V_mn, dot_T) / N
        e_m = np.dot(V_mn, dot_h) / N
        # print('r:', r, 'E_m:', E_m, 'e_m:', e_m)
       
        J = int(np.ceil((1 - alpha) * M_bootstrap))
        sorted_indices = np.argsort(r)

        E_new = (1 - (M_bootstrap - J) / (alpha * M_bootstrap)) * E_m[sorted_indices[J]] + \
                (1 / (alpha * M_bootstrap)) * np.sum(E_m[sorted_indices[J: ]], axis=0)
        # print('E_new shape:', E_new.shape)

        e_new = (1 - (M_bootstrap - J) / (alpha * M_bootstrap)) * e_m[sorted_indices[J]] + \
                (1 / (alpha * M_bootstrap)) * np.sum(e_m[sorted_indices[J: ]])

        w_est = (1 - (M_bootstrap - J) / (alpha * M_bootstrap)) * r[sorted_indices[J]] + \
                (1 / (alpha * M_bootstrap)) * np.sum(r[sorted_indices[J: ]])

        if eta_hat.X >= w_est:
            return False
        self.model.addConstr(E_new @ x_vals + eta_hat >= e_new)
        return True

    def solve_two_stage_apub(self, random_params, alpha=0.1, M_bootstrap=1000):
        """
        求解两阶段APUB问题的L-Shaped算法
        :param random_params: List of {'q': q, 'W': W, 'h': h, 'T': T}
        :param alpha: APUB的置信水平
        :param M_bootstrap: Number of bootstrap samples
        :return: optimal solution x, optimal objective_value
        """
        self.initialize_master_problem()

        num_feasibility_cut = 0
        num_optimal_cut = 0

        while True:
            # Step 1: solve master problem
            *x_vars, eta_var = self.solve_master_problem()

            #Step 2: 生成可行性切割（直接传递变量对象）
            # cut_added = self.check_feasibility(x_vars, params_list=random_params)
            # if cut_added:
            #      num_feasibility_cut += 1
            #      continue

            # Step 3: 生成最优性切割（直接传递变量对象）
            cut_added = self.generate_optimality_cuts(x_vars, params=random_params, M_bootstrap=M_bootstrap,
                                                      alpha=alpha, eta_hat=eta_var)
            if cut_added:
                num_optimal_cut += 1
                # constraints = self.model.getConstrs()
                # for con in constraints:
                #     row = self.model.getRow(con)  # 左侧表达式
                #     expr = f"{row} {con.Sense} {con.RHS}"  # 拼接完整公式
                #     print(f"{con.ConstrName}: {expr}")
                continue
            else:
                break
        # print(f"Total optimality cuts added: {num_optimal_cut}")
        self.model.write("model.lp")
        *x_s, eta = self.model.getVars()
        x_opt = np.array([x_s[i].X for i in range(self.n_items)])
        obj_val = self.model.ObjVal

        return x_opt, eta, obj_val

    def extensive_form(self, params_list, alpha=0.2, M_bootstrap=500):
        N = len(params_list['h'])  # Number of original scenarios
        bootstrap_counts = multinomial.rvs(N, [1 / N] * N, size=M_bootstrap)

        try:
            model = gp.Model("TwoStage_APUB")
            # First-stage variables
            x = model.addVars(self.n_items, lb=0, ub=2000, name="x")
            t = model.addVar(lb=-GRB.INFINITY, name="t")
            s = model.addVars(M_bootstrap, lb=0, name="s")

            # Second-stage variables for each original scenario
            y = {}
            for n in range(N):
                y[n] = model.addVars(3*self.n_machines, lb=0, name=f"y_{n}")
              
            # Set objective: c'x + t + (1/(alpha*M)) * sum(s)
            model.setObjective(
                gp.quicksum(self.c[i] * x[i] for i in range(self.n_items)) + t + (1 / (alpha * M_bootstrap)) * gp.quicksum(
                    s[m] for m in range(M_bootstrap)),
                GRB.MINIMIZE
            )

            # First-stage constraints: Ax = b
            for i in range(self.A.shape[0]):
                model.addConstr(
                    gp.quicksum(self.A[i, j] * x[j] for j in range(self.n_items)) == self.b[i],
                    name=f"first_stage_{i}"
                )
            
            # Bootstrap constraints: s_m + t >= (1/N) * sum(V_mn * q_n' y_n) for each m
            for m in range(M_bootstrap):
                
                V_m = bootstrap_counts[m]
                model.addConstr(
                    s[m] + t >= (1 / N) * gp.quicksum(
                        V_m[n] * gp.quicksum(params_list['q'][n][j] * y[n][j] for j in range(self.n_machines))
                        for n in range(N)
                    ),
                    name=f"bootstrap_{m}"
                )

            # Second-stage constraints: W_n y_n = h_n - T_n x for each n
            for n in range(N):
                for i in range(self.n_machines):
                    model.addConstr(
                        gp.quicksum(params_list['W'][n][i,j] * y[n][j] for j in range(3*self.n_machines)) ==
                        -gp.quicksum(params_list['T'][i, k] * x[k] for k in range(self.n_items)),
                        name=f"second_stage_{n}_{i}"
                    )
                model.addConstr(gp.quicksum(y[n][j] for j in range(self.n_machines,2*self.n_machines)) == params_list['h'][n][-1],
                                 name=f"cap_constr_{n}_{i}")
                   
            # Optimize model
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

if __name__ == "__main__":
    b = np.zeros(4)
    A = np.zeros((4, 12))
    from utils import sample_from_config, load_config
    cfg_path = "config.yaml"
    # Load shared sampling hyperparameters from config
    full_cfg = load_config(cfg_path)
    rg_cfg = full_cfg.get("random_generator", full_cfg)
    c = list(rg_cfg["c"])

    pg = ParametersGenerator()
    alpha = 0.9
    samples = pg.generate_parameters(sample_from_config(cfg_or_path=cfg_path, train=True))
    
    apub = APUB(A, b, c=c, n_items=12, n_machines=4, model=gp.Model("APUB_Model"))
    start1 = time.perf_counter()
    b,a = apub.extensive_form(samples, alpha=alpha, M_bootstrap=1500)
    end1 = time.perf_counter()
    print(f'extensive form: {a}, {b}')
    print(f'extensive form time: {end1-start1}s')
    start2 = time.perf_counter()
    x_optimal, eta_optimal, x = apub.solve_two_stage_apub(samples,alpha=alpha,M_bootstrap=1500)
    end2 = time.perf_counter()
    print(f'lshape: {x_optimal}, {x}')
    print(f'lshapee time: {end2-start2}s')
