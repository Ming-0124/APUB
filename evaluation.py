import pickle
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from apub import APUB
import time
import multiprocessing as mp
from params_generator import ParametersGenerator
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.font_manager import FontProperties
from utils import sample_from_config
from concurrent.futures import ProcessPoolExecutor, as_completed


def evaluate_oos(certificate, x_optimal, test_samples, c, n_items, n_machines):
    costs = []
    N = len(test_samples['h'])

    for m in range(N):
        W_test = test_samples['W'][m]
        h_test = test_samples['h'][m]
        T_test = test_samples['T']
        q_test = test_samples['q'][m]

        # second stage 
        sub_model = gp.Model("OOS_Evaluation")
        y = sub_model.addVars(3*n_machines, lb=0)
        sub_model.setObjective(gp.quicksum(q_test[j] * y[j] for j in range(n_machines)), GRB.MINIMIZE)
        # constraint：Wy = h - Tx
        for i in range(n_machines):
            sub_model.addConstr(
                gp.quicksum(W_test[i, j] * y[j] for j in range(3*n_machines)) == -gp.quicksum(
                    T_test[i, j] * x_optimal[j] for j in range(n_items)),name=f"Sub_Constr_{i}")
        sub_model.addConstr(gp.quicksum(y[j] for j in range(n_machines, 2*n_machines)) == h_test[-1], name="Cap_Constr")

        sub_model.setParam('OutputFlag', 0)
        sub_model.optimize()

        temp = 0
        if sub_model.status == GRB.OPTIMAL:
            for i in range(len(x_optimal)):
                temp += c[i] * x_optimal[i]
            total_cost = temp + sub_model.ObjVal
            costs.append(total_cost)
        else:
            costs.append(np.inf) 

    mean_cost = np.mean(costs)
    return {
        'mean_cost': mean_cost,
        'reliability': int(certificate >= mean_cost)
    }


def process_M(args):
    """Worker function to process a single M value"""
    A, b, n_items, n_machines, M, xi_samples_list = args
    tt1 = []
    tt2 = []
    cuts = []
    
    for i in range(20):
        apub = APUB(A, b, n_items=n_items, n_machines=n_machines, model=gp.Model('Master Problem'))
        
        # Extensive form timing
        start1 = time.perf_counter()
        apub.extensive_form(xi_samples_list[i], alpha=0.1, M_bootstrap=M)
        end1 = time.perf_counter()
        
        # L-shape method timing
        start2 = time.perf_counter()
        _, _, _, num_optimal_cuts = apub.solve_two_stage_apub(
            xi_samples_list[i],
            alpha=0.1,
            M_bootstrap=M,
        )
        end2 = time.perf_counter()
        
        tt1.append(end1 - start1)
        tt2.append(end2 - start2)
        cuts.append(num_optimal_cuts)
    
    return {
        'M': M,
        'extensive_form_time': np.mean(tt1),
        'lshape_time': np.mean(tt2),
        'cuts': np.mean(cuts)
    }

def evaluate_M_T_performance(A, b, M_list, n_items, n_machines, save_path='./results/time.json'):
    result = defaultdict(lambda: defaultdict(dict))
    
    # Get the number of CPU cores (leave one core free)
    n_cores = max(1, mp.cpu_count() - 1)
    
    for data_size in [120, 240, 480]:
        # Load data for this size
        with open(f"./samples/{data_size}/data.pkl", "rb") as f:
            xi_samples_list = pickle.load(f)['train_samples']
        
        # Prepare arguments for parallel processing
        process_args = [
            (A, b, n_items, n_machines, M, xi_samples_list)
            for M in M_list
        ]
        
        # Process M values in parallel
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = [executor.submit(process_M, args) for args in process_args]
            
            # Collect results as they complete
            for future in as_completed(futures):
                res = future.result()
                M = res['M']
                result['extensive form'][data_size][M] = res['extensive_form_time']
                result['ours'][data_size][M] = res['lshape_time']
                result['cuts'][data_size][M] = res['cuts']
                
                print(f"[Data={data_size}, M={M}] EF: {res['extensive_form_time']:.2f}s | "
                      f"L-shape: {res['lshape_time']:.2f}s | Cuts: {res['cuts']:.2f}")
    
    # Convert defaultdict to regular dict for JSON serialization
    result_dict = {k: dict(v) for k, v in result.items()}
    
    # Save results to JSON
    with open(save_path, "w") as f:
        json.dump(result_dict, f, indent=4)
    print(f"Results saved to {save_path}")


   
def worker(alpha, train_samples, test_samples, A, b, c, M, n_items, n_machines):
    apub = APUB(A, b, c=c, n_items=n_items, n_machines=n_machines, model=gp.Model())
    x_optimal, _, certificate,num_optimal_cut = apub.solve_two_stage_apub(train_samples, alpha=alpha, M_bootstrap=M)
    eval_result = evaluate_oos(certificate, x_optimal, test_samples, c=c, n_items=n_items, n_machines=n_machines)
    return alpha, eval_result['mean_cost'], eval_result['reliability'], certificate, num_optimal_cut


def run_experiment(A, b, c, M, n_items, n_machines, data_size, K=30, alpha_list=None, max_workers=None, data_path=None):
    if alpha_list is None:
        alpha_list = [0.05 * i for i in range(1, 21)]
    alpha_list = np.array(alpha_list)

    if data_path is not None:
        with open(f"{data_path}", "rb") as f:
            samples = pickle.load(f)
            train_samples_list = samples['train_samples']
            test_samples = samples['test_samples']
    else:
        pg = ParametersGenerator()
        test_samples = pg.generate_parameters(sample_from_config(cfg_or_path="config.yaml", train=False))

    results = {alpha: {'costs': [], 'reliabilities': []} for alpha in alpha_list}
    
    for trial in range(K):
        if data_path is not None:
            train_samples = train_samples_list[trial]
        else:
            pg = ParametersGenerator()
            train_samples = pg.generate_parameters(sample_from_config(cfg_or_path="config.yaml", train=True))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(worker, alpha, train_samples, test_samples, A, b, c, M, n_items, n_machines): alpha
                for alpha in alpha_list
            }
            for future in as_completed(futures):
                alpha, cost, reliability, certificate, num_optimal_cut = future.result()
                results[alpha]['costs'].append(cost)
                results[alpha]['reliabilities'].append(reliability)
                print(f'epoch {trial+1} of {K}, alpha={alpha:.2f}, '
                      f'cost: {np.mean(results[alpha]["costs"]):.2f}, '
                      f'reliability: {np.mean(results[alpha]["reliabilities"]):.2f}, '
                      f'certificate: {certificate:.2f}, '
                      f'num_optimal_cut: {num_optimal_cut:.2f}')

    serializable_results = {
        str(alpha): {
            'costs': [float(c) for c in vals['costs']],
            'reliabilities': [float(r) for r in vals['reliabilities']]
            #'num_optimal_cuts': [float(n) for n in vals['num_optimal_cuts']]
        }
        for alpha, vals in results.items()
    }

    save_path = f"apub_results_ee{data_size}.json"
    with open(save_path, "w") as f:
        json.dump(serializable_results, f, indent=4)
    print(f"\n Results saved to {save_path}")
    return results


def plot_apub_results(results, mark_outliers=True):
    # try:
    #     plt.rcParams["text.usetex"] = True
    #     plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    # except Exception:
    #     print("Warning: LaTeX not found, using default text rendering.")
    #     plt.rcParams["text.usetex"] = False

    plt.rcParams["font.family"] = "serif"
    bold_times = FontProperties(family='Times New Roman', size=16, weight='bold')
    alpha_list = sorted(results.keys())
    x_vals = [1 - a for a in alpha_list] # axis x: 1 - alpha
    # 将所有 alpha 下的成本结果整理为二维数组 (K trials × len(alpha_list))
    cost_matrix = np.array([results[a]['costs'] for a in alpha_list])  # shape: (len_alpha, K)
    cost_matrix = cost_matrix.T  # shape: (K, len_alpha)

    mean_costs = np.mean(cost_matrix, axis=0)
    lower_quantile = np.quantile(cost_matrix, 0.1, axis=0)
    upper_quantile = np.quantile(cost_matrix, 0.9, axis=0)

    coverage = [np.mean(results[a]['reliabilities']) for a in alpha_list]

    best_idx = np.argmin(mean_costs)
    best_x = x_vals[best_idx]
    best_y = mean_costs[best_idx]

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(x_vals, mean_costs, 'b-', label='Average Loss')
    ax1.fill_between(x_vals, lower_quantile, upper_quantile, alpha=0.35, color='blue')
    # === mark outliers ===
    if mark_outliers:
        for j, a in enumerate(alpha_list):
            q10, q90 = lower_quantile[j], upper_quantile[j]
            outliers = [c for c in results[a]['costs'] if (c < q10 or c > q90)]
            if outliers:  
                ax1.scatter([x_vals[j]]*len(outliers), outliers, color='purple', s=20, alpha=0.6, label='_nolegend_')

    ax1.plot(best_x, best_y, marker='*', markersize=20,
         markeredgecolor='black', markeredgewidth=2,
         color='magenta', label='Lowest Mean', linestyle='None')
    # ax1.set_yticks([4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000])
    ax1.set_xticks(x_vals)

    index_of_zero = x_vals.index(0.0)
    str_nums = [f"{x:.2f}" for x in x_vals]
    str_nums[index_of_zero] = 'SAA'
    ax1.set_xticklabels(str_nums, rotation=90)

    ax1.annotate(f'{best_y:.2f}', xy=(best_x, best_y), xytext=(-30, 15),textcoords='offset points',
                color='black', fontsize=16)
    ax1.set_xlabel(r'$1-\alpha$', fontproperties=bold_times)
    ax1.set_ylabel("Out-of-sample performance", color='blue', fontproperties=bold_times)

    ax1.tick_params(axis='y', labelcolor='blue')

    ax1.grid(which='major', alpha=0.4)
    ax1.grid(which='minor', alpha=0.2)

    # Set the horizontal grid to blue
    ax1.yaxis.grid(True, color='blue', linestyle='-', linewidth=0.1)

    ax2 = ax1.twinx()
    ax2.plot(x_vals, coverage, 'r--', label='Coverage Probability')
    ax2.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax2.set_ylabel('Coverage Probability', color='red', fontproperties=bold_times)
    ax2.tick_params(axis='y', labelcolor='red')

    fig.tight_layout()
    plt.show()
    return best_x, best_y
