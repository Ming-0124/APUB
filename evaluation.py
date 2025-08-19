import numpy as np
import gurobipy as gp
from gurobipy import GRB
from apub import APUB
import time
from params_generator import ParametersGenerator
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.font_manager import FontProperties
from utils import sample_from_config


def evaluate_oos(certificate, x_optimal, test_samples, c, n_items, n_machines):
    """在测试集上评估解的性能"""
    costs = []
    reliability = []
    N = len(test_samples['h'])

    for m in range(N):
        W_test = test_samples['W'][m]
        h_test = test_samples['h'][m]
        T_test = test_samples['T']
        q_test = test_samples['q'][m]

        # 计算第二阶段成本
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
            costs.append(np.inf)  # 标记不可行解

    reliability.append(certificate >= np.mean(costs))
    return {
        'mean_cost': np.mean(costs),
        'reliability': np.mean(reliability)
    }


def evaluate_M_T_performance(A, b, M_list, n_items, n_machines):
    time_list = defaultdict(lambda: defaultdict(dict))
    pg = ParametersGenerator()
    for data_size in [120, 240, 480]:
        for M in M_list:
            tt1 = []
            tt2 = []
            for i in range (10):
                xi_samples = pg.generate_parameters(sample_from_config(cfg_or_path="config.yaml", train=True))
                model = gp.Model('Master Problem')
                apub = APUB(A, b, n_items=n_items, n_machines=n_machines, model=model)
                start1 = time.perf_counter()
                apub.extensive_form(xi_samples, alpha=0.1, M_bootstrap=M)
                end1 = time.perf_counter()
                #print(f'extensive form: {end1 - start1}s')
                start2 = time.perf_counter()
                apub.solve_two_stage_apub(
                    xi_samples,
                    alpha=0.1,
                    M_bootstrap=M,
                )
                end2 = time.perf_counter()
                #print(f'ours: {end2 - start2}s')
                tt1.append(end1 - start1)
                tt2.append(end2 - start2)
            time_list['extensive form'][data_size][M] = np.mean(tt1)
            time_list['ours'][data_size][M] = np.mean(tt2)
            print(f"method: extensive form, data size: {data_size}, M: {M}, time: {time_list['extensive form'][data_size][M]}")
            print(f"method: l-shaped, data size: {data_size}, M: {M}, time: {time_list['ours'][data_size][M]}")

    plt.figure(figsize=(8, 6))
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    bold_times = FontProperties(family='Times New Roman', size=16, weight='bold')

    color_map = {
        60: 'blue',
        120: 'red',
        240: 'black',
        480: 'green'
        # 可以加更多 data_size
    }

    # 定义线型：每个 method 一种线型
    linestyle_map = {
        'extensive form': '-',  # 实线
        'ours': '--',  # 虚线
    }

    for method in time_list:
        for data_size in time_list[method]:
            m_dict = time_list[method][data_size]
            m_sizes = sorted(m_dict.keys())
            times = [m_dict[m] for m in m_sizes]
            label = f"{method}, N={data_size}"
            plt.plot(
                m_sizes,
                times,
                label=label,
                color=color_map.get(data_size, 'gray'),
                linestyle=linestyle_map.get(method, '-')
            )

    # m_dict = time_list['extensive form'][480]
    # m_sizes = sorted(m_dict.keys())
    # times = [m_dict[m] for m in m_sizes]
    # label = f"{'extensive form'}, N={480}"
    # plt.plot(
    #     m_sizes,
    #     times,
    #     label=label,
    #     color=color_map.get(data_size, 'gray'),
    #     linestyle=linestyle_map.get('extensive form', '-')
    # )

    plt.xlabel('bootstrap size', fontproperties=bold_times)
    plt.ylabel('Time (s)', fontproperties=bold_times)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_experiment(A, b, c, M, n_items, n_machines, data_size, test_size=1000, K=30, alpha_list=None):
    if alpha_list is None:
        alpha_list = [0.05 * i for i in range(1, 21)]
    alpha_list = np.array(alpha_list)

    results = {alpha: {'costs': [], 'reliabilities': []} for alpha in alpha_list}
    pg = ParametersGenerator()
    
    for trial in range(K):
        train_samples = pg.generate_parameters(sample_from_config(cfg_or_path = "config.yaml", train=True))
        test_samples = pg.generate_parameters(sample_from_config(cfg_or_path = "config.yaml", train=False))

        for alpha in alpha_list:
            apub = APUB(A, b, c=c, n_items=n_items, n_machines=n_machines, model=gp.Model())
            x_optimal, _, certificate = apub.solve_two_stage_apub(train_samples, alpha=alpha, M_bootstrap=M)
            # x_optimal, certificate = apub.extensive_form(train_samples, alpha=alpha, M_bootstrap=M)
            eval_result = evaluate_oos(certificate, x_optimal, test_samples, c=c, n_items=n_items, n_machines=n_machines)
            results[alpha]['costs'].append(eval_result['mean_cost'])
            results[alpha]['reliabilities'].append(eval_result['reliability'])
            print(f'epoch {trial+1} of {K}, alpha={alpha:.2f}, '
                  f'cost: {np.mean(results[alpha]["costs"]):.2f}, reliability: {np.mean(results[alpha]["reliabilities"]):.2f}, certificate: {certificate:.2f}')
    
    serializable_results = {
        str(alpha): {
            'costs': [float(c) for c in vals['costs']],
            'reliabilities': [float(r) for r in vals['reliabilities']]
        }
        for alpha, vals in results.items()}

    save_path = f"apub_results_ee{data_size}.json"
    with open(save_path, "w") as f:
        json.dump(serializable_results, f, indent=4)
    print(f"\n Results saved to {save_path}")
    return results


def plot_apub_results(results, mark_outliers=True):
    try:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    except Exception:
        print("Warning: LaTeX not found, using default text rendering.")
        plt.rcParams["text.usetex"] = False

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
