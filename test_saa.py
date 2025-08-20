import gurobipy as gp
from apub import APUB
from saa import SAA
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from evaluation import evaluate_oos
from utils import load_config, sample_from_config
from params_generator import ParametersGenerator

if __name__ == "__main__":
    cfg_path = "config.yaml"
    
    # Load shared sampling hyperparameters from config
    full_cfg = load_config(cfg_path)
    rg_cfg = full_cfg.get("random_generator", full_cfg)
    J = int(rg_cfg["J"]) 
    I = int(rg_cfg["I"]) 
    c = list(rg_cfg["c"]) 
    M = int(rg_cfg["M"]) 
    n = int(rg_cfg["train_n"])
    epochs = int(rg_cfg["epochs"])
    
    b = np.zeros(J)
    A = np.zeros((J,I))
    pg = ParametersGenerator()
    obj_values = []
    
    saa = SAA(c=c, n_items=I, n_machines=J, model=gp.Model('SAA'))
    
    apub_costs = []
    apub_reliabilities = []

    train_samples_list = []
    test_samples_list = []

    # SAA results
    for r in range(epochs):
        train_samples = pg.generate_parameters(samples = sample_from_config(cfg_path, train=True))
        test_samples = pg.generate_parameters(samples=sample_from_config(cfg_path, train=False))
        train_samples_list.append(train_samples)
        test_samples_list.append(test_samples)

        x_opt, obj_val = saa.solve_nf(train_samples)
        val = saa.saa_oos(x_opt, test_samples)
        print(f'SAA trial {r+1}/{epochs}: cost={val:.2f}')
        obj_values.append(val)

        apub_solver = APUB(A, b, c=c, n_items=I, n_machines=J, model=gp.Model('APUB'))
        
        x_optimal, eta_optimal, certificate = apub_solver.solve_two_stage_apub(train_samples, alpha=1, M_bootstrap=M)
        
        # Evaluate out-of-sample performance
        eval_result = evaluate_oos(certificate, x_optimal, test_samples, c=c,n_items=I, n_machines=J)
        
        apub_costs.append(eval_result['mean_cost'])
        apub_reliabilities.append(eval_result['reliability'])
        
        print(f'APUB trial {r+1}/{epochs}: cost={eval_result["mean_cost"]:.2f}, reliability={eval_result["reliability"]:.3f}')

    os.makedirs("samples", exist_ok=True)
    with open(f"samples/train_samples.pkl", "wb") as f:
        pickle.dump(train_samples_list, f)
    with open(f"samples/test_samples.pkl", "wb") as f:
        pickle.dump(test_samples_list, f)

    # Create side-by-side boxplot with reliability information
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Side-by-side boxplots
    box_data = [obj_values, apub_costs]
    box_labels = [f'SAA', f'APUB (α={1})']
    box_colors = ['lightblue', 'lightgreen']
    
    bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True,
                     medianprops=dict(color='red', linewidth=2))
    
    # Set colors for each box individually
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
    ax1.set_ylabel("Cost Value")
    ax1.set_title("SAA vs APUB Performance Comparison")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Add reliability information as text on the plot
    saa_reliability = "N/A"  # SAA doesn't have reliability concept
    apub_reliability = f"{np.mean(apub_reliabilities):.3f}"
    
    ax1.text(0.02, 0.98, f'SAA Reliability: {saa_reliability}', 
              transform=ax1.transAxes, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax1.text(0.02, 0.92, f'APUB Reliability: {apub_reliability}', 
              transform=ax1.transAxes, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Reliability distribution plot
    ax2.hist(apub_reliabilities, bins=15, alpha=0.7, color='lightgreen', edgecolor='green')
    ax2.axvline(np.mean(apub_reliabilities), color='red', linestyle='--', linewidth=2, 
                 label=f'Mean: {np.mean(apub_reliabilities):.3f}')
    ax2.set_xlabel("Reliability")
    ax2.set_ylabel("Frequency")
    ax2.set_title("APUB Coverage Probability Distribution (α=0.3)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nSAA Summary:")
    print(f"Mean cost: {np.mean(obj_values):.2f}")
    print(f"Std cost: {np.std(obj_values):.2f}")
    
    print(f"\nAPUB Summary (α={1}):")
    print(f"Mean out-of-sample cost: {np.mean(apub_costs):.2f}")
    print(f"Std out-of-sample cost: {np.std(apub_costs):.2f}")
    print(f"Mean reliability: {np.mean(apub_reliabilities):.3f}")
    print(f"Std reliability: {np.std(apub_reliabilities):.3f}")

