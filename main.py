from evaluation import *
from utils import load_config

if __name__ == '__main__':
    cfg_path = "config.yaml"
    # Load shared sampling hyperparameters from config
    full_cfg = load_config(cfg_path)
    rg_cfg = full_cfg.get("random_generator", full_cfg)
    I = int(rg_cfg["I"]) 
    J = int(rg_cfg["J"])
    M = int(rg_cfg["M"])
    c = list(rg_cfg["c"])
    data_size = int(rg_cfg["train_n"])
    test_size = int(rg_cfg["test_n"])
    epochs = int(rg_cfg["epochs"])
    b = np.zeros(J)
    A = np.zeros((J,I))


    #m_list = [250] + [250 * i for i in range(2, 12)]
    #evaluate_M_T_performance(A, b, m_list, n_items=n_items, n_machines=n_machines)

    results = run_experiment(A, b, c=c, M=M, n_items=I, n_machines=J, data_size=data_size, test_size=test_size, K=epochs)
    plot_apub_results(results)
