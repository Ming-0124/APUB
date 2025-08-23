from evaluation import *
from utils import load_config

if __name__ == '__main__':
    cfg_path = "config.yaml"
    np.random.seed(42)  # For reproducibility
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


    m_list = [500] + [500 * i for i in range(1, 6)]
    #evaluate_M_T_performance(A, b, m_list, n_items=n_items, n_machines=n_machines)

    results = run_experiment(A, b, c=c, M=M, n_items=I, n_machines=J, data_size=data_size, K=epochs, data_path="./samples/240/data.pkl")
    plot_apub_results(results)
