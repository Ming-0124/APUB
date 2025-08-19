import numpy as np

class ParametersGenerator:
    def __init__(self):
        return

    def build_w(self, w: np.ndarray) -> np.ndarray:
        n, j = w.shape
        dtype = w.dtype
        result = np.zeros((n, j+1, 3 * j))

        I = np.eye(j)  
        bottom = np.concatenate([
            np.zeros(j, dtype=dtype),
            np.ones(j,  dtype=dtype),
            np.zeros(j, dtype=dtype)
        ])
    
        for i in range(n):
            diag_w = np.diag(w[i])       
            diag_1 = I                   
            diag_neg1 = -I               
            top = np.hstack([diag_w, diag_1, diag_neg1])
            result[i] = np.vstack([top, bottom])
        return result
    

    def build_q(self, q):
        n_rows, j = q.shape  
        extension = np.full((n_rows, 2 * j), 0)  
        extended_q = np.concatenate([q, extension], axis=1) 
        return extended_q


    def build_h(self, h, J):
        n = len(h)
        extended = np.zeros((n, J+1))  
        extended[:, -1] = h  
        return extended
    

    # def build_t(self):
    #     T_row1 = [-10, -6, -8, -4]
    #     T_row2 = [-6, -2, -3, -2]
    #     T_row3 = [0, 0, 0, 0]
    #     return np.vstack([T_row1, T_row2, T_row3])
    

    def generate_parameters(self, samples):
        J = len(samples["T"])-1
        
        h = self.build_h(samples['h'], J)
        q = self.build_q(samples['q'])
        W = self.build_w(samples['W'])
        T = samples['T']
        
        return dict(q=q,T=T,W=W,h=h)


if __name__ == "__main__":
    from utils import sample_from_config
    config_path = "config.yaml"
    params = sample_from_config(config_path, train=True)    
    pg = ParametersGenerator()
    samples = pg.generate_parameters(params)
    print(len(samples["T"]))

    