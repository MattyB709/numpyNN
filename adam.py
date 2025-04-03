import numpy as np
class Adam:
    def __init__(self, params, lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps=1e-8):
        self.params = params # here params is a list of pairs of weights and their gradients
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 1 # for bias corrections
        self.v_list = [] # normalization values 
        self.m_list = [] # momentum values
        for param in params:
            self.v_list.append(np.zeros_like(param[0]))
            self.m_list.append(np.zeros_like(param[0]))
        
    def step(self):
        for i in range(len(self.params)):
            params, grads = self.params[i]
            # calculate new v values
            v = self.v_list[i]
            v = self.beta2 * v + (1- self.beta2) * np.pow(grads, 2)
            self.v_list[i] = v
            v_hat = v / (1 - self.beta2 ** self.t) # do this for bias correction, because we start at 0

            # calculate new m values
            m = self.m_list[i]
            m = self.beta1 * m + (1 - self.beta1) * grads
            self.m_list[i] = m
            m_hat = m / (1 - self.beta1 ** self.t)
            update = self.lr * m_hat / (np.sqrt(v_hat)+ self.eps)
            params -= update
        self.t += 1




