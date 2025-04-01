import numpy as np

class nn:
    def __init__(self):
        self.layer1 = LinearLayer(28*28, 256)
        self.layer2 = LinearLayer(256, 128)
        self.layer3 = LinearLayer(128, 10)
        self.act = ReLU()
        

class LinearLayer:

    def __init__(self, in_features, out_features):
        # TODO figure out weight initialization
        self.weights = np.zeros((in_features, out_features), dtype = np.float16)
        self.bias = np.zeros(out_features)

    def forward(self, x):
        x = np.matmul(self.weights, x)
        x = x + self.bias

class ReLU:

    def forward(x):
        return np.maximum(0, x)
    
    
        
x = np.random.randn(5,6)
print(x)
print(ReLU(x))
