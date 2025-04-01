import numpy as np

class nn:
    def __init__(self):
        self.layer1 = LinearLayer(28*28, 256)
        self.act1 = ReLU()
        self.layer2 = LinearLayer(256, 128)
        self.act2 = ReLU()
        self.layer3 = LinearLayer(128, 10)
    
    def forward(self, x):
        x = self.layer1.forward(x)
        x = self.act1.forward(x)
        x = self.layer2.forward(x)
        x = self.act2.forward(x)
        x = self.layer3.forward(x)

        
class LinearLayer:

    def __init__(self, in_features, out_features):
        # TODO figure out weight initialization
        self.weights = np.zeros((in_features, out_features), dtype = np.float16)
        self.bias = np.zeros(out_features)

    def forward(self, x):
        # TODO store activations for backward pass
        x = np.matmul(self.weights, x)
        x = x + self.bias

class ReLU:

    def forward(x):
        return np.maximum(0, x)
    

        
x = np.random.randn(5,6)
print(x)
print(ReLU(x))
