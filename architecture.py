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
        return x
    
    def backward(self, delta):
        delta = self.layer3.backward(delta)
        delta = self.act2.backward(delta)
        delta = self.layer2.backward(delta)
        delta = self.act1.backward(delta)
        delta = self.layer1.backward(delta)

    # collects weights and their gradients in a list across all layers for use in optimization
    def parameters(self):
        params = []
        self.layer1.params(params)
        self.layer2.params(params)
        self.layer3.params(params)
        return params
class LinearLayer:

    def __init__(self, in_features, out_features):
        # TODO figure out weight initialization
        std = np.sqrt(2 / in_features)
        self.weights = np.random.randn(in_features, out_features) * std
        self.weights_grad = np.zeros_like(self.weights)
        self.bias = np.zeros(out_features)
        self.bias_grad = np.zeros_like(self.bias)

    def forward(self, x):
        # TODO store activations for backward pass
        self.x = x
        x = np.matmul(x, self.weights)
        x = x + self.bias
        return x

    def backward(self, delta):
        # delta should be of shape (batch, out_features)
        self.weights_grad[:] = self.x.T @ delta # (i,b) @ (b, o) -> (i, o)
        self.bias_grad[:] = np.sum(delta, axis = 0) # (b,o) -> (o)
        d = delta @ self.weights.T # (b, o) @ (o, i) -> (b,i)
        return d 
    
    # collect parameters and their gradients for use in sgd
    def params(self, params):
        params.append((self.weights, self.weights_grad))
        params.append((self.bias, self.bias_grad))

class ReLU:

    def forward(self, x):
        self.input = x
        x = np.maximum(0, x)
        return x
    
    def backward(self, delta):
        # delta is of shape (batch, features)
        mask = self.input > 0
        return delta * mask.astype(float) #(b,f) * f -> (b,f)
    
class CrossEntropy:

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        ex = np.exp(x)
        return ex / np.sum(ex, axis = 1,keepdims=True)

    # here y is just a series of indices
    def calculate_loss(self, logits, y):
        probs = self.softmax(logits) #(batch size, num_classes)
        log_probs = np.log(probs+ 1e-12)
        self.probs = probs 
        self.y = y
        # we do arange here because for every row we want the y_i logprob
        prob = log_probs[np.arange(log_probs.shape[0]), y] 
        return -prob.mean().item()    

    # gradient w.r.t cross entropy and softmax works out to be (y^ - y)
    def backward(self):
        # get a (num_classes, num_classes) identity matrix to index and get one_hot vectors
        Identity = np.eye(self.probs.shape[1]) 
        one_hot = Identity[self.y] 
        return (self.probs - one_hot) / self.probs.shape[0]