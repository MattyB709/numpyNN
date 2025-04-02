import numpy as np

class nn:
    def __init__(self):
        self.layers = []
        # this could be refactored to instantiate all layers to be self.layer1 = ~, but list format seemed cleaner 
        self.layers.append(LinearLayer(28*28, 256))
        self.layers.append(ReLU())
        self.layers.append(LinearLayer(256, 128))
        self.layers.append(ReLU())
        self.layers.append(LinearLayer(128, 10))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, delta):
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
        
class LinearLayer:

    def __init__(self, in_features, out_features):
        # TODO figure out weight initialization
        self.weights = np.random.randn(in_features, out_features)
        self.bias = np.random.randn(out_features)

    def forward(self, x):
        # TODO store activations for backward pass
        self.x = x
        x = np.matmul(x, self.weights)
        x = x + self.bias
        return x

    def backward(self, delta):
        # delta should be of shape (batch, out_features)
        self.weights_grad = self.x.T @ delta # (i,b) @ (b, o) -> (i, o)
        self.bias_grad = np.sum(delta, axis = 0) # (b,o) -> (o)
        d = delta @ self.weights.T # (b, o) @ (o, i) -> (b,i)
        return d 

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
        return -(prob).sum().item()    

    # gradient w.r.t cross entropy and softmax works out to be (y^ - y)
    def backward(self):
        # get a (num_classes, num_classes) identity matrix to index and get one_hot vectors
        Identity = np.eye(self.probs.shape[1]) 
        one_hot = Identity[self.y] 
        return self.probs - one_hot