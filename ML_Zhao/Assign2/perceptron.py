import numpy as np

class perceptron:
    
    iter_time = 60
    learn_rate = 0.25
    
    def __init__(self, inputs, targets, add_on):
        self.nIn = np.shape(inputs)[1]
        self.nOut = np.shape(targets)[1]
        self.add_on = add_on
        self.X = np.c_[inputs, add_on]
        self.Y = targets
        np.random.seed(171803)
        self.weights = np.random.rand(self.nIn+1, self.nOut)*0.1-0.05
        
    def train(self):
        iter_in_prac = 0
        for i in range(self.iter_time):
            if np.sum(np.round(np.dot(self.X, self.weights)) == self.Y) == 4:
                break
            diff = diff = np.dot(self.X, self.weights) - self.Y
            gra = np.dot(np.transpose(self.X), diff)
            self.weights = self.weights - self.learn_rate * gra
            iter_in_prac = iter_in_prac + 1
        return iter_in_prac

    def predict(self, inputs):
        new_X = np.c_[inputs, self.add_on]
        pre_Y = np.dot(new_X, self.weights) > 0.5
        return pre_Y
