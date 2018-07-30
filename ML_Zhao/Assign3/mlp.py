#please put the data file in the same path

import numpy as np
import random

def split(n):
    #assign the id of samples into different lists
    
    all_num = range(n)
    tr_num = random.sample(all_num, round(n*0.50))
    rest_num = list(set(all_num).difference(set(tr_num)))
    va_num = random.sample(rest_num, round(n*0.25))
    test_num = list(set(rest_num).difference(set(va_num)))
    
    return tr_num, va_num, test_num

def Norm(data):
    #normalization
    
    mins = data.min(0)
    maxs = data.max(0)
    diffs = maxs - mins
    normData = np.zeros(np.shape(data))
    num_row = data.shape[0]
    normData = data - np.tile(mins,(num_row,1))
    normData = normData/np.tile(diffs,(num_row,1))
    return normData

class mlp():
    #mlp: a class for multi-layer perceptron
    
    def __init__(self, inputs, targets, _beta = 1):

        self.nIn = np.shape(inputs)[1]
        self.nHid = 30
        self.nOut = 3
        self.beta = _beta
        self.nExam = np.shape(inputs)[0]
        X = np.c_[inputs, np.ones((self.nExam, 1))]
        Y = np.zeros((self.nExam, 3))
        for i in range(self.nExam):
            Y[i, int(targets[i]) - 1] = 1
        self.splits = split(self.nExam)
        self.X_train = X[self.splits[0], :]
        self.Y_train = Y[self.splits[0], :]
        self.X_valid = X[self.splits[1], :]
        self.Y_valid = Y[self.splits[1], :]
        self.X_test = X[self.splits[2], :]
        self.targets_test = targets[self.splits[2]]
        
        np.random.seed(171803)
        self.weights1 = (np.random.rand(self.nIn+1,self.nHid)-0.5)/np.sqrt(self.nIn)
        self.weights2 = (np.random.rand(self.nHid+1,self.nOut)-0.5)/np.sqrt(self.nHid)
        
    def move_forw(self, _X):
        self.hidden = np.dot(_X, self.weights1)
        self.hidden = 1/(1 + np.exp(-self.beta * self.hidden))
        self.hidden = np.c_[self.hidden, np.ones((np.shape(_X)[0], 1))]
        self.outputs = np.dot(self.hidden, self.weights2)
        self.outputs = 1/(1 + np.exp(-self.beta * self.outputs))
        
    def train(self, iter_time, learning_rate):
        
        for i in range(iter_time):
            #print(i + 1)
            self.move_forw(self.X_train)
            deltao = self.beta * (self.outputs - self.Y_train) * self.outputs * (1 - self.outputs)
            deltah = self.beta * self.hidden * (1.0 - self.hidden) * (np.dot(deltao, np.transpose(self.weights2)))
            update_w1 = learning_rate * np.dot(np.transpose(self.X_train), deltah[:, :-1])
            update_w2 = learning_rate * np.dot(np.transpose(self.hidden), deltao)
            self.weights1 = self.weights1 - update_w1
            self.weights2 = self.weights2 - update_w2

            self.move_forw(self.X_valid)
            err = sum(sum((self.outputs - self.Y_valid)**2))
            if err < 0.001:
                break

    def predict(self):
        self.move_forw(self.X_test)
        pre_targets = np.zeros(np.shape(self.X_test)[0])
        for i in range(np.shape(self.X_test)[0]):
            pre_targets[i] = np.where(self.outputs[i, :] == self.outputs[i, :].max())[0] + 1
        print('outputs: ')
        print(pre_targets)
        print('targets: ')
        print(self.targets_test)
        print('corrected-classified/all: ')
        nCorr = sum(pre_targets == self.targets_test)
        print(str(nCorr) + '/' + str(np.shape(self.X_test)[0]))
        print('accuracy: ')
        print(nCorr/np.shape(self.splits[2]))

data = np.loadtxt(open('./wine_data.csv', 'rb'), delimiter=',', skiprows=0)
inputs = data[:, 1:]
inputs = Norm(inputs)
targets = data[:, 0]
my_mlp = mlp(inputs, targets, 1)
my_mlp.train(2000, 0.01)
my_mlp.predict()
print('NO. of nodes in hidden layer:')
print(my_mlp.nHid)
