# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:45:34 2021

@author: 10437
"""
import numpy as np
import random
import struct
import preprocess as pp
import matplotlib.pyplot as plt

class Network():
    
    def __init__(self,sizes):
        self.num_layers=len(sizes)
        self.sizes=sizes
        self.biases=[np.random.randn(y,1) for y in sizes[1:]]
        self.weights=[np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
        
    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a=self.sigmoid(np.dot(w,a)+b)
        return a
        
    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data):
        if(test_data):
            n_test=len(test_data)
        n=len(training_data)
        plt.ion()
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batchs=[training_data[k:k+mini_batch_size]  for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print('Epoch %s:%s/%s'%(j,self.evaluate(test_data),n_test))
                #plt.grid(linestyle='-.')
                plt.title("accuracy=%s"%(self.evaluate(test_data)/n_test))
                plt.xlabel("epoch")
                plt.ylabel("accuracy")
                plt.xlim(0,epochs)
                plt.ylim(0,1) 
                plt.scatter(j, self.evaluate(test_data)/n_test)
                plt.pause(0.5)
            else:
                print('Epoch %s complete'%j)
        plt.ioff()
        plt.show()
                
    def update_mini_batch(self,mini_batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)  #反向传播
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self, x, y):
       nabla_b = [np.zeros(b.shape) for b in self.biases]
       nabla_w = [np.zeros(w.shape) for w in self.weights]
       activation = x
       activations = [x] 
       zs = []
       for b, w in zip(self.biases, self.weights):
           z = np.dot(w, activation)+b
           zs.append(z)
           activation = self.sigmoid(z)
           activations.append(activation)
       delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
       nabla_b[-1] = delta
       nabla_w[-1] = np.dot(delta, activations[-2].transpose())
       for l in range(2, self.num_layers):
           z = zs[-l]
           sp = self.sigmoid_prime(z)
           delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
           nabla_b[-l] = delta
           nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
       return (nabla_b, nabla_w)
   
    #返回神经网络为其输出正确结果的测试输入的数量。请注意，神经网络的输出被假定为最后一层中激活程度最高的任何神经元的索引
    def evaluate(self,test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]#最大值的索引
        #print(test_results)
        
        return sum(int(x == y) for (x, y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    
    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))
    
    def sigmoid_prime(self,z):
        return (self.sigmoid(z))*(1-self.sigmoid(z))
    
    
if __name__=='__main__':
    
    def vectorized_result(j):
        e = np.zeros((10, 1))
        e[int(j)] = 1.0
        return e

    train_images_idx3_ubyte_file = 'D:/git_lab/game/minst/MNIST_Play/1_MNIST/1_MNIST/train-images-idx3-ubyte'
    train_labels_idx1_ubyte_file = 'D:/git_lab/game/minst/MNIST_Play/1_MNIST/1_MNIST/train-labels-idx1-ubyte'
    test_images_idx3_ubyte_file = 'D:/git_lab/game/minst/MNIST_Play/1_MNIST/1_MNIST/t10k-images-idx3-ubyte'
    test_labels_idx1_ubyte_file = 'D:/git_lab/game/minst/MNIST_Play/1_MNIST/1_MNIST/t10k-labels-idx1-ubyte'
    
    
    
    train_images = pp.load_train_images(train_images_idx3_ubyte_file)
    train_labels = [vectorized_result(i) for i in pp.load_train_labels(train_labels_idx1_ubyte_file)]
    
    test_images = pp.load_test_images(test_images_idx3_ubyte_file)
    test_labels = pp.load_test_labels(test_labels_idx1_ubyte_file)
    
    
    
    training_data=list(zip(train_images,train_labels))
    test_data=list(zip(test_images,test_labels))
    
    
    net = Network([784, 30, 10])    
    net.SGD(training_data, 50, 500, 3, test_data=test_data)
    #training_data,epochs,mini_batch_size,eta
    #result 91.71%