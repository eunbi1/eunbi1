import numpy as np
import random
import math
import networkx as nx
import os
import matplotlib.pyplot as plt
import copy
import pandas as pd
import Softmax
import Matrix_Factorization
import itertools
import time
from tqdm import tqdm

node_label=np.loadtxt('karate_label.txt')
with open('karate_club.adjlist') as f:
    data=f.readlines()
del data[:3]

E=[]
for i in range(len(data)):
    ith=data[i].split()
    E.extend([(int(ith[0]),int(j)) for j in ith[1:]])

G=nx.Graph()
G.add_nodes_from(range(len(data)))
#주의할 점은 edge를 넣어버리면 node의 순서가 edge에서 나타나는 노드 순으로 저장이 되므로 먼저 node를 추가하고 edge를 넣어야 순서가 바뀌지 않음
G.add_edges_from(E)
d=dict(G.degree)
label=node_label[:,1]

def logit(x):
    return np.log(x/(1-x))
def sigmoid(x):
    return 1/(1+np.exp(-x))

#phi : n개의 d-dimensional vector를 row vector로 갖고 있는 matrix를 input으로 받음
class BinaryClassifier: #embedding_method
    def __init__(self, embedding_dimension=2):
        self.d=embedding_dimension
        self.w=np.random.randn(self.d)

    def sampling(self, G=G, method_name=Softmax, portion_nodes=80, label=label): # G:network, update할 date 수집.
        self.N=portion_nodes
        a=method_name.Network_Embedding(G)
        if method_name != Matrix_Factorization:
            a.DeepWalk()
        else: a.Matrix_factorization()
        phi=a.phi
        I=random.sample(G.nodes(),math.floor(len(G)*self.N/100))
        J=list(G.nodes())
        for i in I:
            J.remove(i)
        self.training_data=[(phi[i],label[i]) for i in I]
        self.test_data=[(phi[j],label[j]) for j in J]
        self.data=[(phi[i],label[i]) for i in G.nodes()]
        self.method_name=str(method_name).split("'")[1]
        self.embedding=phi
        self.true_label=label

    def Logistic_regression(self, epochs=1000, eta=0.02):
        #training set은 input data x와 outputdata t의 tuple (x,t)의 list 형태로 받게 될거임.
        self.eta=eta
        self.epochs=epochs

        for e in range(1,self.epochs+1):
            self.e=e
            random.shuffle(self.training_data)
            #각 training data를 SGD로 업데이트함.
            for data in self.training_data:
                v=data[0]
                t=data[1]
                y=sigmoid(np.dot(self.w,v.T))
                self.w+=-self.eta*(y-t)*v

        self.label=np.array([sigmoid(np.dot(self.w,v.T)) for v in self.embedding])
        self.test_error=self.error_function(self.test_data)
        self.training_error=self.error_function(self.training_data)
        
        print("training error for this model is {}".format(self.training_error))            
        print("test error for this model is {} \n".format(self.test_error))
        
        
        #decision boundary ploting
        X=np.arange(-2,2.1,0.1)
        Y=self.boundary(X,self.w[0],self.w[1])
        plt.plot(X,Y,label='epoch {}'.format(e), color='black')
        
        
        #decision boundary의 normal vector 그리기
        plt.quiver(0.3, self.boundary(0.3,self.w[0],self.w[1]), self.w[0], self.w[1])
        plt.text(0.3, self.boundary(0.3,self.w[0],self.w[1]), (round(self.w[0],1),round(self.w[1],1)))
        plt.axis([-2, 2, -2, 2])
        plt.legend()
        
        
        #node vector embedding ploting
        for i, z in enumerate(self.data):
            x=z[0][0]
            y=z[0][1]
            l=z[1]
            if self.output(z[0])< 0.5 :
                color='b'
            else: 
                color='r'
            plt.scatter(x,y, alpha=0.8, color=color)
            plt.text(x,y,i)
            if l==0:
                label_color='b'
            else:
                label_color='r'
            plt.text(x+0.1,y,int(l),color=label_color, alpha=0.8)

        tm = time.localtime()
        time_='{}.{}.{} {}시 {}분'.format(tm.tm_year,tm.tm_mon,tm.tm_mday,tm.tm_hour,tm.tm_min)


        # 각 class 별 대표노드 계산하기
        class1_x=[]
        class1_y=[]
        class0_x=[]
        class0_y=[]
        for d in self.data: # data의 원소=[(x,y),label]
            if d[1]==0:
                class0_x.append(d[0][0])
                class0_y.append(d[0][1])
            else:
                class1_x.append(d[0][0])
                class1_y.append(d[0][1])
        class1_x=np.array(class1_x)
        class1_y=np.array(class1_y)
        class0_x=np.array(class0_x)
        class0_y=np.array(class0_y)
        z1_x=np.mean(class1_x)
        z1_y=np.mean(class1_y)
        z0_x=np.mean(class0_x)
        z0_y=np.mean(class0_y)
        
        self.class1_representive_vector=np.array([z1_x,z1_y])
        self.class0_representive_vector=np.array([z0_x,z0_y])
        self.representive_nodes=np.array([self.class0_representive_vector,self.class1_representive_vector])
        #대표노드 ploing
        plt.scatter(z1_x,z1_y,color='red', s=100)
        plt.text(z1_x,z1_y,'class1', color='red', fontsize=15)
        plt.scatter(z0_x,z0_y, color='blue', s=100)
        plt.text(z0_x,z0_y,'class0', color='blue',fontsize=15)
        plt.savefig('decision boundary{}.png'.format(time_), dpi=500)

    def output(self, x):
        return sigmoid(np.dot(self.w,x.T))
    def boundary(self,x,w1,w2):
        return logit(0.5)/w2-(w1/w2)*x


    def error_function(self, datas): #list of tuple(np.array[x1,x2,..xn],t) #평균 뺐음. 
        error=0
        for ith in datas:
            v=ith[0]
            t=ith[1]
            y=sigmoid(np.dot(self.w,v.T))
            error += 1/2*np.inner(y-t,y-t)
        return error 

B=BinaryClassifier()
B.sampling()
B.Logistic_regression()