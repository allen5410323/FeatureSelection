# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics
from sklearn.metrics import make_scorer
from sklearn.svm import LinearSVR
import MyPlot as p
import joblib

warnings.filterwarnings("ignore")
matplotlib.use("Agg")


class FeatureSelection:
    def __init__(self, trainingPara):
        
        # training parameter setting
        self.agent = trainingPara["agent"]
        self.ndim = trainingPara["ndim"]
        self._iter_max = trainingPara["iter_max"]
        self.pos = trainingPara["pos"]
        self.scores = np.zeros([self.agent, 1])
        self.scores_gBest = float('Inf')
        self.isTrained=False
        
        
        # boundary setting for X & C
        self.boundarySetting()
        X_x = np.random.uniform(low=self.X_min, high=self.X_max, size=[self.agent, self.ndim])        
        X_C = np.random.uniform(low=self.C_min, high=self.C_max, size=[self.agent, self.pos])
        self.X = np.concatenate([X_C, X_x],axis=1)
        
                

    def Optimize(self, Xtrain, Ytrain):
        self.Xtrain=Xtrain
        self.Ytrain=Ytrain
        
        self.initialCheckBest()

        # iteration of modeling
        self._iter = 0
        while(self._iter < self._iter_max):
            self.whaleOptizeAlgorithm()
            self.boundaryHandle()
            self.featureEncoding()
            self.crossValScore()
            self.checkBest()
            self.pintResult()
            self._iter = self._iter+1
        
        # final training
        self.regressor=[]
        p.MyPlot.NewFig('prediction result')
        for pos in range(self.pos):   
            regressor=LinearSVR(epsilon=5, C=self.X_gBest[0+pos], fit_intercept=False)
            regressor.fit(self.Xtrain[pos][:, self.X_feature_best], self.Ytrain[pos])
            self.regressor.append(regressor)
            Ypred = regressor.predict(self.Xtrain[pos][:, self.X_feature_best])      
            p.MyPlot.Plot(Ytrain[pos], Ypred)
        p.MyPlot.GetResult()
        self.isTrained=True
        
        
    def OutputPara(self):
        if self.isTrained:
            return self.X_gBest[:self.pos], self.X_feature_best
        else :
            print("Please Optimize First")
        
    def OutputModel(self):
        if self.isTrained:
            return self.regressor
        else :
            print("Please Optimize First")
        
    def Predict(self, Xtest, Ytest):
        if self.isTrained:
            Ypred=[]
            p.MyPlot.NewFig('training result')
            for pos in range(self.pos):
                pred = self.regressor[pos].predict(Xtest[pos][:, self.X_feature_best])               
                Ypred.append(pred)
                p.MyPlot.Plot(Ytest[pos], pred)
                print('error = '+str(metrics.max_error(pred,Ytest[pos])))
            p.MyPlot.GetResult()
             
            return Ypred
        else:
            print("Please Optimize First")



    def lossfunction(self,y,ypred):
        # weighting for choosing features
        w2 = 10/self.pos
        # MAE & MAX 
        s1 = metrics.mean_absolute_error(y, ypred)
        s2 = max(np.abs(y-ypred.reshape(-1,1)))
        score = s1 + s2
        # loss = MAE + MAX(error) + cost
        cost = np.count_nonzero(self.feature_input == True)/self.Xtrain[0].shape[1]
        return score+w2*cost
        

    def featureEncoding(self):
        # convert to binary type by sigmoid function
        X_temp = self.X[:,self.pos:].copy()
        rand = np.random.uniform(0, 1)
        encoding = 1/(1+np.exp(-X_temp))
        X_temp = 1*(encoding <= rand)

        for i in range(X_temp.shape[0]):
            while(np.sum(X_temp[i,:]) == 0):
                X_temp[i,:] = np.random.randint(2, size=[1, self.ndim])

        self.X_feature = (X_temp.copy()==1)


    def crossValScore(self):
        # sum of all position themal error
        for i in range(self.X_feature.shape[0]):
            self.scores[i]=0
            for pos in range(0,self.pos,1):   
                self.feature_input = self.X_feature[i] 
                my_loss = make_scorer(self.lossfunction, greater_is_better=False)
                # k-fold
                error = cross_val_score(LinearSVR(epsilon=5, C=self.X[i,pos], fit_intercept=False), self.Xtrain[pos][:, self.X_feature[i]], self.Ytrain[pos], cv=4, scoring=my_loss)                
                self.scores[i]=self.scores[i]+np.abs(error).mean()


    # for self-defining kernel
    def build_k_gaussian(self, sigma):
        def k_gaussian(_x1, _x2):
            diff = _x1 - 1
            normsq = np.square(diff)
            _x1 = np.exp(-normsq*sigma)
            _x2 =_x1.copy().T
            return np.dot(_x1, _x2)

        return k_gaussian


    def whaleOptizeAlgorithm(self):
        A = 2*(self._iter_max-self._iter/self._iter_max)
        for i in range(self.agent):
            p = np.random.uniform(0.4,1)
            r = np.random.uniform()
            C = 2*r
            l = np.random.uniform(-1,1)
            b = 0.1#np.random.randint(low=0, high=500)

            if p >= 0.7:
                # Encircling prey
                if np.abs(A)<1: 
                    D = np.abs(r*self.X_gBest - self.X[i, :])
                    self.X[i, :] = self.X_gBest-A*D
                    
                # Bubble-net attacking method
                else: 
                    rand = np.random.randint(0,i+1)             
                    Xrand=self.X[rand,:]
                    D = np.abs(C*Xrand - self.X[i, :])                           
                    self.X[i,:] = Xrand-A*D 
                    
            # Searching for prey
            else: 
                D = np.abs(C*self.X_gBest - self.X[i, :])
                self.X[i, :] = self.X_gBest-A*D



    def boundaryHandle(self):
        # if(X or C is out of boundary) give a random value which in the bound
        mask = (self.X[:,:self.pos] > self.C_max)|(self.X[:,:self.pos] < self.C_min)
        self.X[:,0:self.pos][mask] = np.random.uniform(self.C_min, self.C_max, mask[mask == True].shape[0])
      
        mask = (self.X[:,self.pos:] > self.X_max)|(self.X[:,self.pos:] < self.X_min)
        self.X[:,self.pos:][mask] = np.random.uniform(self.X_min, self.X_max, mask[mask == True].shape[0])



    def checkBest(self):
        if np.nanmin(self.scores) < self.scores_gBest:
            indexBest = self.scores.argmin()          
            self.scores_gBest = self.scores[indexBest].copy()
            self.X_gBest = self.X.copy()[indexBest, :]
            self.X_feature_best = self.X_feature[indexBest, :]


    def initialCheckBest(self):
        self.featureEncoding()
        self.crossValScore()
        self.checkBest()


    def pintResult(self):
        print("-------------", self._iter, "-------------")
        # global minimum
        print("gBest: ", self.scores_gBest, "/", " - Features: ", np.where(self.X_feature_best))
        print("C: {0}".format(self.X_gBest[:self.pos]))
        
        # local minimum
        indexBest = self.scores.argmin()        
        print("pBest: ", np.nanmin(self.scores), "/", " - Features: ", np.where(self.X_feature[self.scores.argmin()]))   
        print("C: {0}".format(self.X[indexBest,:self.pos]))


    def boundarySetting(self):
        # bound for X
        self.X_min=-5
        self.X_max=5
        bound_X=np.ones([2,self.ndim])
        bound_X[0,:]=bound_X[0,:]*self.X_min
        bound_X[1,:]=bound_X[1,:]*self.X_max
        
        # bound for C
        self.C_min=1
        self.C_max=10
        bound_C=np.ones([2,self.pos])
        bound_C[0,:]=bound_C[0,:]*self.C_min
        bound_C[1,:]=bound_C[1,:]*self.C_max
        
        
        
        self.bounary_X=np.concatenate([bound_C,bound_X],axis=1)



        

