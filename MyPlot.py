# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0], color="royalblue"),
                Line2D([0], [0], color="orange"),
                Line2D([0], [0], color="limegreen")]

class MyPlot:
    
    def __init__(self, names):
        self.names=names
              
    @staticmethod
    def NewFig(figname):
        plt.figure(figname)
        
        
    @staticmethod
    def Plot(Ytrain, Ypred):
        Res=Ypred.reshape(-1, 1)-Ytrain
        plt.xlabel='timeseries'
        plt.ylabel='thermal error(um)'
        plt.plot(Ytrain, color='royalblue', label='measurement')
        plt.plot(Ypred, color='orange', label='prediction')
        plt.plot(Res, color='limegreen', label='residual')
        
        
    @staticmethod
    def GetResult():
        plt.legend(custom_lines,['measurement','validation','residual'])
        plt.show()