from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np


class HillPlotter:
    def __init__(self,dimrange,func,args):
        self.dim = dimrange
        self.func = func
        self.args = args
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,projection='3d')

    def getResults(self,X,Y):
        res = []
        for x,y in zip(X,Y):
            res.append(self.func(self.args,(x,y)))
        return np.array(res)

    def plot(self,point,spottedPoints=None,spottedValues=None):
        x = y = np.arange(self.dim[0],self.dim[1],0.05)
        X,Y = np.meshgrid(x,y)
        Z = self.getResults(X,Y)
        self.ax.plot_wireframe(X,Y,Z,color="#00B5BD")
        x,y,z = np.array(point[0]),np.array(point[1]),np.array(self.func(self.args,(point[0],point[1])))
        self.ax.scatter(x,y,z+1,color='r' ,marker='p')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        if spottedPoints!=None and spottedValues!=None:
            sx = []
            sy = []
            sz = []
            for p,z in zip(spottedPoints,spottedValues):
                sx.append(p[0])
                sy.append(p[1])
                sz.append(z)
            self.ax.scatter(np.array(sx),np.array(sy),np.array(z),color='k',marker='^')
        plt.show()


