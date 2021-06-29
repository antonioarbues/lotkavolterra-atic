import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ConfigLoader import ConfigLoader
plt.rcParams.update({'font.size': 16})
plt.rcParams['legend.loc'] = "upper right"


class Plotter:
    def __init__(self):
        config = ConfigLoader()
        self.config = config.params
        self.dt = self.config['dt']
        self.plotBuffer = [[], [], [], []]
        self.derivativePlotBuffer = [[], [], [], []]
        self.plot = self.config['plot']
        self.controlInputs = []

    def updatePlotterBuffer(self, x, y, z, w, dx, dy, dz, dw, *args):
        self.plotBuffer[0].append(x)
        self.plotBuffer[1].append(y)
        self.plotBuffer[2].append(z)
        self.plotBuffer[3].append(w)
        # derivatives for vector field
        self.derivativePlotBuffer[0].append(dx)
        self.derivativePlotBuffer[1].append(dy)
        self.derivativePlotBuffer[2].append(dz)
        self.derivativePlotBuffer[3].append(dw)

        for el in args:
            self.controlInputs.append(el)

        if self.plot and len(self.plotBuffer[0]) == min(self.config['plotSize'], self.config['it']) and not self.config['compareIC']:
            if self.config['plotEvolution']:
                if self.config['showControlInputs'] and self.config['useControl']:
                    self.plot2d4(self.plotBuffer[0], self.plotBuffer[1], self.plotBuffer[2],self.plotBuffer[3], True, self.controlInputs)
                else:
                    self.plot2d4(self.plotBuffer[0], self.plotBuffer[1], self.plotBuffer[2],self.plotBuffer[3])
            if self.config['plotStateSpace']:
                self.plotstatespace4(self.plotBuffer[0], self.plotBuffer[1], self.plotBuffer[2], self.plotBuffer[3])
            if self.config['plotVectorField']:
                self.plot2dvf4(self.plotBuffer[0], self.plotBuffer[1], self.plotBuffer[2],self.plotBuffer[3], \
                    self.derivativePlotBuffer[0], self.derivativePlotBuffer[1],self.derivativePlotBuffer[2], self.derivativePlotBuffer[3])
            if self.config['plot4dLimitCycle']:
                self.plot4dLimitcycle(self.plotBuffer[0], self.plotBuffer[1],self.plotBuffer[2], self.plotBuffer[3])
            if self.config['resetPlotBuffer'] and not self.config['compareIC']:
                self.plotBuffer = [[], [], [], []]    # reset plotter buffer
                self.derivativePlotBuffer = [[], [], [], []]
        return self.plotBuffer, self.derivativePlotBuffer

    def plotstatespace4(self, v1, v2, v3, v4):
        '''
        plots the state space between four states in subplots
        '''
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(9,7))
        ax1.plot(v1, v2, label='x1-x2')
        ax2.plot(v1, v3, label='x1-x3')
        ax3.plot(v1, v4, label='x1-x4')
        ax4.plot(v2, v3, label='x2-x3')
        ax5.plot(v2, v4, label='x2-x4')
        ax6.plot(v3, v4, label='x3-x4')
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        ax5.legend()
        ax6.legend()
        ax1.tick_params(axis='x', labelrotation=45)
        ax2.tick_params(axis='x', labelrotation=45)
        ax3.tick_params(axis='x', labelrotation=45)
        ax4.tick_params(axis='x', labelrotation=45)
        ax5.tick_params(axis='x', labelrotation=45)
        ax6.tick_params(axis='x', labelrotation=45)
        plt.tight_layout()
        plt.show()


    def plot2d4(self, v1, v2, v3, v4, plotinput:bool=False, *args):
        '''
        plots four vectors
        pass a bool and more vectors if you want them to get plotted together with the trajectories
        ex: pass 4 vectors, True, inputs
        '''
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,7))
        ax.plot((self.dt * np.array(list(range(len(v1))))).tolist(), v1, label = 'rabbit',linewidth=3)
        ax.plot((self.dt * np.array(list(range(len(v2))))).tolist(), v2, label = 'snake',linewidth=3)
        ax.plot((self.dt * np.array(list(range(len(v3))))).tolist(), v3, label = 'deer',linewidth=3)
        ax.plot((self.dt * np.array(list(range(len(v4))))).tolist(), v4, label = 'eagle',linewidth=3)
        if plotinput:
            for el in args:
                ax.plot((self.dt * np.array(list(range(len(el))))).tolist(), el, '--', label = 'control input')
        plt.xlabel('Time', fontsize = 16)
        plt.ylabel('Population size', fontsize = 16)
        ax.legend()

        plt.tight_layout()
        plt.show()

    def plot2dvf4(self, x, y, z, w, dx, dy, dz, dw):
        '''
        plots 2D vector field
        x and y are the coordinates of the arrows, u and v are the directions,
        where for instance u=1, v=1 is a 45deg arrow
        '''
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(9,7))
        ax1.quiver(x,y,dx,dy, label = 'x1-x2')
        ax2.quiver(x,z,dx,dz, label = 'x1-x3')
        ax3.quiver(x,w,dx,dw, label = 'x1-x4')
        ax4.quiver(y,z,dy,dz, label = 'x2-x3')
        ax5.quiver(y,w,dy,dw, label = 'x2-x4')
        ax6.quiver(z,w,dz,dw, label = 'x3-x4')
        plt.show()

    def plotEigen(self, eigens):
        x = [a.real for a in eigens]
        y = [a.imag for a in eigens]
        plt.scatter(x, y, color='red')
        plt.show()

    def plotStability(self, X_st, Y_st, X_unst, Y_unst):
        plt.scatter(X_st, Y_st, color='green')
        plt.scatter(X_unst, Y_unst, color='red')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def plot4dLimitcycle(self, x, y, z, w):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        img = ax.scatter(x, y, z, c=w, cmap = plt.summer())
        cbar = fig.colorbar(img)
        ax.set_xlabel('rabbit', fontsize=18)
        ax.set_ylabel('snake', fontsize=18)    
        ax.set_zlabel('deer', fontsize=18, rotation=90)
        cbar.set_label('eagle', fontsize = 18)
        ax.tick_params(axis='x', labelrotation=30)
        plt.show()
