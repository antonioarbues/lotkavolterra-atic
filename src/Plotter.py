import numpy as np
import matplotlib.pyplot as plt
from ConfigLoader import ConfigLoader


class Plotter:
    def __init__(self):
        config = ConfigLoader()
        self.config = config.params
        self.dt = self.config['dt']
        self.plotBuffer = [[], [], [], []]
        self.derivativePlotBuffer = [[], [], [], []]
        self.plot = self.config['plot']

    def updatePlotterBuffer(self, x, y, z, w, dx, dy, dz, dw):
        self.plotBuffer[0].append(x)
        self.plotBuffer[1].append(y)
        self.plotBuffer[2].append(z)
        self.plotBuffer[3].append(w)
        # derivatives for vector field
        self.derivativePlotBuffer[0].append(dx)
        self.derivativePlotBuffer[1].append(dy)
        self.derivativePlotBuffer[2].append(dz)
        self.derivativePlotBuffer[3].append(dw)

        if self.plot and len(self.plotBuffer[0]) == min(self.config['plotSize'], self.config['it']):
            if self.config['useDecoupled']:
                if self.config['plotEvolution']:
                    self.plot2d(self.plotBuffer[0], self.plotBuffer[1], 1)
                    self.plot2d(self.plotBuffer[2], self.plotBuffer[3], 2)
                if self.config['plotStateSpace']:
                    self.plotstatespace(self.plotBuffer[0], self.plotBuffer[1], 1)
                    self.plotstatespace(self.plotBuffer[2], self.plotBuffer[3], 2)
                if self.config['plotVectorField']:
                    self.plot2dvf(self.plotBuffer[0], self.plotBuffer[1], self.plotBuffer[2],self.plotBuffer[3], \
                        self.derivativePlotBuffer[0], self.derivativePlotBuffer[1],self.derivativePlotBuffer[2], self.derivativePlotBuffer[3])
                self.plotBuffer = [[], [], [], []]    # reset plotter buffer
            elif self.config['useCoupled']:
                if self.config['plotEvolution']:
                    self.plot2d4(self.plotBuffer[0], self.plotBuffer[1], self.plotBuffer[2],self.plotBuffer[3])
                if self.config['plotStateSpace']:
                    self.plotstatespace4(self.plotBuffer[0], self.plotBuffer[1], self.plotBuffer[2], self.plotBuffer[3])
                if self.config['plotVectorField']:
                    self.plot2dvf4(self.plotBuffer[0], self.plotBuffer[1], self.plotBuffer[2],self.plotBuffer[3], \
                        self.derivativePlotBuffer[0], self.derivativePlotBuffer[1],self.derivativePlotBuffer[2], self.derivativePlotBuffer[3])
                self.plotBuffer = [[], [], [], []]    # reset plotter buffer
                self.derivativePlotBuffer = [[], [], [], []]


    def plot2d(self, v1, v2, systemN):
        '''
        plots two vectors
        '''
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,7))
        ax.plot((self.dt * np.array(list(range(len(v1))))).tolist(), v1, label = 'Prey')
        ax.plot((self.dt * np.array(list(range(len(v2))))).tolist(), v2, label = 'Predator')
        plt.title('System number ' + str(systemN))
        plt.xlabel('Time')
        plt.ylabel('Population size')
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    def plotstatespace(self, v1, v2, systemN):
        '''
        plots the state space between two states
        '''
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,7))
        ax.plot(v1, v2)
        plt.title('State space of system number ' + str(systemN))
        plt.xlabel('Time')
        plt.ylabel('Population size')
        ax.legend()
        plt.tight_layout()
        plt.show()

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
        # plt.title('State space of the coupled system')
        # plt.xlabel('Time')
        # plt.ylabel('Population size')
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        ax5.legend()
        ax6.legend()
        plt.tight_layout()
        plt.show()


    def plot2d4(self, v1, v2, v3, v4):
        '''
        plots four vectors
        '''
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,7))
        ax.plot((self.dt * np.array(list(range(len(v1))))).tolist(), v1, label = 'Prey 1')
        ax.plot((self.dt * np.array(list(range(len(v2))))).tolist(), v2, label = 'Predator 1')
        ax.plot((self.dt * np.array(list(range(len(v3))))).tolist(), v3, label = 'Prey 2')
        ax.plot((self.dt * np.array(list(range(len(v4))))).tolist(), v4, label = 'Predator 2')
        plt.title('Coupled system')
        plt.xlabel('Time')
        plt.ylabel('Population size')
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

    def plot2dvf(self, x, y, z, w, dx, dy, dz, dw):
        '''
        plots 2D vector field
        x and y are the coordinates of the arrows, u and v are the directions,
        where for instance u=1, v=1 is a 45deg arrow
        '''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,7))
        ax1.quiver(x,y,dx,dy, label = 'x1-x2')
        ax2.quiver(z,w,dz,dw, label = 'x3-x4')
        plt.show()

    def plot3dvf(self):
        '''
        plots 3D vector field
        '''
        pass


# for testing:
if __name__ == '__main__':
    pass