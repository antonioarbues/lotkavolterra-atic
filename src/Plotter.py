import numpy as np
import matplotlib.pyplot as plt
from ConfigLoader import ConfigLoader


class Plotter:
    def __init__(self):
        config = ConfigLoader()
        self.config = config.params
        self.dt = self.config['dt']
        self.plotBuffer = [[], [], [], []]
        self.plot = self.config['plot']

    def updatePlotterBuffer(self, x, y, z, w):
        self.plotBuffer[0].append(x)
        self.plotBuffer[1].append(y)
        self.plotBuffer[2].append(z)
        self.plotBuffer[3].append(w)
        if self.plot and len(self.plotBuffer[0]) == min(self.config['plotSize'], self.config['it']):
            self.plot2d(self.plotBuffer[0], self.plotBuffer[1], 1)
            self.plot2d(self.plotBuffer[2], self.plotBuffer[3], 2)
            self.plotBuffer = [[], [], [], []]    # reset plotter buffer

    def plot2d(self, v1, v2, systemN):
        '''
        plots 2D vector
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

    def plot2dvf(self):
        '''
        plots 2D vector field
        '''
        pass

    def plot3dvf(self):
        '''
        plots 3D vector field
        '''
        pass


# for testing:
if __name__ == '__main__':
    pass