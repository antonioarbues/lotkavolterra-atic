from ConfigLoader import ConfigLoader
import numpy as np
from numpy import linalg as LA
from Plotter import Plotter

class FindEquilibria:
    '''
    this class implements the algorithms to find the equilibria of the system
    '''
    def __init__(self):
        config = ConfigLoader()
        self.config = config.params
        self.a, self.b = self.setParameters()
        self.plotter = Plotter()

    def setParameters(self):
        a = [[self.config['a11'], self.config['a12'], self.config['a13'], self.config['a14']], \
            [self.config['a21'], self.config['a22'], self.config['a23'], self.config['a24']], \
            [self.config['a31'], self.config['a32'], self.config['a33'], self.config['a34']], \
            [self.config['a41'], self.config['a42'], self.config['a43'], self.config['a44']]]
        b = [self.config['b1'], self.config['b2'], self.config['b3'], self.config['b4']]

        if 'X' in self.config and 'Y' in self.config:
            X = self.config['X']
            Y = self.config['Y']
            a[1][2] = -X
            a[2][1] = X/2
            a[1][3] = -Y
            a[3][1] = Y/2
        return a, b
    
    def setParametersAgain(self):
        self.a, self.b = self.setParameters()

    def findEquilibria(self, printEq=True):
        eq = np.matmul(np.linalg.inv(np.array(self.a)), self.b).tolist()
        if self.config['printEquilibria']:
            print('The equilibria of the system are:\n')
            if printEq:
                for el in eq:
                    print(str(el) + '\n')
        return eq

    def linearise(self, equilibrium):
        # linearised dynamics 
        b = self.b
        a = self.a
        x0 = equilibrium[0]
        y0 = equilibrium[1]
        z0 = equilibrium[2]
        w0 = equilibrium[3]
        dyn_lin = [[b[0] - 2*a[0][0]*x0 - a[0][1]*y0 - a[0][2]*z0 - a[0][3]*w0, -a[0][1]*x0, -a[0][2]*x0, -a[0][3]*x0], \
            [-a[1][0]*y0, b[1] - a[1][0]*x0 - 2*a[1][1]*y0 - a[1][2]*z0 - a[1][3]*w0, -a[1][2]*y0, -a[1][3]*y0], \
            [-a[2][0]*z0, -a[2][1]*z0, b[2] - a[2][0]*x0 - a[2][1]*y0 - 2*a[2][2]*z0 - a[2][3]*w0, -a[2][3]*z0], \
            [-a[3][0]*w0, -a[3][1]*w0, -a[3][2]*w0, b[3] - a[3][0]*x0 - a[3][1]*y0 - a[3][2]*z0 - 2*a[3][3]*w0]]
        print(equilibrium)
        print('---')
        print(dyn_lin)
        return dyn_lin
    
    def getEigens(self):
        eq = self.findEquilibria()
        dyn_lin = self.linearise(eq)
        eigenvalues, _ = LA.eig(dyn_lin)
        if self.config['plotEigenvalues']:
            self.plotter.plotEigen(eigenvalues)
        return eigenvalues
