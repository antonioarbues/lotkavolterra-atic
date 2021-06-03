from ConfigLoader import ConfigLoader
import numpy as np

class FindEquilibria:
    '''
    this class implements the algorithms to find the equilibria of the system
    '''
    def __init__(self):
        config = ConfigLoader()
        self.config = config.params
        self.a, self.b = self.setParameters()

    def findEquilibria(self):
        eq = np.matmul(np.linalg.inv(np.array(self.a)), self.b)
        print('The equilibria of the system are:\n')
        for el in eq:
            print(str(el) + '\n')
        return eq

    def setParameters(self):
        a = [[self.config['a11'], self.config['a12'], self.config['a13'], self.config['a14']], \
            [self.config['a21'], self.config['a22'], self.config['a23'], self.config['a24']], \
            [self.config['a31'], self.config['a32'], self.config['a33'], self.config['a34']], \
            [self.config['a41'], self.config['a42'], self.config['a43'], self.config['a44']]]
        b = [self.config['b1'], self.config['b2'], self.config['b3'], self.config['b4']]
        return a, b