from numpy.lib.function_base import select
from ConfigLoader import ConfigLoader
from Plotter import Plotter
from FindEquilibria import FindEquilibria
import numpy as np

class DynamicsUnited:
    '''
    In this class the coupled system of 4 animals is modeled
    '''
    def __init__(self):
        config = ConfigLoader()
        self.config = config.params
        self.dt = self.config['dt']
        self.x0, self.y0, self.z0, self.w0 = self.setInitialConditions()
        self.dx, self.dy, self.dz, self.dw = 0, 0, 0, 0
        self.a, self.b = self.setParameters()
        self.kp, self.sigma = self.setPositiveControlParameters()
        self.e = self.findEquilibria()
        self.isFirstIteration = True
        self.plotter = Plotter()

    def f(self, x0, y0, z0, w0, a, b):
        '''
        returns the dynamics of the coupled system
        '''
        # Closed Loop dynamics
        if self.config['useControl']:
            if self.config['usePositiveControl']:
                # CL dynamics with Positive Control: dx/dt = diag(x)*(A*(x-e)+k*u)
                k = np.array(self.kp)
                e = np.array(self.e)
                x = np.array([x0, y0, z0, w0])
                sigma = self.sigma
                u = sigma * (-np.matmul(np.transpose(k), (x - e)))
                #print('control input u=' + str(u))
                dx0 = x0 * (b[0] - a[0][0]*x0 - a[0][1]*y0 - a[0][2]*z0 - a[0][3]*w0 + k[0]*u)
                dy0 = y0 * (b[1] - a[1][0]*x0 - a[1][1]*y0 - a[1][2]*z0 - a[1][3]*w0 + k[1]*u)
                dz0 = z0 * (b[2] - a[2][0]*x0 - a[2][1]*y0 - a[2][2]*z0 - a[2][3]*w0 + k[2]*u)
                dw0 = w0 * (b[3] - a[3][0]*x0 - a[3][1]*y0 - a[3][2]*z0 - a[3][3]*w0 + k[3]*u)
        else:
            dx0 = x0 * (b[0] - a[0][0]*x0 - a[0][1]*y0 - a[0][2]*z0 - a[0][3]*w0)
            dy0 = y0 * (b[1] - a[1][0]*x0 - a[1][1]*y0 - a[1][2]*z0 - a[1][3]*w0)
            dz0 = z0 * (b[2] - a[2][0]*x0 - a[2][1]*y0 - a[2][2]*z0 - a[2][3]*w0)
            dw0 = w0 * (b[3] - a[3][0]*x0 - a[3][1]*y0 - a[3][2]*z0 - a[3][3]*w0)
        return dx0, dy0, dz0, dw0

    def updateRK4(self, x0, y0, z0, w0, a, b):
        dt = self.dt
        k1x, k1y, k1z, k1w = self.f(x0, y0, z0, w0, a, b)
        k2x, k2y, k2z, k2w = self.f(x0 + 0.5 * dt * k1x, y0 + 0.5 * dt * k1y, z0 + 0.5 * dt * k1z, w0 + 0.5 * dt * k1w, a, b)
        k3x, k3y, k3z, k3w = self.f(x0 + 0.5 * dt * k2x, y0 + 0.5 * dt * k2y, z0 + 0.5 * dt * k2z, w0 + 0.5 * dt * k2w, a, b)
        k4x, k4y, k4z, k4w = self.f(x0 + dt * k3x, y0 + dt * k3y, z0 + dt * k3z, w0 + dt * k3w, a, b)
        x1 = x0 + (1/6) * dt * (k1x + 2 * k2x + 2 * k3x + k4x)
        y1 = y0 + (1/6) * dt * (k1y + 2 * k2y + 2 * k3y + k4y)
        z1 = z0 + (1/6) * dt * (k1z + 2 * k2z + 2 * k3z + k4z)
        w1 = w0 + (1/6) * dt * (k1w + 2 * k2w + 2 * k3w + k4w)
        return x1, y1, z1, w1

    # Integration method
    def update(self):
        '''
        returns the updated x1, y1, z1, w1
        '''
        self.dx, self.dy, self.dz, self.dw = self.f(self.x0, self.y0, self.z0, self.w0, self.a, self.b)
        if self.config['useRK4']:
            self.x0, self.y0, self.z0, self.w0 = self.updateRK4(self.x0, self.y0, self.z0, self.w0, self.a, self.b)

        elif self.config['useEulerForward']:
            # self.x0, self.y0 = self.updateLVEuler(self.x0, self.y0, self.alpha1, self.beta1, self.gamma1, self.delta1)
            # self.z0, self.w0 = self.updateLVEuler(self.z0, self.w0, self.alpha2, self.beta2, self.gamma2, self.delta2)
            # return self.x0, self.y0, self.z0, self.w0
            print('Euler forward integration is not available yet.')
        plotBuffer, derivativePlotBuffer = self.plotter.updatePlotterBuffer(self.x0, self.y0, self.z0, self.w0, self.dx, self.dy, self.dz, self.dw)
        return plotBuffer, derivativePlotBuffer

    def setInitialConditions(self):
        return self.config['x0'], self.config['y0'], self.config['z0'], self.config['w0']

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
        self.x0, self.y0, self.z0, self.w0 = self.setInitialConditions()
    
    def setPositiveControlParameters(self):
        k = [self.config['k1'], self.config['k2'], self.config['k3'], self.config['k4']]
        sigma = self.config['sigma']
        return k, sigma

    def findEquilibria(self):
        equilibria = FindEquilibria()
        e = equilibria.findEquilibria(printEq=False)
        return e

# for testing:
if __name__ == '__main__':
    dynamics = DynamicsUnited()
    for i in range(0, 5):
        print(dynamics.update())