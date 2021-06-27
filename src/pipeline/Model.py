import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))
sys.path.append(os.path.abspath(os.path.join('.', 'src')))
from ConfigLoader import ConfigLoader
from FindEquilibria import FindEquilibria
import numpy as np

class Model:
    '''
    '''
    def __init__(self):
        config = ConfigLoader()
        self.config = config.params
        self.kp, self.sigma = self.setPositiveControlParameters()
        self.Q, self.R = self.getCovariances()
        self.a, self.b = self.setParameters()
        self.x0, self.y0, self.z0, self.w0 = self.setInitialConditions()
        self.dx, self.dy, self.dz, self.dw = 0, 0, 0, 0
        self.dt = self.config['dt']
        self.equilibria = FindEquilibria()
        self.e = self.equilibria.findEquilibria(printEq=False)
        x0_vec = np.array([self.x0, self.y0, self.z0, self.w0])
        self.u = self.sigma * (-np.matmul(np.transpose(self.kp), (x0_vec - self.e)))

    def f(self, x0, y0, z0, w0, a, b, processnoise:bool, measurementnoise:bool, usecontrol:bool=True):
        '''
        returns the dynamics of the coupled system
        '''
        # Closed Loop dynamics
        if self.config['useControl'] and usecontrol:
            if self.config['usePositiveControl']:
                # CL dynamics with Positive Control: dx/dt = diag(x)*(A*(x-e)+k*u)
                k = np.array(self.kp)
                e = np.array(self.e)
                x = np.array([x0, y0, z0, w0])
                sigma = self.sigma
                self.u = sigma * (-np.matmul(np.transpose(k), (x - e)))
                u = self.u
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

    def updateRK4(self, x0, y0, z0, w0, a, b, processnoise:bool, measurementnoise:bool, usecontrol:bool=True):
        dt = self.dt
        k1x, k1y, k1z, k1w = self.f(x0, y0, z0, w0, a, b, processnoise, measurementnoise, usecontrol)
        k2x, k2y, k2z, k2w = self.f(x0 + 0.5 * dt * k1x, y0 + 0.5 * dt * k1y, z0 + 0.5 * dt * k1z, w0 + 0.5 * dt * k1w, a, b, processnoise, measurementnoise, usecontrol)
        k3x, k3y, k3z, k3w = self.f(x0 + 0.5 * dt * k2x, y0 + 0.5 * dt * k2y, z0 + 0.5 * dt * k2z, w0 + 0.5 * dt * k2w, a, b, processnoise, measurementnoise, usecontrol)
        k4x, k4y, k4z, k4w = self.f(x0 + dt * k3x, y0 + dt * k3y, z0 + dt * k3z, w0 + dt * k3w, a, b, processnoise, measurementnoise, usecontrol)
        x1 = x0 + (1/6) * dt * (k1x + 2 * k2x + 2 * k3x + k4x)
        y1 = y0 + (1/6) * dt * (k1y + 2 * k2y + 2 * k3y + k4y)
        z1 = z0 + (1/6) * dt * (k1z + 2 * k2z + 2 * k3z + k4z)
        w1 = w0 + (1/6) * dt * (k1w + 2 * k2w + 2 * k3w + k4w)

        # print('GROUND TRUTH\n')
        # print(str(x1) + ' ' + str(y1) + ' ' + str(z1) + ' ' + str(w1) + '\n')

        if processnoise:
            noise = self.getProcessNoise()
            return x1 + noise[0], y1 + noise[1], z1 + noise[2], w1 + noise[3]
        elif measurementnoise:
            noise = self.getMeasurementNoise()
            return x1 + noise[0], y1 + noise[1], z1 + noise[2], w1 + noise[3]
        else:
            return x1, y1, z1, w1

    def setPositiveControlParameters(self):
        k = [self.config['k1'], self.config['k2'], self.config['k3'], self.config['k4']]
        sigma = self.config['sigma']
        return k, sigma
    
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

    def getProcessNoise(self):
        '''
        Process noise is the noise applied to the states
        '''
        noise = []
        for i in range(4):
            noise.append(np.random.normal(0, self.Q[i][i], 1)[0])
        return np.array(noise)

    def getMeasurementNoise(self):
        '''
        Measurement noise is the noise applied to the measurements
        '''
        noise = []
        for i in range(4):
            noise.append(np.random.normal(0, self.R[i][i], 1)[0])
        return np.array(noise)

    def getCovariances(self):
        Q = np.array([[self.config['Q11'], 0, 0, 0],\
            [0, self.config['Q22'], 0, 0], \
            [0, 0, self.config['Q33'], 0], \
            [0, 0, 0, self.config['Q44']]])
        R = np.array([[self.config['R11'], 0, 0, 0],\
            [0, self.config['R22'], 0, 0], \
            [0, 0, self.config['R33'], 0], \
            [0, 0, 0, self.config['R44']]])
        return Q, R

    def setInitialConditions(self):
        return self.config['x0'], self.config['y0'], self.config['z0'], self.config['w0']