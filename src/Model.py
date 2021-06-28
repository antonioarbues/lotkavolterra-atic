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
        self.Q, self.R, self.QS = self.getCovariances()
        self.a, self.b = self.setParameters()
        self.x0, self.y0, self.z0, self.w0 = self.setInitialConditions()
        self.ref = self.getReference()
        self.ko = self.setOptimalControlParameters()
        self.dx, self.dy, self.dz, self.dw = 0, 0, 0, 0
        self.dt = self.config['dt']
        self.equilibria = FindEquilibria()
        self.e = self.equilibria.findEquilibria(printEq=False)
        x0_vec = np.array([self.x0, self.y0, self.z0, self.w0])
        self.u = self.sigma * (-np.matmul(np.transpose(self.kp), (x0_vec - self.e)))

    def f(self, x0, y0, z0, w0, a, b, usecontrol:bool=True):
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
            elif self.config['useOptimalControl']:
                F0, F1, F2, F3 = self.computeOptimalControl(x0, y0, z0, w0, a, b)
                dx0 = x0 * (b[0] - a[0][0]*x0 - a[0][1]*y0 - a[0][2]*z0 - a[0][3]*w0) + F0
                dy0 = y0 * (b[1] - a[1][0]*x0 - a[1][1]*y0 - a[1][2]*z0 - a[1][3]*w0) + F1
                dz0 = z0 * (b[2] - a[2][0]*x0 - a[2][1]*y0 - a[2][2]*z0 - a[2][3]*w0) + F2
                dw0 = w0 * (b[3] - a[3][0]*x0 - a[3][1]*y0 - a[3][2]*z0 - a[3][3]*w0) + F3
        else:
            dx0 = x0 * (b[0] - a[0][0]*x0 - a[0][1]*y0 - a[0][2]*z0 - a[0][3]*w0)
            dy0 = y0 * (b[1] - a[1][0]*x0 - a[1][1]*y0 - a[1][2]*z0 - a[1][3]*w0)
            dz0 = z0 * (b[2] - a[2][0]*x0 - a[2][1]*y0 - a[2][2]*z0 - a[2][3]*w0)
            dw0 = w0 * (b[3] - a[3][0]*x0 - a[3][1]*y0 - a[3][2]*z0 - a[3][3]*w0)

        return dx0, dy0, dz0, dw0

    def updateRK4(self, x0, y0, z0, w0, a, b, processnoise:bool, measurementnoise:bool, simulatornoise:bool, usecontrol:bool=True):
        dt = self.dt
        k1x, k1y, k1z, k1w = self.f(x0, y0, z0, w0, a, b, usecontrol)
        k2x, k2y, k2z, k2w = self.f(x0 + 0.5 * dt * k1x, y0 + 0.5 * dt * k1y, z0 + 0.5 * dt * k1z, w0 + 0.5 * dt * k1w, a, b, usecontrol)
        k3x, k3y, k3z, k3w = self.f(x0 + 0.5 * dt * k2x, y0 + 0.5 * dt * k2y, z0 + 0.5 * dt * k2z, w0 + 0.5 * dt * k2w, a, b, usecontrol)
        k4x, k4y, k4z, k4w = self.f(x0 + dt * k3x, y0 + dt * k3y, z0 + dt * k3z, w0 + dt * k3w, a, b, usecontrol)
        x1 = x0 + (1/6) * dt * (k1x + 2 * k2x + 2 * k3x + k4x)
        y1 = y0 + (1/6) * dt * (k1y + 2 * k2y + 2 * k3y + k4y)
        z1 = z0 + (1/6) * dt * (k1z + 2 * k2z + 2 * k3z + k4z)
        w1 = w0 + (1/6) * dt * (k1w + 2 * k2w + 2 * k3w + k4w)

        noise_p = np.array([0,0,0,0])
        noise_m = np.array([0,0,0,0])
        noise_s = np.array([0,0,0,0])
        if processnoise:
            noise_p = self.getProcessNoise()
        if measurementnoise:
            noise_m = self.getMeasurementNoise()
        if simulatornoise:
            noise_s = self.getSimulatorNoise()
        return x1 + noise_p[0] + noise_m[0] + noise_s[0], \
             y1 + noise_p[1] + noise_m[0] + noise_s[0], \
                  z1 + noise_p[2] + noise_m[0] + noise_s[0], \
                       w1 + noise_p[3] + noise_m[0] + noise_s[0]

    def computeOptimalControl(self, x0, y0, z0, w0, a, b):

        ko = np.array(self.ko)
        e = self.ref
        x = np.array([x0, y0, z0, w0])

        for i in range(4):
            for j in range(4):
                a[i][j] = -a[i][j]

        #Computing the control functions 
        eps0 = (x[0] - e[0])
        eps1 = (x[1] - e[1])
        eps2 = (x[2] - e[2])
        eps3 = (x[3] - e[3])

        sum0 = 0
        sum1 = 0
        sum2 = 0
        sum3 = 0

        eps= np.array([eps0, eps1, eps2, eps3])

        #Computing the steady states 
        sum_ss0 = 0
        sum_ss1 = 0
        sum_ss2 = 0
        sum_ss3 = 0

        for i in range(4):
            sum_ss0 += a[0][i] * e[0]*e[i]
            sum_ss1 += a[1][i] * e[1]*e[i]
            sum_ss2 += a[2][i] * e[2]*e[i]
            sum_ss3 += a[3][i] * e[3]*e[i]

        for i in range(4):
            sum0 = sum0 + a[0][i] * (e[0] * eps[i] + e[i]*eps[0] + eps[0]*eps[i])
            sum1 = sum1 + a[1][i] * (e[1] * eps[i] + e[i]*eps[1] + eps[1]*eps[i])
            sum2 = sum2 + a[2][i] * (e[2] * eps[i] + e[i]*eps[2] + eps[2]*eps[i])
            sum3 = sum3 + a[3][i] * (e[3] * eps[i] + e[i]*eps[3] + eps[3]*eps[i])
        
        V0 = (-(ko[0] + b[0]) * eps[0])  - sum0
        V1 = (-(ko[1] + b[1]) * eps[1])  - sum1
        V2 = (-(ko[2] + b[2]) * eps[2])  - sum2
        V3 = (-(ko[3] + b[3]) * eps[3])  - sum3
        
        F0_ss = -b[0] * e[0] - sum_ss0 
        F1_ss = -b[1] * e[1] - sum_ss1 
        F2_ss = -b[2] * e[2] - sum_ss2 
        F3_ss = -b[3] * e[3] - sum_ss3

        F0 = V0 + F0_ss
        F1 = V1 + F1_ss
        F2 = V2 + F2_ss
        F3 = V3 + F3_ss

        return F0, F1, F2, F3
    
    def setOptimalControlParameters(self):
        k =[self.config['k1o'], self.config['k2o'], self.config['k3o'], self.config['k4o']]
        return k
    
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

    def getReference(self):
        ref = np.array([self.config['ref_x'], self.config['ref_y'], self.config['ref_z'], self.config['ref_w']])
        return ref

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

    def getSimulatorNoise(self):
        '''
        Simulator noise is the noise applied to the measurements
        '''
        noise = []
        for i in range(4):
            noise.append(np.random.normal(0, self.QS[i][i], 1)[0])
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
        QS = np.array([[self.config['QS11'], 0, 0, 0],\
            [0, self.config['QS22'], 0, 0], \
            [0, 0, self.config['QS33'], 0], \
            [0, 0, 0, self.config['QS44']]])
        return Q, R, QS

    def setInitialConditions(self):
        return self.config['x0'], self.config['y0'], self.config['z0'], self.config['w0']

    def setParametersAgain(self):
        self.a, self.b = self.setParameters()
        self.x0, self.y0, self.z0, self.w0 = self.setInitialConditions()
