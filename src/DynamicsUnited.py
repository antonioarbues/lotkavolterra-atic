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
        self.ko = self.setOptimalControlParameters()
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
                r=np.array([2, 3, 4, 1])
                sigma = self.sigma
                #u = sigma * (-np.matmul(np.transpose(k), (x -e)))
                u = sigma * (-np.matmul(np.transpose(k), (x -r)))
                #print('control input u=' + str(u))
                dx0 = x0 * (b[0] - a[0][0]*x0 - a[0][1]*y0 - a[0][2]*z0 - a[0][3]*w0 + k[0]*u)
                dy0 = y0 * (b[1] - a[1][0]*x0 - a[1][1]*y0 - a[1][2]*z0 - a[1][3]*w0 + k[1]*u)
                dz0 = z0 * (b[2] - a[2][0]*x0 - a[2][1]*y0 - a[2][2]*z0 - a[2][3]*w0 + k[2]*u)
                dw0 = w0 * (b[3] - a[3][0]*x0 - a[3][1]*y0 - a[3][2]*z0 - a[3][3]*w0 + k[3]*u)
            if self.config['useOptimalControl']:
                F0, F1, F2, F3 = self.computeOptimalControl(x0, y0, z0, w0, a, b)
                # print('control input F0 =' + str(F0))
                # print('control input F1 =' + str(F1))
                # print('control input F2 =' + str(F2))
                # print('control input F3 =' + str(F3))

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

    def setOptimalControlParameters(self):
        k =[self.config['k1o'], self.config['k2o'], self.config['k3o'], self.config['k4o']]
        return k
    
    def computeOptimalControl(self, x0, y0, z0, w0, a, b):


        ko = np.array(self.ko)
        e = np.array(self.e)
        x = np.array([x0, y0, z0, w0])
        dt = self.dt

        e=np.array([2, 3, 4, 1]) #Reference tracking (gypsy change for testing, should be done properly)

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

        # for i in range(4):
        #     sum0 = sum0 + a[0][i] * (e[0] * eps[i] * np.exp(-kp[i] * dt) + e[i]*eps[0]*np.exp(-kp[0]*dt) + eps[0] * eps[i] *np.exp(-(kp[0]+kp[i])*dt))
        #     sum1 = sum1 + a[1][i] * (e[1] * eps[i] * np.exp(-kp[i] * dt) + e[i]*eps[1]*np.exp(-kp[1]*dt) + eps[1] * eps[i] *np.exp(-(kp[1]+kp[i])*dt))
        #     sum2 =  sum2 + a[2][i] * (e[2] * eps[i] * np.exp(-kp[i] * dt) + e[i]*eps[2]*np.exp(-kp[2]*dt) + eps[2] * eps[i] *np.exp(-(kp[2]+kp[i])*dt))
        #     sum3 = sum3 + a[3][i] * (e[3] * eps[i] * np.exp(-kp[i] * dt) + e[i]*eps[3]*np.exp(-kp[3]*dt) + eps[3] * eps[i] *np.exp(-(kp[3]+kp[i])*dt))

        # V0 = (-(kp[0] + b[0]) * eps[0] * np.exp(-kp[0]*dt)) - sum0
        # V1 = (-(kp[1] + b[1]) * eps[1] * np.exp(-kp[1]*dt)) - sum1
        # V2 = (-(kp[2] + b[2]) * eps[2] * np.exp(-kp[2]*dt)) - sum2
        # V3 = (-(kp[3] + b[3]) * eps[3] * np.exp(-kp[3]*dt)) - sum3

        #Option without synchro

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
        
        # print('F0_ss =' + str(F0_ss))
        # print('F1_ss =' + str(F1_ss))
        # print('F2_ss =' + str(F2_ss))
        # print('F3_ss =' + str(F3_ss))


        F0 = V0 + F0_ss
        F1 = V1 + F1_ss
        F2 = V2 + F2_ss
        F3 = V3 + F3_ss


        #return V0, V1, V2, V3
        return F0, F1, F2, F3


    def findEquilibria(self):
        equilibria = FindEquilibria()
        e = equilibria.findEquilibria(printEq=False)
        return e
    
    def plotStabilityXY(self):
        X_st = []
        X_unst = []
        Y_st = []
        Y_unst = []
        range_stab = self.config['stability_discretization']
        for i in range(0, range_stab):
            for j in range (0, range_stab):
                equilibria = FindEquilibria()
                equilibria.config['X'] = i / range_stab
                equilibria.config['Y'] = j / range_stab
                equilibria.setParametersAgain()
                eigs = equilibria.getEigens()
                stable = True
                for el in eigs:
                    if el.real > 0:
                        stable = False
                if stable:
                    X_st.append(i/range_stab)
                    Y_st.append(j/range_stab)
                else:
                    X_unst.append(i/range_stab)
                    Y_unst.append(j/range_stab)

        self.plotter.plotStability(X_st, Y_st, X_unst, Y_unst)

        equilibria = FindEquilibria()   # reset config X and Y
    

# for testing:
if __name__ == '__main__':
    dynamics = DynamicsUnited()
    for i in range(0, 5):
        print(dynamics.update())