import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))
sys.path.append(os.path.abspath(os.path.join('.', 'src')))
from ConfigLoader import ConfigLoader
from FindEquilibria import FindEquilibria
from Model import Model
import numpy as np

class Estimator:
    '''
    The class Estimator implements the Extended Kalman Filter
    for Generalized Lotka-Volterra dynamics

    Legend:
    x00 = x(k-1|k-1)
    x10 = x(k|k-1)
    x11 = x(k|k)
    '''
    def __init__(self):
        config = ConfigLoader()
        self.config = config.params
        self.model = Model()
        self.x00 = np.array([self.model.x0, self.model.y0, self.model.z0, self.model.w0])
        self.P00 = np.zeros((4, 4))

    def predict(self, u):
        x0 = self.x00[0]
        y0 = self.x00[1]
        z0 = self.x00[2]
        w0 = self.x00[3]
        F = self.getJacobianOfF(self.x00, u)
        # Predicted state estimate
        self.x10 = np.array(self.model.updateRK4(x0, y0, z0, w0, self.model.a, self.model.b, processnoise=True, measurementnoise=False))
        # Predicted covariance estimate
        self.P10 = np.matmul(np.matmul(F, self.P00), np.transpose(F)) + self.model.Q
        return self.x10, self.P10

    def update(self, u):
        self.predict(u)
        x0 = self.x00[0]
        y0 = self.x00[1]
        z0 = self.x00[2]
        w0 = self.x00[3]
        H = self.getJacobianOfH()
        # Innovation or measurement residual
        y1x, y1y, y1z, y1w = self.model.updateRK4(x0, y0, z0, w0, self.model.a, self.model.b, processnoise=True, measurementnoise=False) + self.model.getMeasurementNoise() - self.x10
        self.y1 = np.array([y1x, y1y, y1z, y1w])
        # Innovation (or residual) covariance
        self.S1 = np.matmul(np.matmul(H, self.P10), np.transpose(H)) + self.model.R
        # Near-optimal Kalman gain
        self.K = np.matmul(np.matmul(self.P10, np.transpose(H)), np.linalg.inv(self.S1))
        # Updated state estimate (called already x00 instead of x11 for next timestep)
        self.x00 = self.x10 + np.matmul(self.K, self.y1)

        # print('ESTIMATED STATES\n')
        # print(str(self.x00.tolist()) + '\n')

        # Updated covariance estimate
        self.P00 = np.matmul(np.eye(4) - np.matmul(self.K, H), self.P10)
        return self.x00

    def getJacobianOfF(self, x00, u):
        jacobian = self.model.equilibria.linearise(x00, True, u)
        return jacobian

    def getJacobianOfH(self):
        return np.eye(4)


# for testing
if __name__ == '__main__':
    est = Estimator()
    sigma = est.model.sigma
    k = est.model.kp
    x = est.x00
    e = est.model.equilibria.findEquilibria()
    u = sigma * (-np.matmul(np.transpose(k), (x - e)))
    for i in range(100):
        est.update(u)