from ConfigLoader import ConfigLoader
from Plotter import Plotter

class Dynamics:
    def __init__(self):
        config = ConfigLoader()
        self.config = config.params
        self.dt = self.config['dt']
        self.x0, self.y0, self.z0, self.w0 = self.setInitialConditions()
        self.alpha1, self.beta1, self.gamma1, self.delta1, \
            self.alpha2, self.beta2, self.gamma2, self.delta2 = self.setParameters()
        self.isFirstIteration = True
        self.plotter = Plotter()

    def updateLVEuler(self, x0, y0, alpha, beta, gamma, delta):
        if self.isFirstIteration:
            self.isFirstIteration = False
            return x0, y0
        x1 = x0 + self.dt * (alpha * x0 - beta * x0 * y0)
        y1 = y0 + self.dt * (delta * x0 * y0 - gamma * y0)
        return x1, y1

    def updateRK4(self, x0, y0, alpha, beta, gamma, delta):
        dt = self.dt
        def f(x0, y0):
            '''
            returns the dynamics of the system
            '''
            return alpha * x0 - beta * x0 * y0, delta * x0 * y0 - gamma * y0
        k1x, k1y = f(x0, y0)
        k2x, k2y = f(x0 + 0.5 * dt * k1x, y0 + 0.5 * dt * k1y)
        k3x, k3y = f(x0 + 0.5 * dt * k2x, y0 + 0.5 * dt * k2y)
        k4x, k4y = f(x0 + dt * k3x, y0 + dt * k3y)
        x1 = x0 + (1/6) * dt * (k1x + 2 * k2x + 2 * k3x + k4x)
        y1 = y0 + (1/6) * dt * (k1y + 2 * k2y + 2 * k3y + k4y)
        return x1, y1

    # Integration method
    def update(self):
        '''
        returns the updated x1, y1, z1, w1
        '''
        self.plotter.updatePlotterBuffer(self.x0, self.y0, self.z0, self.w0)
        if self.config['useRK4']:
            self.x0, self.y0 = self.updateRK4(self.x0, self.y0, self.alpha1, self.beta1, self.gamma1, self.delta1)
            self.z0, self.w0 = self.updateRK4(self.z0, self.w0, self.alpha2, self.beta2, self.gamma2, self.delta2)
            return self.x0, self.y0, self.z0, self.w0

        elif self.config['useEulerForward']:
            self.x0, self.y0 = self.updateLVEuler(self.x0, self.y0, self.alpha1, self.beta1, self.gamma1, self.delta1)
            self.z0, self.w0 = self.updateLVEuler(self.z0, self.w0, self.alpha2, self.beta2, self.gamma2, self.delta2)
            return self.x0, self.y0, self.z0, self.w0

    def setInitialConditions(self):
        return self.config['x0'], self.config['y0'], self.config['z0'], self.config['w0']

    def setParameters(self):
        return self.config['alpha1'], self.config['beta1'], self.config['gamma1'], self.config['delta1'], \
            self.config['alpha2'], self.config['beta2'], self.config['gamma2'], self.config['delta2']

# for testing:
if __name__ == '__main__':
    dynamics = Dynamics()
    for i in range(0, 5):
        print(dynamics.update())