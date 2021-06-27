from Controller import Controller
from Estimator import Estimator
from Model import Model
from ConfigLoader import ConfigLoader
from Plotter import Plotter
import numpy as np

class Simulator:
    def __init__(self):
        config = ConfigLoader()
        self.config = config.params
        self.model = Model()  # model to use for the GT update
        self.modelP = Model() # model to use for the pipeline update
        self.estimator = Estimator()
        self.plotter = Plotter()
        self.plotterP = Plotter()
    
    def updateGT(self):
        '''
        returns the updated x1, y1, z1, w1 (ground truth)
        '''
        self.model.dx, self.model.dy, self.model.dz, self.model.dw = self.model.f(self.model.x0, self.model.y0, self.model.z0, self.model.w0, self.model.a, self.model.b, \
            processnoise=False, measurementnoise=False, usecontrol=True)
        self.model.x0, self.model.y0, self.model.z0, self.model.w0 = \
             self.model.updateRK4(self.model.x0, self.model.y0, self.model.z0, self.model.w0, self.model.a, self.model.b, \
                 processnoise=False, measurementnoise=False, usecontrol=True)
        if self.config['showControlInputs'] and self.config['useControl']:
            plotBuffer, derivativePlotBuffer = self.plotter.updatePlotterBuffer(self.model.x0, self.model.y0, self.model.z0, self.model.w0, self.model.dx, self.model.dy, \
                 self.model.dz, self.model.dw, self.model.u)
        else:
            plotBuffer, derivativePlotBuffer = self.plotter.updatePlotterBuffer(self.model.x0, self.model.y0, self.model.z0, self.model.w0, self.model.dx, self.model.dy, self.model.dz, self.model.dw)
        return plotBuffer, derivativePlotBuffer

    def updatePipeline(self):
        '''
        returns the updated x1, y1, z1, w1 (computed with the estimator in the loop)
        ''' 
        self.modelP.dx, self.modelP.dy, self.modelP.dz, self.modelP.dw = self.modelP.f(self.modelP.x0, self.modelP.y0, self.modelP.z0, self.modelP.w0, self.modelP.a, self.modelP.b, \
            processnoise=False, measurementnoise=False, usecontrol=True)
        # Estimation and Control feedback
        u = self.modelP.sigma * (-np.matmul(np.transpose(self.modelP.kp), (self.estimator.x00 - self.modelP.e)))
        self.modelP.x0, self.modelP.y0, self.modelP.z0, self.modelP.w0 = \
            self.estimator.update(self.modelP.u)
        if self.config['showControlInputs'] and self.config['useControl']:
            plotBuffer, derivativePlotBuffer = self.plotterP.updatePlotterBuffer(self.modelP.x0, self.modelP.y0, self.modelP.z0, self.modelP.w0, self.modelP.dx, self.modelP.dy, \
                 self.modelP.dz, self.modelP.dw, self.modelP.u)
        else:
            plotBuffer, derivativePlotBuffer = self.plotterP.updatePlotterBuffer(self.modelP.x0, self.modelP.y0, self.modelP.z0, self.modelP.w0, self.modelP.dx, self.modelP.dy, self.modelP.dz, self.modelP.dw)
        return plotBuffer, derivativePlotBuffer

if __name__ == '__main__':
    sim = Simulator()
    nIterations = sim.config['it']
    for i in range(nIterations):
        # sim.updateGT()
        sim.updatePipeline()