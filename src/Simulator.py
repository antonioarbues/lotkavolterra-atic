from Estimator import Estimator
from Model import Model
from ConfigLoader import ConfigLoader
from Plotter import Plotter
from FindEquilibria import FindEquilibria
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
            usecontrol=True)
        self.model.x0, self.model.y0, self.model.z0, self.model.w0 = \
             self.model.updateRK4(self.model.x0, self.model.y0, self.model.z0, self.model.w0, self.model.a, self.model.b, \
                 processnoise=False, measurementnoise=False, simulatornoise=False, usecontrol=True)
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
            usecontrol=True)
        # Estimation and Control feedback
        u = self.modelP.sigma * (-np.matmul(np.transpose(self.modelP.kp), (self.estimator.x00 - self.modelP.e)))
        x00 = self.estimator.update(u)
        self.modelP.x0 = x00[0]
        self.modelP.y0 = x00[1]
        self.modelP.z0 = x00[2]
        self.modelP.w0 = x00[3]
        if self.config['showControlInputs'] and self.config['useControl']:
            plotBuffer, derivativePlotBuffer = self.plotterP.updatePlotterBuffer(self.modelP.x0, self.modelP.y0, self.modelP.z0, self.modelP.w0, self.modelP.dx, self.modelP.dy, \
                 self.modelP.dz, self.modelP.dw, self.modelP.u)
        else:
            plotBuffer, derivativePlotBuffer = self.plotterP.updatePlotterBuffer(self.modelP.x0, self.modelP.y0, self.modelP.z0, self.modelP.w0, self.modelP.dx, self.modelP.dy, self.modelP.dz, self.modelP.dw)
        return plotBuffer, derivativePlotBuffer
    
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
    
    def simDifferentIC(self):
        plotBuffer = [[], [], [], []]
        nIterations = self.config['it']
        for i in range(1, self.config['n_IC'] + 1):
            self.model.config['x0'] = self.config['x' + str(i)]
            self.model.config['y0'] = self.config['y' + str(i)]
            self.model.config['z0'] = self.config['z' + str(i)]
            self.model.config['w0'] = self.config['w' + str(i)]
            self.model.setParametersAgain()
            for i in range(nIterations):
                if i != nIterations-1:
                    if self.config['simulateGT']:
                        self.updateGT()
                    elif self.config['simulatePipeline']:
                        self.updatePipeline()
                else:
                    if self.config['simulateGT']:
                        plotBufferIteration, _ = self.updateGT()
                    elif self.config['simulatePipeline']:
                        plotBufferIteration, _ = self.updatePipeline()
                    for it in range(len(plotBufferIteration)):
                        plotBuffer[it] += plotBufferIteration[it]
        self.plotter.plotstatespace4(plotBuffer[0], plotBuffer[1],\
            plotBuffer[2], plotBuffer[3])
        self.plotter.plot4dLimitcycle(plotBuffer[0], plotBuffer[1],\
            plotBuffer[2], plotBuffer[3])

if __name__ == '__main__':
    sim = Simulator()
    nIterations = sim.config['it']
    print(sim.model.equilibria.findEquilibria())
    for i in range(nIterations):
        sim.updateGT()
        sim.updatePipeline()