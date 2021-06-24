import sys
sys.path.insert(1, "./src/")
from src.ConfigLoader import ConfigLoader
from src.Dynamics import Dynamics
from src.DynamicsUnited import DynamicsUnited
from src.FindEquilibria import FindEquilibria
from src.Plotter import Plotter

config = ConfigLoader()
model = Dynamics()
modelUnited = DynamicsUnited()
nIterations = config.params['it']
equilibria = FindEquilibria()

equilibria.getEigens()

if config.params['plotStabilityXY'] and config.params['useCoupled']:
    modelUnited.plotStabilityXY()

# Simulation of a single system
if not config.params['compareIC']:
    for i in range(nIterations):
        if config.params['useDecoupled']:
            model.update()
        elif config.params['useCoupled']:
            modelUnited.update()

# Simulation of the same system with different IC
elif config.params['useCoupled']:
    plotBuffer = [[], [], [], []]
    for i in range(1, config.params['n_IC'] + 1):
        modelUnited = DynamicsUnited()
        modelUnited.config['x0'] = config.params['x' + str(i)]
        modelUnited.config['y0'] = config.params['y' + str(i)]
        modelUnited.config['z0'] = config.params['z' + str(i)]
        modelUnited.config['w0'] = config.params['w' + str(i)]
        modelUnited.setParametersAgain()
        for i in range(nIterations):
            if i != nIterations-1:
                modelUnited.update()
            else:
                plotBufferIteration, _ = modelUnited.update()
                for it in range(len(plotBufferIteration)):
                    plotBuffer[it] += plotBufferIteration[it]
    modelUnited.plotter.plotstatespace4(plotBuffer[0], plotBuffer[1],\
         plotBuffer[2], plotBuffer[3])

else:
    print('Cannot simulate decoupled systems with different IC')