import sys
sys.path.insert(1, "./src/")
from src.ConfigLoader import ConfigLoader
from src.Dynamics import Dynamics
from src.DynamicsUnited import DynamicsUnited

config = ConfigLoader()
model = Dynamics()
modelUnited = DynamicsUnited()
nIterations = config.params['it']

for i in range(nIterations):
    if config.params['useDecoupled']:
        model.update()
    elif config.params['useCoupled']:
        modelUnited.update()