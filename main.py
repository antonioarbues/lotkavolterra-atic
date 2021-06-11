import sys
sys.path.insert(1, "./src/")
from src.ConfigLoader import ConfigLoader
from src.Dynamics import Dynamics
from src.DynamicsUnited import DynamicsUnited
from src.FindEquilibrium import FindEquilibria

config = ConfigLoader()
model = Dynamics()
modelUnited = DynamicsUnited()
nIterations = config.params['it']
equilibria = FindEquilibria()

equilibria.findEquilibria()

for i in range(nIterations):
    if config.params['useDecoupled']:
        model.update()
    elif config.params['useCoupled']:
        modelUnited.update()
    