import sys
sys.path.insert(1, "./src/")
from src.ConfigLoader import ConfigLoader
from src.Dynamics import Dynamics

config = ConfigLoader()
model = Dynamics()
nIterations = config.params['it']

for i in range(nIterations):
    model.update()