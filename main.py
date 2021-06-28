import sys
sys.path.insert(1, "./src/")
from src.Simulator import Simulator

sim = Simulator()
nIterations = sim.config['it']

if sim.config['plotStabilityXY']:
    sim.plotStabilityXY()

# Simulation of a single system
if not sim.config['compareIC']:
    for i in range(nIterations):
            if sim.config['simulateGT']:
                sim.updateGT()
            if sim.config['simulatePipeline']:
                sim.updatePipeline()

# Simulation of the same system with different IC
else:
    sim.simDifferentIC()