import yaml

class ConfigLoader:
    def __init__(self):
        self.params = self.getParamsFromConfig()

    def getParamsFromConfig(self):
        with open(r'./config.yaml') as file:
            return yaml.load(file, Loader=yaml.FullLoader)
