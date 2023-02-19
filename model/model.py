# Imports:

# PyTorch imports:
import torch.nn as nn

# ModuleFactory imports:
from module_factory import ModuleFactory

# Other imports:
import os
import oyaml

# Defining the default path for the model configuration .yaml file:
DEFUALT_MODEL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_config.yaml')


class Model(nn.Module):
    def __init__(self, config_path=DEFUALT_MODEL_CONFIG_PATH):
        super(Model, self).__init__()
        # Loading the model's configuration:
        self.model_config = self.load_config()

        # --- Creating the Model: ------------------------------

        # Creating an instance of the ModuleFactory:
        module_factory = ModuleFactory()

        layer_names = []
        layers = []

        for layer_name, layer_config in self.model_config.items():
            # Creating the layer from the configuration file using the
            # ModuleFactory:
            layer = module_factory.create_layer(layer_config)

            layers.append(layer)
            layer_names.append(layer_name)

        self.model = nn.ModuleDict(zip(layer_names, layers))

        # --- Weights and Biases Initialization: ---------------
        self.initialize_weights_and_biased()


    def initialize_weights_and_biased(self):
        """
        Initializes the weights and biases of all the modules that have parameters:

        """
        has_params = lambda module: True if (len(list(module.parameters()))) > 0 else False
        for _, layer in self.model.items():
            for module in layer:
                if has_params(module):
                    if isinstance(module, nn.BatchNorm1d):
                        # Initializing the weights of the batch norms to ones and
                        # the biases to zeroes:
                        nn.init.ones_(module.weight)
                        nn.init.zeros_(module.bias)
                    else:
                        nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
                        nn.init.constant_(module.bias, 0.0)

    def load_config(self):
        """
        Loads the configuration .yaml file located in the self.config_path.

        Returns:
        --------
        model_config (dict):
            A dictionary defining the configuration of the model.

        """
        with open(DEFUALT_MODEL_CONFIG_PATH, 'r') as file:
            model_config = oyaml.safe_load(file)

        return model_config
