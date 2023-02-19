# Imports:

# PyTorch imports:
import torch.nn as nn

# Collections imports:
from collections import OrderedDict

class ModuleFactory():

    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)

        return cls.instance


    def create_module(self, module_key : str, module_params : dict):
        """
        []

        Parameters:
        -----------

        Returns:
        --------

        """
        # Handling the case where the module name is not supported by the create_module
        # function:
        allowed_modules = ['Linear', 'BatchNorm1d', 'Dropout', 'LeakyReLU']
        assert module_key in allowed_modules, 'Unkown module name, should be in ' + allowed_modules

        module = None
        module_name = None

        if module_key == 'Linear':
            module = nn.Linear(**module_params)
            module_name = 'fully_connected'

        elif module_key == 'BatchNorm1d':
            module = nn.BatchNorm1d(**module_params)
            module_name = 'batch_norm'

        elif module_key == 'Dropout':
            module = nn.Dropout(**module_params)
            module_name = 'dropout'

        elif module_key == 'LeakyReLU':
            module = nn.LeakyReLU(**module_params)
            module_name = 'leaky_relu'

        return module, module_name


    def create_layer(self, layer_dict : dict):
        """

        Parameters:
        -----------
        layer_dict (dict):

        Returns:
        --------
        layer (torch.nn.Module):


        """

        modules = []
        module_names = []
        # Creating the layer:
        for module_key, module_params in layer_dict.items():

            # Creating the module:
            module, module_name = self.create_module(module_key, module_params)

            modules.append(module)
            module_names.append(module_name)

        layer = nn.Sequential(OrderedDict(zip(module_names, modules)))

        return layer

