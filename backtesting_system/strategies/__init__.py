# # strategies/__init__.py

# from .buy_and_hold import BuyAndHold
# from .sma_crossover import SMACrossOver
# from .sma_crossover_with_rsi import SMA_CrossOver_with_Indicators

# __all__ = ['BuyAndHold', 'SMACrossOver', 'SMA_CrossOver_with_Indicators']


import os
import importlib

# Get the directory path for the current module (strategies)
strategies_dir = os.path.dirname(__file__)

# List to hold strategy class names
__all__ = []

# Loop through each file in the directory
for file in os.listdir(strategies_dir):
    if file.endswith('.py') and file != '__init__.py':  # Exclude __init__.py
        module_name = file[:-3]  # Strip the .py extension
        module = importlib.import_module(f'.{module_name}', package='strategies')
        
        # Import each class from the module and assign it to globals()
        for attr in dir(module):
            attr_obj = getattr(module, attr)
            if isinstance(attr_obj, type):  # Check if it's a class
                globals()[attr] = attr_obj  # Add to global namespace
                __all__.append(attr)  # Add class name to __all__
