import tensorflow as tf
print('TensorFlow version:', tf.__version__)

from prophet import Prophet
print('Prophet is installed')

import pmdarima as pm
print('pmdarima version:', pm.__version__)

import statsmodels
print('statsmodels version:', statsmodels.__version__)

print('\nAll packages successfully imported!') 