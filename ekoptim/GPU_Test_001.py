import tensorflow as tf
from autograd_minimize import minimize
import numpy as np

def rosen_tf(x):
    return tf.reduce_sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

res = minimize(rosen_tf, np.array([0.,0.]))
print(res.x)