import numpy as np
import scipy
from .sample import stopping_time

def seq_power(sample, alpha, beta):
  val, st = stopping_time(sample, alpha, beta)
  if st < len(sample):
    # uniform random variable in [0, 1]
    u = np.random.uniform(0, 1)
    if u < val:
      return val, 1, st
    else:
      return val, 0, st
  elif st == len(sample):
    if val > scipy.stats.norm.ppf(1-alpha):
      return val, 1, st
    else:
      return val, 0, st
    
def std_power(sample, alpha):
  val = np.mean(sample) * np.sqrt(len(sample))
  if val > scipy.stats.norm.ppf(1-alpha):
    return val, 1, len(sample)
  else:
    return val, 0, len(sample)