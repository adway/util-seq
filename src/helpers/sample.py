import numpy as np
import scipy

def make_sample(N, theta):
  """Make a sample of size N from a normal(theta, 1) distribution."""
  return np.random.normal(theta, 1, N)

def cumulative_sum(sample):
  """Return the cumulative sum of the sample."""
  return np.cumsum(sample)

def stopping_time(sample, alpha, beta):
  """Return the stopping time and test statistic for the sample."""
  cumulative = cumulative_sum(sample)
  stopping_time = sample.size
  for i in range(len(cumulative)):
    boundary = np.sqrt(sample.size) * scipy.stats.norm.ppf(1 - alpha) + np.sqrt(sample.size - i) * scipy.stats.norm.ppf(beta)
    if cumulative[i] > boundary:
      stopping_time = i + 1
      val = scipy.stats.norm.cdf((cumulative[i] - np.sqrt(sample.size) * scipy.stats.norm.ppf(1 - alpha)) / np.sqrt(sample.size - i))
      return val, stopping_time
  return cumulative[-1] / np.sqrt(sample.size), stopping_time