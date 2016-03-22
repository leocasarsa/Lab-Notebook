import sys
sys.path.append('/home/casarsa/Git/gpmcc/')

import numpy as np
import time
import pickle
import unittest

from sigma_cmi import automatic_Sigma, analytic_cmi
from scipy.stats import norm
from gpmcc.engine import Engine


def dag_and_noise(scaleX=1,scaleY=1,scaleZ=1):
  dag = np.array([[0,0,0],
                  [1,0,0],
                  [0,1,0]])
  noise = np.array([scaleX, scaleY, scaleZ])
  return dag, noise

def simulate_X(N, loc=0, scale=1):
  return norm.rvs(loc=loc, scale=scale , size=[N,1])

def simulate_Y_gX(N, gX, loc=0, scale=1):
  assert gX.shape[0]==N
  return gX + norm.rvs(loc=loc, scale=scale , size=[N,1])

def simulate_Z_gY(N, gY, loc=0, scale=1):
  assert gY.shape[0]==N
  return gY + norm.rvs(loc=loc, scale=scale , size=[N,1])

def simulate_chain(N, scaleX=1, scaleY=1, scaleZ=1):
  X = simulate_X(N, scale=scaleX)
  Y = simulate_Y_gX(N,X, scale=scaleY)
  Z = simulate_Z_gY(N,Y, scale=scaleZ)
  return np.hstack((X,Y,Z))

def get_thresholds_cmi(N_samples, N_repeats):
  col_num = dict({'X':0, 'Y':1, 'Z':2})
	
  dag, noise = dag_and_noise()
  auto_sigma = automatic_Sigma(dag, noise)

  # values analytically obtained from generative model (not unit tested)	
  mi_XZ_gen = analytic_cmi(auto_sigma, col_num['X'], col_num['Z'], [])
  cmi_XZ_gY_gen = analytic_cmi(auto_sigma, col_num['X'], col_num['Z'], [col_num['Y']])

  mi_XZ_emp = np.array([])
  cmi_XZ_gY_emp = np.array([])
  for i_repeat in range(N_repeats):
    data = simulate_chain(N_samples)
    emp_sigma = np.cov(data.T)

    mi_XZ_emp = np.append(mi_XZ_emp, analytic_cmi(emp_sigma, col_num['X'], col_num['Z'], []))
    cmi_XZ_gY_emp = np.append(cmi_XZ_gY_emp, analytic_cmi(emp_sigma, col_num['X'], col_num['Z'], [col_num['Y']]))

  return mi_XZ_emp, cmi_XZ_gY_emp

print get_thresholds_cmi(100,100)


# class TestGetThreshold(unittest.testcase):
  

# class TestCmi(unittest.TestCase):

# 	def test_gen_mi_XZ_nonzero(self):
# 		self.assertNotAlmostEqual(mi_XZ_gen, 0)

# 	def test_gen_cmi_XZ_gY_zero(self):
# 		self.assertAlmostEqual(cmi_XZ_gZ_gen, 0)

# 	def test_mi_XZ_gen_gpmcc(self):
# 		self.assertAlmostEqual(mi_XZ_gen, mi_XZ_gpmcc)

# 	def test_cmi_XZ_gY_gen_gpmcc(self):
# 		self.assertAlmostEqual(cmi_XZ_gZ_gen, cmi_XZ_gZ_gpmcc)

# if __name__ == '__main__':
#     unittest.main()