import numpy as np
from typing import List
from classifier import Classifier

class DecisionStump(Classifier):
	def __init__(self, s:int, b:float, d:int):
		self.clf_name = "Decision_stump"
		self.s = s
		self.b = b
		self.d = d

	def train(self, features: List[List[float]], labels: List[int]):
		pass
		
	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		##################################################
		# TODO: implement "predict"
		##################################################
		features = np.array(features)
		N = features.shape[0]
		D = features.shape[1]
		cur = features[:,self.d]
		pred = self.s * (2 * np.minimum(1, np.ceil(np.maximum(self.b,cur) - self.b))- 1).astype(int)
		
		return pred.tolist()
