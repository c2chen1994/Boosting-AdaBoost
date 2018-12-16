import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################
		featuresArray = np.array(features)
		N = featuresArray.shape[0]
		res = np.zeros(N)

		for t in range(0, self.T):
			res += self.betas[t] * np.array(self.clfs_picked[t].predict(features))
		
		return (2 * np.maximum(0, np.floor(np.minimum(0, res)) + 1) - 1).astype(int).tolist()



class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################
		featuresArray = np.array(features)
		labels = np.array(labels)
		N = featuresArray.shape[0]
		Dn = np.zeros(N)
		Dn = Dn + 1.0 / N
		cList = []
		classifiers = []
		for c in self.clfs:
			classifiers.append(c)
			cList.append((np.abs(labels - np.array(c.predict(features))) / 2).tolist())
		cListArray = np.array(cList)

		for iter in range(0, self.T):
			a = np.sum(cListArray * np.tile(Dn, (len(self.clfs), 1)), axis = 1)
			index = np.argsort(a)[0]
			et = a[index]
			self.clfs_picked.append(classifiers[index])
			self.betas.append(0.5 * np.log((1 - et) / et)) 
			Dn = Dn * np.exp((2 * cListArray[index] - 1) * self.betas[iter])
			sumDn = np.sum(Dn)
			Dn = Dn / sumDn
			

	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	