import math as m
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as r
"""

#For now, will stick to basics and probably not use numpy; will be using regular random libraries
 -- after a while will use pytorch and others to speed up the process

General implementation:
1. class Layer (perceptron activations, next layer weights, matrix-mul operations to pass weights to next)
2. class LayerList (all layers as continuous values; implements Layer for forward prop and also performs back-prop; also weight adjustment)

"""


"""

Notes for self-improvement:
1. You could probably package all of these into the LayerList class; general theme is that this would allow you to exploit cache localization
 - This will make more sense when implementing NumPy later, which allows you to perform operations in-bulk
   ~ Storing all weights of all the matrices in the layer directly and just performing a matrix multiplication accordingly
"""


#For next implementation, don't make a separate Perceptron class; vectorize all the inputs; use an enum?

#When writing in C, you don't need enum; just use DEFINE and make it three numbers (ALPHA, W, B) -> (0, 1, 2) respectively
class Perceptron:
	def __init__(self, z: float, W: (float, ...), b: float) -> None:
		self.z = z #preactivation
		self.W = W
		self.b = b
	def set_activation(self, sigma_func):
		self.a = sigma_func(self.z)

class Layer:
	def __init__(self, layer: [Perceptron, ...]) -> None:
		self.layer = layer
	def calc_curr_layer(self, prev: Layer, sigma_func: float) -> None:
		for i in range(self.layer):
			self.layer[i].z = self.layer[i].b
			for j in range(len(prev.layer)):
				self.layer[i].z += prev.layer[j].a * prev.layer[j].W[i]
			self.layer[i].set_activation(self, sigma_func)



class CostGrad:
	
	def __init__(self) -> None:
		self.W = self.b = []
		# Note: pretty inefficient but will basically be implementing a stack; 
		# Better way of doing this would to make your own machine learning library (or numpy), pass in dimensions, and then make a list of those exact same dimensions
	
	def important_partial(self, a_i: float, y_i: float, n: int):
		# Got a little lazy here; there are probably other ways to approximate loss, so justm make sure you learn and get acquainted with those
		return (a_i - y_i)/n
		

class LayerList:
	def __init__(self, nn: [Layer, ...], cost_func: float) -> None:
		self.nn = nn
		self.cost_func = cost_func
		self.cost_grad = None

	def flatten(self, sample):
		pass

	def feedforward(self, sample) -> None:
		flatten(sample) #Define later
		if (len(sample) != len(self.nn[0].layer)): 
			#Could probably change this so if input layer is smaller than flattened sample use that 
			print("Sample doesn't match input layer size!")
			return

		for (i in range(1, len(self.nn))):
			self.nn[i].calc_curr_layer(self.nn[i-1], sigma_func)
	
	def find_grad(self) -> None:
		self.cost_grad = CostGrad()
		pass	
						
		

	def backprop(self, eta: float):
		#eta is the learning rate

