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



#For next implementation, don't make a separate Perceptron class; vectorize all the inputs; use an enum?

#When writing in C, you don't need enum; just use DEFINE and make it three numbers (ALPHA, W, B) -> (0, 1, 2) respectively

	def __init__(self, alpha: float, w: (float, ...), b: float) -> None:
		self.alpha = alpha
		self.w = w
		self.b = b

class Layer:
	def __init__(self, layer: tuple(Perceptron, ...)) -> None:
		self.layer = layer

		#Making weights random
		for i in range(len(self.layer)):
			self.layer[i] = Perceptron(0, r.random(), 0) #Note; later, when making random weights, implement Xavier/Glorot
	def calc_curr_layer(self, prev: Layer, sigma_func: float):
		#Note: Sigma_func is a cost function
		for i in range(len(self.layer)):
			curr_p = self.layer[i]
			curr_p.alpha += curr_p.b
			for j in range(len(prevLayer.layer)):
				curr_p.alpha += prevLayer[j].w * sigma(prevLayer[j].alpha)



class LayerList:
	def __init__(self, nn: [Layer, ...], sigma_func: float, cost_func: float) -> None:
		self.nn = nn
		self.sigma_func
		self.cost_func = cost_func	
	def flatten(self, sample):
		pass

	def feedforward(self, sample) -> None:
		flatten(sample) #Define later
		if (len(sample) != len(self.nn[0].layer)): 
			#Could probably change this so if input layer is smaller than flattened sample use that 
			print("Sample doesn't match input layer size!")
			return
		
		for i in range(1, len(self.nn)):
			self.nn.layer[i].calc_curr_layer(self.nn.layer[i-1])	
