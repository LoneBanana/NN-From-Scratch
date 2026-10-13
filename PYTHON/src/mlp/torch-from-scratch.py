import numpy as np
import torch
from torch import nn

"""

Interesting remarks:
1. We cannot use a step activation because they don't allow for gradient descent; zero-derivative except for origin, where it is undefined
 - This is one of the reason sigmoid is popular; it approximates this distribution quite well
  ~ Remember: Idea behind sigmoid is to sort of have a derivative that is integral -infinity to x of e^-t^2 dt (from nn book by 3b1b)

2. If you were to build this on your own, make sure you wrote the program sort of like this; you would, ideally, want the functions to be mapped out like this; then, you can build the structs, functions, etc... that implement all the linear algebra stuff

"""

class Perceptron(nn.Module):
	def __init__(self, input_size, output_size):
		self.linear = nn.Linear(layer_size, output_size)
	
	def forward(self, sample):
		z = self.linear(sample)
		return torch.sigmoid(z)


class PerceptronList(nn.Module):
	def __init__(self, sizes): #Note: output layer connects directly to the nn.ReLU
		super().__init__()
		self.layers = nn.Sequential(*[Perceptron(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)
	def forward(self, sample):
		return self.layers(sample)
	def calc_loss(self, sample, loss):
		return loss(forward(sample))
	def backward(self, loss):
		loss.backward()



