import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
dataset = MNIST(root="./data", train=True, download=True, transform=ToTensor())
loader = DataLoader(dataset, batch_size=64, shuffle=True) 

num_epochs = 100

"""

Interesting remarks:
1. We cannot use a step activation because they don't allow for gradient descent; zero-derivative except for origin, where it is undefined
 - This is one of the reason sigmoid is popular; it approximates this distribution quite well
  ~ Remember: Idea behind sigmoid is to sort of have a derivative that is integral -infinity to x of e^-t^2 dt (from nn book by 3b1b)

2. If you were to build this on your own, make sure you wrote the program sort of like this; you would, ideally, want the functions to be mapped out like this; then, you can build the structs, functions, etc... that implement all the linear algebra stuff

3. You should never call self.forward() directly, becasue that bypasses the PyTorch registry

"""

class Perceptron(nn.Module):
	def __init__(self, input_size, output_size):
		super().__init__()
		self.linear = nn.Linear(input_size, output_size)
	
	def forward(self, sample):
		z = self.linear(sample)
		return torch.sigmoid(z)


class PerceptronList(nn.Module):
	def __init__(self, sizes): #Note: output layer connects directly to the nn.ReLU
		super().__init__()
		self.layers = nn.Sequential(*[Perceptron(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
	def forward(self, sample):
		return self.layers(sample)
	def calc_loss(self, sample, loss, target):
		return loss(self(sample), target)

model = PerceptronList([784, 8, 8, 10]) #How do we find the number 784 just using the data set, if I didn't know it's contents beforehand?
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()


for epoch in range(num_epochs):
	for images, labels in loader:
		x = images.flatten(start_dim=1)
		y = nn.functional.one_hot(labels, num_classes=10).float() #Why do we need to make this a float; what does one_hot encoder do? why couldn't we have just used the 'labels', which is given to us?

		optimizer.zero_grad()
		prediction = model(x)
		loss = loss_fn(prediction, y)
		loss.backward()
		optimizer.step()


#saving the model:
torch.save(model.state.dict(), "model.pth") #What is .pth file extension?

