import os
import torch
from torch import nn
import torch

class NeuralNetwork(nn.Module):

	def __init__(self):

		super(NeuralNetwork, self).__init__()

		self.stack_first_few_layers =nn.Sequential(

			nn.Linear(16,100),                          #Layer 1
			nn.ReLU(),                                  #Layer 2
			nn.Linear(100,200),                         #Layer 3
			nn.ReLU()                                   #Layer 4
		)

		
		self.V_specific_layers =nn.Sequential(

			nn.Linear(200,4),                          #V_specific_layer 5
			nn.ReLU()                                   #V_specific_layer 6
		)

		self.A_specific_layers =nn.Sequential(

			nn.Linear(200,4),                           #A_specific_layer 5
			nn.ReLU()                                   #A_specific_layer 6
 
		)

		self.layer_7 = nn.ReLU()


	def forward(self,s):                                   #one dimensional tensor of size 16 is input an dthta of size 4 is output

		#print(s)
		s = torch.flatten(s)
		s = s.float()
		#print(s)

		out= self.stack_first_few_layers(s)

		V=self.V_specific_layers(out)
		A=self.A_specific_layers(out)

		Q = V + (A-torch.mean(A))

		Q = self.layer_7(Q)

		return Q




