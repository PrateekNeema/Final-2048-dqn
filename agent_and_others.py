import game2048
import Dddqn_network

import numpy as np 
import random

import torch
from torch import nn

import pickle
import matplotlib.pyplot as plt


class agent:

	def __init__(self,agent_parameters_dict):


		self.base_epsilon=agent_parameters_dict['base_epsilon']
		self.gamma=agent_parameters_dict['gamma']                   #discount rate
		self.learning_rate=agent_parameters_dict['learning_rate']
		self.batch_size=agent_parameters_dict['batch_size']
		self.optimise_local_after=agent_parameters_dict['optimise_local_after']
		self.replay_capacity=agent_parameters_dict['replay_capacity']
		self.copy_local_to_target = agent_parameters_dict['copy_local_to_target']
		self.num_to_train_in_set= agent_parameters_dict['num_to_train_in_set']

		self.total_episodes_trained = 0
		self.epsilon = 1.0

		self.array_of_scores =np.zeros(100)
		self.array_of_average_scores = np.array([])

		self.df_every_epi = pd.DataFrame(np.empty((10000000,3),dtype='uint16'))

		#Initialising models of the networok
		self.local_network =  Dddqn_network.NeuralNetwork()
		self.target_network =  Dddqn_network.NeuralNetwork()

		#Initialising replay memory
		self.my_replay_memory = replay_memory(self.replay_capacity)

		#Initialising optimiser
		self.optimizer = torch.optim.SGD(self.local_network.parameters(), self.learning_rate)


	def choose_action(self,state):

		#reducing epsilon till base_epsilon
		if self.epsilon <=self.base_epsilon:
			epsilon = self.base_epsilon
		else:
			self.epsilon -=0.001

		#epsilon_greedy
		if random.random() < self.epsilon :
			return random.randint(0,3)
		else:
			return torch.argmax(self.target_network.forward(torch.tensor(state)))

	def learn(self):

		sample = self.my_replay_memory.sample(self.batch_size)     #get a list of sample tuples

		loss_function= nn.MSELoss()

		for i in range(0,len(sample)):               #update local netwrok with each experience

			#calculate from expression
			qa_value_from_calculation = sample[i][2] + self.gamma * torch.max(self.target_network.forward(torch.tensor(sample[i][3]))) *(1-sample[i][4])

			#from netwrok
			qa_value_from_target = self.target_network.forward(torch.tensor(sample[i][0]))[sample[i][1]]

			loss= loss_function(qa_value_from_calculation,qa_value_from_target)

			#optimise the loss
			self.optimizer.zero_grad()
			loss.backward()                               
			self.optimizer.step()

		#end for

	def save_average_score(self,array_of_scores,index_of_score):

		self.array_of_average_scores[index_of_score] = np.mean(array_of_scores)
		print("Index ", index_of_score, " : Score : " , np.mean(array_of_scores) )



class replay_memory:

	def __init__(self,capacity):
		self.capacity=capacity

		self.replays = []                   #an empty list
		
		self.oldest_index=0
		self.number_of_replays_stored = 0

	def store(self,transition_tuple):

		if self.number_of_replays_stored < self.capacity:
			self.replays.append(transition_tuple)
			self.number_of_replays_stored +=1

		else:
			self.replays[self.oldest_index] = transition_tuple

			if self.oldest_index == (self.capacity-1):
				self.oldest_index = 0
			else:
				self.oldest_index +=1

	def sample(self,batch_size):

		batch_length = batch_size if self.number_of_replays_stored >= batch_size else self.number_of_replays_stored

		indexes_of_tuples = random.sample(range(0,self.number_of_replays_stored),batch_length)

		return [self.replays[index] for index in indexes_of_tuples]       #returns list of tuples



def train(env,agt):

	for epi_num in range(0,agt.num_to_train_in_set):

		agt.current_state = env.reset()                   #this is in the board form

		while env.isend() == False:

			#select action with epsilon-greedy
			agt.action = agt.choose_action(agt.current_state)

			#do action in game and observe reward and next state
			agt.next_state , agt.reward, agt.done ,agt.info = env.step(agt.action)

			#store this transition in replay memory
			agt.transition_tuple = (agt.current_state , agt.action , agt.reward , agt.next_state , agt.done)
			agt.my_replay_memory.store(agt.transition_tuple)

			
		#end while

		agt.df_every_epi.loc[epi_num,0] = env.score
		agt.df_every_epi.loc[epi_num,1] = env.highest()
		agt.df_every_epi.loc[epi_num,2] = sum(agt.current_state[0]) + sum(agt.current_state[1]) + sum(agt.current_state[2]) + sum(agt.current_state[3])


		#after certain number of episodes optimise the local network parameters
		if epi_num%agt.optimise_local_after == 0:
			agt.learn()

		#after certain numbe rof eposides copy the values from loacal to target network
		if epi_num % agt.copy_local_to_target == 0:
			agt.target_network.load_state_dict(agt.local_network.state_dict())

		
		#Every 100 episodes,store the average score of those 100 episodes in an array and render the board and values
		if epi_num% 100 == 0:                                                            #1000 value can be changed
			agt.save_average_score(agt.array_of_scores,int(epi_num/100) -1 )
			print( "For ",int(epi_num/100) , " set of 100 episodes,average score is ", np.mean(agt.array_of_scores))
		else:
			agt.array_of_scores[epi_num%100-1] = env.score


	#end for


	agt.total_episodes_trained += agt.num_to_train_in_set
	
	save_model(agt)                 #This is when i train fully for some decided time then save model




def save_model(agt):                      

	pickle_filename = "Pickle_RL_Model.pkl"         #saving only the agent as it has all the inoformation

	with open(pickle_filename, 'wb') as file:
		pickle.dump(agt, file)

def load_model():

	pickle_filename = "Pickle_RL_Model.pkl" 
 
	with open(pickle_filename, 'rb') as file:  
		loaded_agt = pickle.load(file)

	return loaded_agt


def perform(env,dqn_network):

	current_state = env_obj.reset()      #start new game

	env.render()

	while env_obj.isend() == False :          #exit when game over

		action = torch.argmax(dqn_network.forward(current_state))   #choose action according to q values

		next_state ,reward,done,info = env.step(action)      #do action in game

		env_obj.render()

		current_state =next_state

	#end while (game)


def plot_scores(agt):

	epi_per_100 = np.arange(0,int(agt.total_episodes_trained/100))

	scores = agt.array_of_average_scores[0:int(agt.total_episodes_trained/100)]

	plt.plot(epi_per_100,scores)

	plt.xlabel("Number of episodes(per 100)")
	plt.ylabel("Average Score")












