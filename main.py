import game2048
import Dddqn_network
import agent_and_others


#env=game2048.Game2048Env()

agent_params={
	'base_epsilon' : 0.1,
	'gamma' : 0.99,
	'learning_rate': 0.8,
	'batch_size': 64,
	'optimise_local_after' : 100,
	'replay_capacity' :1000,
	'copy_local_to_target' : 500,
	'num_to_train_in_set' : 1000
}



#agt = agent_and_others.agent(agent_params)

#agent_and_others.train(env,agt)

#agt = agent_and_others.load_model()

#print(agt.array_of_average_scores[0:20])

#print(agt.total_episodes_trained)







