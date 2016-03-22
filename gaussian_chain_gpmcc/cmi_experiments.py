learn_params =  {'num_points': 1000, 'num_states': 10, 'num_transitions': 100}
mi_params = {'num_samples': 1000}
cmi_params = {'num_samples': 1000, 'num_condition': 100}

engine, data, train_time = load_gpmcc_chain(**learn_params)
engine.num_points = learn_params['num_points']
engine.num_transitions = learn_params['num_transitions']
engine.data = data
