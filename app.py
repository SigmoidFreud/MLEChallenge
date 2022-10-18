import json
import pathlib
import pickle
import numpy as np
from flask import Flask, request

app = Flask(__name__)
model_id = 1
model = pickle.load(open(f'featureData/lg_models/{model_id}_lg_model.sav', 'rb'))


@app.route('/click_probability', methods=['POST'])
def click_probability():
	global model
	global model_id
	data = json.loads(request.get_json())
	f = open('featureData/feature_column_index_map.json')
	feature_column_index_map = json.load(f)
	feature_vector = [None] * len(feature_column_index_map)
	
	for field, value in data.items():
		if field in feature_column_index_map:
			feature_vector[feature_column_index_map[field]] = value
	if model is not None:
		feature_vector = np.array(feature_vector).reshape(1, -1)
		prob = model.predict_proba(feature_vector)[0][0]
		return {'click_probability': prob, 'model_id': model_id}
	else:
		return None


@app.route('/model_selection', methods=['POST'])
def model_selection():
	global model, model_id
	data = json.loads(request.get_json())
	selected_model_id = data['model_id']
	print(selected_model_id)
	model_id = selected_model_id
	model = pickle.load(open(f'featureData/lg_models/{model_id}_lg_model.sav', 'rb'))
	return {'model_id': model_id}


@app.route('/current_model_id', methods=['GET'])
def current_model_id():
	global model_id
	print(model_id)
	return {'model_id': model_id}


@app.route('/available_model_ids', methods=['GET'])
def available_model_ids():
	global model_id
	model_ids = []
	for lg_model in pathlib.Path('featureData/lg_models').glob('*.sav'):
		lg_model_filename = str(lg_model)
		f_name = lg_model_filename.split('\\')[2]
		model_id = f_name.split('_')[0]
		model_ids.append(model_id)
	return {'model_ids': model_ids}


if __name__ == '__main__':
	app.run()
