import json
import os
import pathlib
import pickle
import numpy as np
from flask import Flask, request
import redis

app = Flask(__name__)
# os.system('docker run -d -p 6379:6379 redis')
cache_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)


def is_redis_available(r):
	try:
		r.ping()
		print("Successfully connected to redis")
	except (redis.exceptions.ConnectionError, ConnectionRefusedError):
		print("Redis connection error, creating new connection!")
		return False
	return True


if not is_redis_available(cache_client):
	os.system('docker run -d -p 6379:6379 redis')
	cache_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
	
model_id = 1
cache_client.set('model_id', model_id)
loaded_model = pickle.load(open(f'featureData/lg_models/{model_id}_lg_model.sav', 'rb'))


def generate_click_probability(data):
	f = open('featureData/feature_column_index_map.json')
	feature_column_index_map = json.load(f)
	feature_vector = [None] * len(feature_column_index_map)
	
	for field, value in data.items():
		if field in feature_column_index_map:
			feature_vector[feature_column_index_map[field]] = value
	if loaded_model is not None:
		feature_vector = np.array(feature_vector).reshape(1, -1)
		prob = loaded_model.predict_proba(feature_vector)[0][0]
		return {'click_probability': prob, 'model_id': model_id}
	else:
		return {'click_probability': None, 'model_id': None}


def model_selection_proc(data):
	global loaded_model
	selected_model_id = data['model_id']
	cache_client.set('model_id', selected_model_id)
	loaded_model = pickle.load(open(f'featureData/lg_models/{selected_model_id}_lg_model.sav', 'rb'))
	return {'model_id': selected_model_id}


def get_selected_model():
	return {'model_id': str(cache_client.get('model_id'))}


@app.route('/click_probability', methods=['POST'])
def click_probability():
	data = json.loads(request.get_json())
	return generate_click_probability(data)


@app.route('/model_selection', methods=['POST'])
def model_selection():
	data = json.loads(request.get_json())
	return model_selection_proc(data)


@app.route('/current_model_id', methods=['GET'])
def current_model_id():
	return get_selected_model()


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
