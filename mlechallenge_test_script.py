import pathlib
import requests
import json


# call get service with headers and params
def get_click_probability_test(endpoint, url='http://127.0.0.1:5000'):
	with open('featureData/generated_test_data.json') as f:
		output_list = []
		
		test_data = json.load(f)
		url = f'{url}/{endpoint}'
		for feature_data in test_data:
			feature_data_dict = feature_data
			feature_data = json.dumps(feature_data)
			# print(feature_data)
			response = requests.post(url, json=feature_data)
			output_json = {'tested_endpoint': endpoint, 'input': feature_data_dict, 'output': response.json()}
			output_list.append(output_json)
		return output_list


def model_id_test(selected_model_id, url='http://127.0.0.1:5000'):
	output_list = []
	selection_endpoint = 'model_selection'
	verification_endpoint = 'current_model_id'
	selection_url = f'{url}/{selection_endpoint}'
	verification_url = f'{url}/{verification_endpoint}'
	selection_json = {'model_id': selected_model_id}
	selection_json = json.dumps(selection_json)
	selection_response = requests.post(selection_url, json=selection_json)
	selection_output_json = {'tested_endpoint': selection_endpoint,
	                         'input': selected_model_id,
	                         'output': selection_response.json()}
	output_list.append(selection_output_json)
	verification_response = requests.get(verification_url)
	verification_output_json = {'tested_endpoint': verification_endpoint,
	                            'input': None,
	                            'output': verification_response.json()}
	output_list.append(verification_output_json)
	return output_list


def available_model_ids(model_availability_endpoint, url='http://127.0.0.1:5000'):
	model_availability_url = f'{url}/{model_availability_endpoint}'
	model_availability_response = requests.get(model_availability_url)
	model_availability_output_json = {'tested_endpoint': model_availability_endpoint,
	                                  'input': None,
	                                  'output': model_availability_response.json()}
	return model_availability_output_json


def fully_test_endpoints():
	full_output_test = [available_model_ids('available_model_ids')]
	for lg_model in pathlib.Path('featureData/lg_models').glob('*.sav'):
		lg_model_filename = str(lg_model)
		f_name = lg_model_filename.split('\\')[2]
		model_id = f_name.split('_')[0]
		full_output_test.extend(model_id_test(model_id))
		full_output_test.extend(get_click_probability_test('click_probability'))
	with open('fully_tested_endpoint_data.json', 'w') as f:
		json.dump(full_output_test, f, indent=4)


if __name__ == '__main__':
	fully_test_endpoints()
