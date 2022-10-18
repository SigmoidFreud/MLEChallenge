import json
import pathlib
import pickle
import random
import sqlite3
import uuid

import numpy as np
from pandas import read_parquet
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression


class NpEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		if isinstance(obj, np.floating):
			return float(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return super(NpEncoder, self).default(obj)

def random_with_n_digits(n):
	range_start = 10**(n-1)
	range_end = (10**n)-1
	return random.randint(range_start, range_end)

def read_data_into_memory_and_process_feature_data(connect, conn_cursor):
	conn_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
	table_names = conn_cursor.fetchall()
	categorical_data_table_map = {'ds_leads': {'credit', 'loan_purpose'}}
	table_names_df_dict = {}
	for table_name in table_names:
		table_name = table_name[0]
		df = pd.read_sql_query(f'SELECT * FROM {table_name}', connect)
		table_names_df_dict[table_name] = df
	
	for table_name in categorical_data_table_map:
		df = table_names_df_dict[table_name]
		process_feature_data_by_table(categorical_data_table_map, table_name, df)
	clicked_offers = table_names_df_dict['ds_clicks']['offer_id'].unique()
	ds_offers_combined = pd.merge(table_names_df_dict['ds_offers'], table_names_df_dict['ds_leads'],
	                              left_on='lead_uuid', right_on='lead_uuid', how='left')
	clicked = []
	for offer_id in ds_offers_combined['offer_id']:
		if offer_id in clicked_offers:
			clicked.append(1)
		else:
			clicked.append(0)
	ds_offers_combined['clicked'] = clicked
	return ds_offers_combined


def generate_sample_test_data(test_data_length, X_train):
	test_data = []
	for i in range(test_data_length):
		sample_features = {}
		for feature in X_train.columns:
			random_num = random.randint(0, len(X_train))
			sample_features[feature] = X_train[feature][random_num]
		sample_features['offer_id'] = random_with_n_digits(9)
		sample_features['lead_uuid'] = str(uuid.uuid4())
		test_data.append(sample_features)
	with open('generated_test_data.json', 'w') as f:
		json.dump(test_data, f, cls=NpEncoder)


def generate_training_data(combined_data, unused_columns_in_training):
	combined_data.drop(unused_columns_in_training, axis=1, inplace=True)
	combined_data = combined_data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
	feature_set = set(combined_data.columns)
	feature_set.remove('clicked')
	training_features = list(feature_set)
	X_train = combined_data[training_features]
	Y_train = combined_data['clicked']
	generate_sample_test_data(10, X_train)
	feature_column_index_map = {column: i for i, column in enumerate(X_train.columns)}
	with open('feature_column_index_map.json', 'w') as f:
		json.dump(feature_column_index_map, f)
	return X_train, Y_train


def generate_lg_model(model_id, X_train, Y_train, penalty="l2",
                      dual=False,
                      tol=1e-4,
                      C=1.0,
                      fit_intercept=True,
                      intercept_scaling=1,
                      class_weight=None,
                      random_state=None,
                      solver="lbfgs",
                      max_iter=100,
                      multi_class="auto",
                      verbose=0,
                      warm_start=False,
                      n_jobs=None,
                      l1_ratio=None):
	model = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
	                           intercept_scaling=intercept_scaling, class_weight=class_weight,
	                           random_state=random_state,
	                           solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose,
	                           warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio)
	model.fit(X_train, Y_train)
	filename = f'lg_models/{model_id}_lg_model.sav'
	pickle.dump(model, open(filename, 'wb'))


def generate_two_lg_models(X_train, Y_train):
	generate_lg_model(1, X_train, Y_train, penalty='l2')
	generate_lg_model(2, X_train, Y_train, penalty='none')


def process_feature_data_by_table(categorical_data_table_map, table_name, table_df):
	for column in categorical_data_table_map[table_name]:
		column_unique_vals = table_df[column].unique()
		table_df[column].replace(column_unique_vals,
		                         range(len(column_unique_vals)), inplace=True)


def extract_data_and_produce_ddl_from_parquet(conn_cursor):
	for parquet_file in pathlib.Path('.').glob('*.parquet.*'):
		parquet_df = read_parquet(parquet_file)
		table_name = str(parquet_file).split('.')[0]
		parquet_df.to_sql(table_name, conn, if_exists="replace", index=False)
	ddl_query_str = f"SELECT sql FROM sqlite_master;"
	conn_cursor.execute(ddl_query_str)
	ddl_str = conn_cursor.fetchall()
	if os.path.exists('ds_ddl.sql'):
		os.remove('ds_ddl.sql')
	f = open('ds_ddl.sql', 'w')
	with open("ds_ddl.sql", "a") as file:
		for creation_str in ddl_str:
			creation_str = creation_str[0]
			file.write(f'{creation_str};\n')


if __name__ == '__main__':
	conn = sqlite3.connect("MLEChallengeDB.db")
	cursor = conn.cursor()
	extract_data_and_produce_ddl_from_parquet(cursor)
	combined_feature_data = read_data_into_memory_and_process_feature_data(conn, cursor)
	X_train_data, Y_train_data = generate_training_data(combined_feature_data, ['lead_uuid', 'offer_id'])
	generate_two_lg_models(X_train_data, Y_train_data)
# print(pd.read_sql(f'select * from {table_name}', conn))
