"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler
from regression import (logreg, utils)

def test_prediction_simple():
	"""
	Simple test case to ensure that format and output of .make_prediction() is correct.
	"""
	# simple, single feature dataset
	X = np.array([[0], [10]])
	y_true = np.array([0, 1])

	simple_model = logreg.LogisticRegressor(num_feats=1)

	# pad X with 1s for bias term and generate y_pred
	X_padded = np.hstack([X, np.ones((X.shape[0], 1))])
	y_pred = simple_model.make_prediction(X_padded)

	# check that predictions are between 0 and 1
	assert np.all(y_pred >=0)
	assert np.all(y_pred <=1)

	# assert that y_pred is same size as y_true
	assert y_true.shape == y_pred.shape


def test_prediction():
	"""
	Test case to ensure that format and output of .make_prediction() is correct
	when used on larger NSCLC dataset. 
	"""
	# Load data
	X_train, _, y_train, _ = utils.loadDataset(
		features=[
			'Penicillin V Potassium 500 MG',
			'Computed tomography of chest and abdomen',
			'Plain chest X-ray (procedure)',
			'Low Density Lipoprotein Cholesterol',
			'Creatinine',
			'AGE_DIAGNOSIS'
		],
		split_percent=0.8,
		split_seed=42
	)

	# scale data using training data
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)

	# padding data with an extra column of ones for bias term
	X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])

	# initiate model and run make_prediction function
	big_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.01, tol=0.01, max_iter=20, batch_size=10)
	
	y_pred = big_model.make_prediction(X_train)


	# check that predictions are between 0 and 1
	assert np.all(y_pred >=0)
	assert np.all(y_pred <=1)

	# assert that y_pred is same size as y_true
	assert y_train.shape == y_pred.shape


def test_loss_function_simple():
	"""
	Simple test case to sanity check that loss outputs from .loss_function()
	follow expected logic. 

	A nearly correct solution should have a loss relatively close to 0 (encouraging
	the model to only make no to very minor tweaks to the weights).

	A partially incorrect and a very incorrect solution should have a loss value greater than 0,
	but the very incorrect solution should at least have a greater loss value comparatively (in
	order to encourage greater changes in the model weights). 
	"""
	simple_model = logreg.LogisticRegressor(num_feats=1)

	# setting up a few simple test cases
	y_true = np.array([0, 1, 0, 1 ,0])

	y_correct = np.array([0.001, 0.99, 0.001, 0.999 ,0.001], dtype='float64') # nearly perfect soln
	y_incorrect = np.array([0.99, 0.001, 0.999, 0.001, 0.999], dtype='float64') # completely incorrect soln
	y_partial = np.array([0.001, 0.001, 0.001, 0.001, 0.001], dtype='float64') # partially correct soln

	# nearly correct solutions should have a BCE of almost 0
	print(simple_model.loss_function(y_true, y_correct))
	assert np.isclose(simple_model.loss_function(y_true, y_correct), 0, atol=0.01)

	# both the partially and completely incorrect solns should have positive losses (partial < complete incorr)
	partial_BCE = simple_model.loss_function(y_true, y_partial)
	incorrect_BCE = simple_model.loss_function(y_true, y_incorrect)

	assert partial_BCE > 0
	assert incorrect_BCE > 0
	assert partial_BCE < incorrect_BCE


def test_loss_function():
	"""
	Test case to check that format and output of loss function is correct
	when used on larger NSCLC dataset. 
	"""
	# Load data
	X_train, _, y_train, _ = utils.loadDataset(
		features=[
			'Penicillin V Potassium 500 MG',
			'Computed tomography of chest and abdomen',
			'Plain chest X-ray (procedure)',
			'Low Density Lipoprotein Cholesterol',
			'Creatinine',
			'AGE_DIAGNOSIS'
		],
		split_percent=0.8,
		split_seed=42
	)

	# scale data using training data
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)

	# padding data with an extra column of ones for bias term
	X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])

	# initiate model and run loss function
	big_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.01, tol=0.01, max_iter=20, batch_size=10)
	
	y_pred = big_model.make_prediction(X_train)

	test_BCE = big_model.loss_function(y_train, y_pred)

	# in all likelihood, the initial weights randomized by model should not be perfect, so loss function should output a value >0
	assert test_BCE > 0


def test_gradient_simple():
	"""
	A simple test case to assert that output from .calculate_gradient() is in
	the format and that gradient update logic are as expected.

	If we set the model weights to zero for a dataset that is very easily separable,
	the first gradient should at least include one negative and non-zero value so that
	one of the weights is updated. 
	"""
	# simple example data with one feature
	num_feats = 1

	X = np.array([[0], [10]])
	X_padded = np.hstack([X, np.ones((X.shape[0], 1))])

	y_true = np.array([0, 1])

	# setting model weights to 0 to test
	simple_model = logreg.LogisticRegressor(num_feats=num_feats)
	simple_model.W = np.zeros(num_feats + 1).flatten()

	grad = simple_model.calculate_gradient(y_true, X_padded)

	# assert that gradient vector correctly contains one value per features (and one for bias)
	assert grad.size == num_feats + 1 

	# in this case, at least one element of the grad should be < 0, in order to suggest model to increase weights from 0s
	assert np.any(grad < 0)

def test_gradient():
	"""
	A simple test case to assert that output from .calculate_gradient() is in
	the format of gradient function output is as expected.
	"""
	# Load data
	X_train, _, y_train, _ = utils.loadDataset(
		features=[
			'Penicillin V Potassium 500 MG',
			'Computed tomography of chest and abdomen',
			'Plain chest X-ray (procedure)',
			'Low Density Lipoprotein Cholesterol',
			'Creatinine',
			'AGE_DIAGNOSIS'
		],
		split_percent=0.8,
		split_seed=42
	)

	# scale data using training data
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)

	# padding data with an extra column of ones for bias term
	X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])

	# initiate model and run calculate_gradient function
	big_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.01, tol=0.01, max_iter=20, batch_size=10)

	grad = big_model.calculate_gradient(y_train, X_train)

	# assert that gradient vector correctly contains one value per features (and one for bias)
	assert grad.size == big_model.num_feats + 1

	# given that initial, random weights will likely not be perfect, at least one element of the grad should prob be != 0, 
	# in order to suggest model nudge weights somehow
	assert np.any(grad != 0)


def test_training():
	"""
	Test case to check that weights were updated after training
	"""
	X_train, X_val, y_train, y_val = utils.loadDataset(
		features=[
			'Penicillin V Potassium 500 MG',
			'Computed tomography of chest and abdomen',
			'Plain chest X-ray (procedure)',
			'Low Density Lipoprotein Cholesterol',
			'Creatinine',
			'AGE_DIAGNOSIS'
		],
		split_percent=0.8,
		split_seed=42
	)

	# scale data using training data and apply scalar to val dataset
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

	# initiate model and record initial weights
	big_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.01, tol=0.01, max_iter=20, batch_size=10)
	initial_weights = big_model.W

	# train model and record weights after training
	big_model.train_model(X_train, y_train, X_val, y_val)
	trained_weights = big_model.W

	# given that random, initial weights were probably not perfect, at least one should have changed from its initial setting
	assert np.any(initial_weights-trained_weights !=0)