import numpy as np
import scipy.io
from copy import copy

# data generation
N = 64
Total_set = 600000
l = 1 # frame length

# method 1 : variational sparsity (Beronulli sampling)
X_ = scipy.io.loadmat('x_32_12_32_8.mat')
X = X_['x_32_12_32_8']
X = X.reshape((Total_set, N))

X_frame = copy(X)
X_frame = X_frame.reshape((Total_set, l, N))

train_test_ratio = 9/10
train_valid_ratio = 8/9
n_split_test = int(X.shape[0]*train_test_ratio)
n_split_valid = int(n_split_test*train_valid_ratio)

X_train = X[:n_split_valid]
X_train_frame = X_frame[:n_split_valid]
Y_train = copy(X_train)
Y_train[np.nonzero(Y_train)] = 1

X_valid = X[n_split_valid:n_split_test]
X_valid_frame = X_frame[n_split_valid:n_split_test]
Y_valid = copy(X_valid)
Y_valid[np.nonzero(X_valid)] = 1

X_test = X[n_split_test:]
X_test_frame = X_frame[n_split_test:]
Y_test = copy(X_test)
Y_test[np.nonzero(X_test)] = 1

data = {
	'train' : {'x' : X_train_frame, 'y' : Y_train},
	'valid' : {'x' : X_valid_frame, 'y' : Y_valid},
	'test' : {'x' : X_test_frame, 'y' : Y_test}
}


print('Training dataset:\n-------------------')
print('x:', data['train']['x'].shape)
print('y:', data['train']['y'].shape)

print('Validation dataset:\n-------------------')
print('x:', data['valid']['x'].shape)
print('y:', data['valid']['y'].shape)

print('Test dataset:\n-------------------')
print('x:', data['test']['x'].shape)
print('y:', data['test']['y'].shape)
print(data['train']['x'][0])
print(data['train']['y'][0])
print(data['test']['x'][0])
print(data['test']['y'][0])
np.save('dataset_32_12_32_8.npy', data)