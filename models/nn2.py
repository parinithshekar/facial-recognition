from keras import backend as K
from keras.layers import Conv2D, Input, concatenate, Lambda
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense, Dropout

# data_format='channels_last' by default!

# Custom L2 Pooling function
def l2_norm(x):
    x = K.square(x)
    x = K.pool2d(x, pool_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last', pool_mode='avg')
    x = x * 9
    x = K.sqrt(x)
    return x

'''
def output_of_lambda(input_shape):
    return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])
'''

# Defining each layer

def L1_Conv1(in_tensor):
	output = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same',
		input_shape=(int(in_tensor.shape[1]), int(in_tensor.shape[2]), int(in_tensor.shape[3])),
		activation='relu', name='1_conv_7x7/2')(in_tensor)
	return output

def L12_MP_N(in_tensor):
	output = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='12_MaxPool_3x3/2')(in_tensor)
	output = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, name='12_BatchNorm')(output)
	return output

def L2_Inception2(in_tensor):
	output = Conv2D(64, kernel_size=(1, 1), activation='relu', padding='same', name='2_Inception2_3red')(in_tensor)
	output = Conv2D(192, kernel_size=(3, 3), activation='relu', padding='same', name='2_Inception2_3x3')(output)
	return output

def L22_MP_N(in_tensor):
	output = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, name='22_BatchNorm')(in_tensor)
	output = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='22_MaxPool_3x3/2')(output)
	return output

def L3_Inception3a(in_tensor):
	tower1 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu', name='3_Inception3a_1x1')(in_tensor)

	tower3 = Conv2D(96, kernel_size=(1, 1), padding='same', activation='relu', name='3_Inception3a_3red')(in_tensor)
	tower3 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name='3_Inception3a_3x3')(tower3)

	tower5 = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu', name='3_Inception3a_5red')(in_tensor)
	tower5 = Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu', name='3_Inception3a_5x5')(tower5)

	towermp = MaxPooling2D(pool_size=(3, 3),  padding='same', strides=(1, 1), name='3_Inception3a_MaxPool_3x3/1')(in_tensor)
	towermp = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu', name='3_Inception3a_MaxPoolConv_1x1')(towermp)

	output = concatenate([tower1, tower3, tower5, towermp], axis=3, name='3_Inception3a')
	return output

def L4_Inception3b(in_tensor):
	tower1 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu', name='4_Inception3b_1x1')(in_tensor)

	tower3 = Conv2D(96, kernel_size=(1, 1), padding='same', activation='relu', name='4_Inception3b_3red')(in_tensor)
	tower3 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name='4_Inception3b_3x3')(tower3)

	tower5 = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu', name='4_Inception3b_5red')(in_tensor)
	tower5 = Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu', name='4_Inception3b_5x5')(tower5)

	towerl2 = Lambda(lambda x: l2_norm(x), name='4_Inception3b_L2Pool_3x3/1')(in_tensor)
	towerl2 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu', name='4_Inception3b_L2PoolConv_1x1')(towerl2)

	output = concatenate([tower1, tower3, tower5, towerl2], axis=3, name='4_Inception3b')
	return output

def L5_Inception3c(in_tensor):
	tower3 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu', name='5_Inception3c_3red')(in_tensor)
	tower3 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', name='5_Inception3c_3x3/2')(tower3)

	tower5 = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu', name='5_Inception3c_5red')(in_tensor)
	tower5 = Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu', name='5_Inception3c_5x5/2')(tower5)

	towermp = MaxPooling2D(pool_size=(3, 3),  padding='same', strides=(2, 2), name='5_Inception3c_MaxPool_3x3/2')(in_tensor)

	output = concatenate([tower3, tower5, towermp], axis=3, name='5_Inception3c')
	return output

def L6_Inception4a(in_tensor):
	tower1 = Conv2D(256, kernel_size=(1, 1), padding='same', activation='relu', name='6_Inception4a_1x1')(in_tensor)

	tower3 = Conv2D(96, kernel_size=(1, 1), padding='same', activation='relu', name='6_Inception4a_3red')(in_tensor)
	tower3 = Conv2D(192, kernel_size=(3, 3), padding='same', activation='relu', name='6_Inception4a_3x3')(tower3)

	tower5 = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu', name='6_Inception4a_5red')(in_tensor)
	tower5 = Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu', name='6_Inception4a_5x5')(tower5)

	towerl2 = Lambda(lambda x: l2_norm(x), name='6_Inception4a_L2Pool_3x3/1')(in_tensor)
	towerl2 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu', name='6_Inception4a_L2PoolConv_1x1')(towerl2)

	output = concatenate([tower1, tower3, tower5, towerl2], axis=3, name='6_Inception4a')
	return output

def L7_Inception4b(in_tensor):
	tower1 = Conv2D(224, kernel_size=(1, 1), padding='same', activation='relu', name='7_Inception4b_1x1')(in_tensor)

	tower3 = Conv2D(112, kernel_size=(1, 1), padding='same', activation='relu', name='7_Inception4b_3red')(in_tensor)
	tower3 = Conv2D(224, kernel_size=(3, 3), padding='same', activation='relu', name='7_Inception4b_3x3')(tower3)

	tower5 = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu', name='7_Inception4b_5red')(in_tensor)
	tower5 = Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu', name='7_Inception4b_5x5')(tower5)

	towerl2 = Lambda(lambda x: l2_norm(x), name='7_Inception4b_L2Pool_3x3/1')(in_tensor)
	towerl2 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu', name='7_Inception4b_L2PoolConv_1x1')(towerl2)

	output = concatenate([tower1, tower3, tower5, towerl2], axis=3, name='7_Inception4b')
	return output

def L8_Inception4c(in_tensor):
	tower1 = Conv2D(192, kernel_size=(1, 1), padding='same', activation='relu', name='8_Inception4c_1x1')(in_tensor)

	tower3 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu', name='8_Inception4c_3red')(in_tensor)
	tower3 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='8_Inception4c_3x3')(tower3)

	tower5 = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu', name='8_Inception4c_5red')(in_tensor)
	tower5 = Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu', name='8_Inception4c_5x5')(tower5)

	towerl2 = Lambda(lambda x: l2_norm(x), name='8_Inception4c_L2Pool_3x3/1')(in_tensor)
	towerl2 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu', name='8_Inception4c_L2PoolConv_1x1')(towerl2)

	output = concatenate([tower1, tower3, tower5, towerl2], axis=3, name='8_Inception4c')
	return output

def L9_Inception4d(in_tensor):
	tower1 = Conv2D(160, kernel_size=(1, 1), padding='same', activation='relu', name='9_Inception4d_1x1')(in_tensor)

	tower3 = Conv2D(144, kernel_size=(1, 1), padding='same', activation='relu', name='9_Inception4d_3red')(in_tensor)
	tower3 = Conv2D(288, kernel_size=(3, 3), padding='same', activation='relu', name='9_Inception4d_3x3')(tower3)

	tower5 = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu', name='9_Inception4d_5red')(in_tensor)
	tower5 = Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu', name='9_Inception4d_5x5')(tower5)

	towerl2 = Lambda(lambda x: l2_norm(x), name='9_Inception4d_L2Pool_3x3/1')(in_tensor)
	towerl2 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu', name='9_Inception4d_L2PoolConv_1x1')(towerl2)

	output = concatenate([tower1, tower3, tower5, towerl2], axis=3, name='9_Inception4d')
	return output

def L10_Inception4e(in_tensor):
	#tower1 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu', name='10_Inception4e_1x1')(in_tensor)

	tower3 = Conv2D(160, kernel_size=(1, 1), padding='same', activation='relu', name='10_Inception4e_3red')(in_tensor)
	tower3 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', name='10_Inception4e_3x3/2')(tower3)

	tower5 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu', name='10_Inception4e_5red')(in_tensor)
	tower5 = Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu', name='10_Inception4e_5x5/2')(tower5)

	towermp = MaxPooling2D(pool_size=(3, 3),  padding='same', strides=(2, 2), name='10_Inception4e_MaxPool_3x3/2')(in_tensor)
	#towermp = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu', name='10_Inception4e_MaxPoolConv_1x1')(towermp)

	output = concatenate([tower3, tower5, towermp], axis=3, name='10_Inception4e')
	return output

def L11_Inception5a(in_tensor):
	tower1 = Conv2D(384, kernel_size=(1, 1), padding='same', activation='relu', name='11_Inception5a_1x1')(in_tensor)

	tower3 = Conv2D(192, kernel_size=(1, 1), padding='same', activation='relu', name='11_Inception5a_3red')(in_tensor)
	tower3 = Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu', name='11_Inception5a_3x3')(tower3)

	tower5 = Conv2D(48, kernel_size=(1, 1), padding='same', activation='relu', name='11_Inception5a_5red')(in_tensor)
	tower5 = Conv2D(128, kernel_size=(5, 5), padding='same', activation='relu', name='11_Inception5a_5x5')(tower5)

	towerl2 = Lambda(lambda x: l2_norm(x), name='11_Inception5a_L2Pool_3x3/1')(in_tensor)
	towerl2 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu', name='11_Inception5a_L2PoolConv_1x1')(towerl2)

	output = concatenate([tower1, tower3, tower5, towerl2], axis=3, name='11_Inception5a')
	return output

def L12_Inception5b(in_tensor):
	tower1 = Conv2D(384, kernel_size=(1, 1), padding='same', activation='relu', name='12_Inception5b_1x1')(in_tensor)

	tower3 = Conv2D(192, kernel_size=(1, 1), padding='same', activation='relu', name='12_Inception5b_3red')(in_tensor)
	tower3 = Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu', name='12_Inception5b_3x3')(tower3)

	tower5 = Conv2D(48, kernel_size=(1, 1), padding='same', activation='relu', name='12_Inception5b_5red')(in_tensor)
	tower5 = Conv2D(128, kernel_size=(5, 5), padding='same', activation='relu', name='12_Inception5b_5x5')(tower5)

	towermp = MaxPooling2D(pool_size=(3, 3),  padding='same', strides=(1, 1), name='12_Inception5b_MaxPool_3x3')(in_tensor)
	towermp = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu', name='12_Inception5b_L2PoolConv_1x1')(towermp)

	output = concatenate([tower1, tower3, tower5, towermp], axis=3, name='12_Inception5b')
	return output

def L13_AvgPoolFC(in_tensor):
	output = AveragePooling2D(pool_size=(int(in_tensor.shape[1]), int(in_tensor.shape[2])), padding='same', name='13_AvgPool_7x7')(in_tensor)
	output = Flatten(name='13_Flatten')(output)
	output = Dense(128, activation='relu', name='13_FC')(output)
	output = Dropout(rate=0.2, seed=7, name='13_FC_Dropout')(output)
	output = Lambda(lambda  x: K.l2_normalize(x,axis=1), name='13_FC_L2Norm')(output)

	return output

def create_model(input_img):
	input_img = Input(shape=input_img)
	out = L1_Conv1(input_img)
	out = L12_MP_N(out)
	out = L2_Inception2(out)
	out = L22_MP_N(out)
	out = L3_Inception3a(out)
	out = L4_Inception3b(out)
	out = L5_Inception3c(out)
	out = L6_Inception4a(out)
	out = L7_Inception4b(out)
	out = L8_Inception4c(out)
	out = L9_Inception4d(out)
	out = L10_Inception4e(out)
	out = L11_Inception5a(out)
	out = L12_Inception5b(out)
	out = L13_AvgPoolFC(out)

	model = Model(inputs = input_img, outputs = out)
	return model

'''
# Input image shape
shape = (220, 220, 3)

model = getNN2(input_img)
print(model.summary())
'''
