from keras import backend as K
from keras.layers import Conv2D, Activation, Input, concatenate, Lambda
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense, Dropout


def stem(in_tensor):
	#First
	out = Conv2D(64, (7, 7), padding='same', strides=(2, 2))(in_tensor)
	out = BatchNormalization()(out)
	out = Activation('relu')(out)
	out = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(out)

	# Second
	out = Conv2D(64, (1, 1), padding='same', strides=(1, 1))(out)
	out = BatchNormalization()(out)
	out = Activation('relu')(out)
	out = MaxPooling2D(pool_size=(1, 1), padding='same', strides=2)(out)

	#Third
	out = Conv2D(192, (3, 3), padding='same', strides=(1, 1))(out)
	out = BatchNormalization()(out)
	out = Activation('relu')(out)
	out = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(out)

	return out

def inception1a(in_tensor):
	t3 = Conv2D(96, (1, 1), padding='same')(in_tensor)
	t3 = BatchNormalization()(t3)
	t3 = Activation('relu')(t3)
	t3 = Conv2D(128, (3, 3), padding='same')(t3)
	t3 = BatchNormalization()(t3)
	t3 = Activation('relu')(t3)

	t5 = Conv2D(16, (1, 1), padding='same')(in_tensor)
	t5 = BatchNormalization()(t5)
	t5 = Activation('relu')(t5)
	t5 = Conv2D(32, (5, 5), padding='same')(t5)
	t5 = BatchNormalization()(t5)
	t5 = Activation('relu')(t5)
	
	tpool = MaxPooling2D(pool_size=(3, 3), padding='same', strides=(1, 1))(in_tensor)
	tpool = Conv2D(32, (1, 1), padding='same')(tpool)
	tpool = BatchNormalization()(tpool)
	tpool = Activation('relu')(tpool)
	
	
	t1 = Conv2D(64, (1, 1), padding='same')(in_tensor)
	t1 = BatchNormalization()(t1)
	t1 = Activation('relu')(t1)

	output = concatenate([t3, t5, tpool, t1])
	return output

def inception2a(in_tensor):
	t3 = Conv2D(96, (1, 1), padding='same')(in_tensor)
	t3 = BatchNormalization()(t3)
	t3 = Activation('relu')(t3)
	t3 = Conv2D(192, (3, 3), padding='same')(t3)
	t3 = BatchNormalization()(t3)
	t3 = Activation('relu')(t3)

	t5 = Conv2D(32, (1, 1), padding='same')(in_tensor)
	t5 = BatchNormalization()(t5)
	t5 = Activation('relu')(t5)
	t5 = Conv2D(64, (5, 5), padding='same')(t5)
	t5 = BatchNormalization()(t5)
	t5 = Activation('relu')(t5)
	
	tpool = AveragePooling2D(pool_size=(3, 3), padding='same', strides=(1, 1))(in_tensor)
	tpool = Conv2D(128, (1, 1), padding='same')(tpool)
	tpool = BatchNormalization()(tpool)
	tpool = Activation('relu')(tpool)
	
	
	t1 = Conv2D(256, (1, 1), padding='same')(in_tensor)
	t1 = BatchNormalization()(t1)
	t1 = Activation('relu')(t1)

	output = concatenate([t3, t5, tpool, t1])
	return output

def inception3a(in_tensor):
	t3 = Conv2D(96, (1, 1), padding='same')(in_tensor)
	t3 = BatchNormalization()(t3)
	t3 = Activation('relu')(t3)
	t3 = Conv2D(384, (3, 3), padding='same')(t3)
	t3 = BatchNormalization()(t3)
	t3 = Activation('relu')(t3)
	'''
	t5 = Conv2D(16, (1, 1), padding='same')(in_tensor)
	t5 = BatchNormalization()(t5)
	t5 = Activation('relu')(t5)
	t5 = Conv2D(32, (5, 5), padding='same')(t5)
	t5 = BatchNormalization()(t5)
	t5 = Activation('relu')(t5)
	'''
	tpool = AveragePooling2D(pool_size=(3, 3), padding='same', strides=(1, 1))(in_tensor)
	tpool = Conv2D(96, (1, 1), padding='same')(tpool)
	tpool = BatchNormalization()(tpool)
	tpool = Activation('relu')(tpool)
	
	
	t1 = Conv2D(256, (1, 1), padding='same')(in_tensor)
	t1 = BatchNormalization()(t1)
	t1 = Activation('relu')(t1)

	output = concatenate([t3, tpool, t1])
	return output

def AFC(in_tensor):
	out = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(in_tensor)
	out = Flatten()(out)
	out = Dropout(rate=0.2)(out)
	out = Dense(128)(out)
	out = Lambda(lambda  x: K.l2_normalize(x, axis=1))(out)
	return out

def create_model(input_img):
	input_img = Input(shape=input_img)
	out = stem(input_img)
	out = inception1a(out)
	out = inception2a(out)
	out = inception3a(out)
	out = AFC(out)

	model = Model(inputs = input_img, outputs = out)
	return model

'''
# Input image shape
shape = (96, 96, 3)

model = getANG(shape)
print(model.summary())
'''