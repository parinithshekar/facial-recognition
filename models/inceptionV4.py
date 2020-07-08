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

def Stem_1(in_tensor):
	output = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu', name='Stem_11_conv_3x3V/2')(in_tensor)
	output = BatchNormalization()(output)
	output = Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', name='Stem_12_conv_3x3V')(output)
	output = BatchNormalization()(output)
	output = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name='Stem_13_conv_3x3')(output)
	output = BatchNormalization()(output)

	mp_branch = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='Stem_14a_mp_3x3V/2')(output)
	conv_branch = Conv2D(96, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu', name='Stem_14b_conv_3x3V/2')(output)
	conv_branch = BatchNormalization()(conv_branch)

	output = concatenate([conv_branch, mp_branch], axis=3, name='Stem_1')
	return output

def Stem_2(in_tensor):
	b1 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu', name='Stem_21b1_conv_1x1')(in_tensor)
	b1 = BatchNormalization()(b1)
	b1 = Conv2D(96, kernel_size=(3, 3), padding='valid', activation='relu', name='Stem_22b1_conv_3x3V')(b1)
	b1 = BatchNormalization()(b1)

	b2 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu', name='Stem_21b2_conv_1x1')(in_tensor)
	b2 = BatchNormalization()(b2)
	b2 = Conv2D(64, kernel_size=(7, 1), padding='same', activation='relu', name='Stem_22b2_conv_7x1')(b2)
	b2 = BatchNormalization()(b2)
	b2 = Conv2D(64, kernel_size=(1, 7), padding='same', activation='relu', name='Stem_23b2_conv_1x7')(b2)
	b2 = BatchNormalization()(b2)
	b2 = Conv2D(96, kernel_size=(3, 3), padding='valid', activation='relu', name='Stem_24b2_conv_3x3V')(b2)
	b2 = BatchNormalization()(b2)

	output = concatenate([b2, b1], axis=3, name='Stem_2')
	return output

def Stem_3(in_tensor):
	conv = Conv2D(192, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu', name='Stem_31conv_conv_3x3V')(in_tensor)
	conv = BatchNormalization()(conv)

	mp = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='Stem_31mp_mp_3x3V/2')(in_tensor)

	output = concatenate([mp, conv], axis=3, name='Stem_3')
	return output

def stem(in_tensor):
	output = Stem_1(in_tensor)
	output = Stem_2(output)
	output = Stem_3(output)
	return output

def Inception_v4_A1(in_tensor):
	towerap = AveragePooling2D(pool_size=(1, 1), padding='same', name='Inception_v4A1_tap_ap1_1x1')(in_tensor)
	towerap = Conv2D(96, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4A1_tap_conv2_1x1')(towerap)
	towerap = BatchNormalization()(towerap)

	tower1 = Conv2D(96, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4A1_t1_conv1_1x1')(in_tensor)
	tower1 = BatchNormalization()(tower1)

	tower3 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4A1_t3_conv1_1x1')(in_tensor)
	tower3 = BatchNormalization()(tower3)
	tower3 = Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu', name='Inception_v4A1_t3_conv2_3x3')(tower3)
	tower3 = BatchNormalization()(tower3)

	tower5 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4A1_t5_conv1_1x1')(in_tensor)
	tower5 = BatchNormalization()(tower5)
	tower5 = Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu', name='Inception_v4A1_t5_conv2_3x3')(tower5)
	tower5 = BatchNormalization()(tower5)
	tower5 = Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu', name='Inception_v4A1_t5_conv3_3x3')(tower5)
	tower5 = BatchNormalization()(tower5)

	output = concatenate([tower5, tower3, tower1, towerap], axis=3, name='Inception_v4_A1')
	return output

def Inception_v4_A2(in_tensor):
	towerap = AveragePooling2D(pool_size=(1, 1), padding='same', name='Inception_v4A2_tap_ap1_1x1')(in_tensor)
	towerap = Conv2D(96, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4A2_tap_conv2_1x1')(towerap)
	towerap = BatchNormalization()(towerap)

	tower1 = Conv2D(96, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4A2_t1_conv1_1x1')(in_tensor)
	tower1 = BatchNormalization()(tower1)

	tower3 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4A2_t3_conv1_1x1')(in_tensor)
	tower3 = BatchNormalization()(tower3)
	tower3 = Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu', name='Inception_v4A2_t3_conv2_3x3')(tower3)
	tower3 = BatchNormalization()(tower3)

	tower5 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4A2_t5_conv1_1x1')(in_tensor)
	tower5 = BatchNormalization()(tower5)
	tower5 = Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu', name='Inception_v4A2_t5_conv2_3x3')(tower5)
	tower5 = BatchNormalization()(tower5)
	tower5 = Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu', name='Inception_v4A2_t5_conv3_3x3')(tower5)
	tower5 = BatchNormalization()(tower5)

	output = concatenate([tower5, tower3, tower1, towerap], axis=3, name='Inception_v4_A2')
	return output

def Inception_v4_ReduxA(in_tensor):
	towermp = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='Inception_v4_ReduxA_tmp_mp1_3x3V/2')(in_tensor)

	tower3 = Conv2D(384, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu', name='Inception_v4_ReduxA_t3_conv1_3x3V/2')(in_tensor)
	tower3 = BatchNormalization()(tower3)

	tower5 = Conv2D(192, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4_ReduxA_t5_conv1_1x1')(in_tensor)
	tower5 = BatchNormalization()(tower5)
	tower5 = Conv2D(224, kernel_size=(3, 3), padding='same', activation='relu', name='Inception_v4_ReduxA_t5_conv2_3x3')(tower5)
	tower5 = BatchNormalization()(tower5)
	tower5 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu', name='Inception_v4_ReduxA_t5_conv3_3x3V/2')(tower5)
	tower5 = BatchNormalization()(tower5)

	output = concatenate([tower5, tower3, towermp], axis=3, name='Inception_v4_ReduxA')
	return output

def Inception_v4_B1(in_tensor):
	towerap = AveragePooling2D(pool_size=(1, 1), padding='same', name='Inception_v4B1_tap_ap1_1x1')(in_tensor)
	towerap = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4B1_tap_conv2_1x1')(towerap)
	towerap = BatchNormalization()(towerap)

	tower1 = Conv2D(384, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4B1_t1_conv1_1x1')(in_tensor)
	tower1 = BatchNormalization()(tower1)

	tower7 = Conv2D(192, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4B1_t7_conv1_1x1')(in_tensor)
	tower7 = BatchNormalization()(tower7)
	tower7 = Conv2D(224, kernel_size=(1, 7), padding='same', activation='relu', name='Inception_v4B1_t7_conv2_1x7')(tower7)
	tower7 = BatchNormalization()(tower7)
	tower7 = Conv2D(256, kernel_size=(1, 7), padding='same', activation='relu', name='Inception_v4B1_t7_conv3_1x7')(tower7)
	tower7 = BatchNormalization()(tower7)

	tower77 = Conv2D(192, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4B1_t77_conv1_1x1')(in_tensor)
	tower77 = BatchNormalization()(tower77)
	tower77 = Conv2D(192, kernel_size=(1, 7), padding='same', activation='relu', name='Inception_v4B1_t77_conv2_1x7')(tower77)
	tower77 = BatchNormalization()(tower77)
	tower77 = Conv2D(224, kernel_size=(7, 1), padding='same', activation='relu', name='Inception_v4B1_t77_conv3_7x1')(tower77)
	tower77 = BatchNormalization()(tower77)
	tower77 = Conv2D(224, kernel_size=(1, 7), padding='same', activation='relu', name='Inception_v4B1_t77_conv4_1x7')(tower77)
	tower77 = BatchNormalization()(tower77)
	tower77 = Conv2D(256, kernel_size=(7, 1), padding='same', activation='relu', name='Inception_v4B1_t77_conv5_7x1')(tower77)
	tower77 = BatchNormalization()(tower77)

	output = concatenate([tower77, tower7, tower1, towerap], axis=3, name='Inception_v4_B1')
	return output

def Inception_v4_B2(in_tensor):
	towerap = AveragePooling2D(pool_size=(1, 1), padding='same', name='Inception_v4B2_tap_ap1_1x1')(in_tensor)
	towerap = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4B2_tap_conv2_1x1')(towerap)
	towerap = BatchNormalization()(towerap)

	tower1 = Conv2D(384, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4B2_t1_conv1_1x1')(in_tensor)
	tower1 = BatchNormalization()(tower1)

	tower7 = Conv2D(192, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4B2_t7_conv1_1x1')(in_tensor)
	tower7 = BatchNormalization()(tower7)
	tower7 = Conv2D(224, kernel_size=(1, 7), padding='same', activation='relu', name='Inception_v4B2_t7_conv2_1x7')(tower7)
	tower7 = BatchNormalization()(tower7)
	tower7 = Conv2D(256, kernel_size=(1, 7), padding='same', activation='relu', name='Inception_v4B2_t7_conv3_1x7')(tower7)
	tower7 = BatchNormalization()(tower7)

	tower77 = Conv2D(192, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4B2_t77_conv1_1x1')(in_tensor)
	tower77 = BatchNormalization()(tower77)
	tower77 = Conv2D(192, kernel_size=(1, 7), padding='same', activation='relu', name='Inception_v4B2_t77_conv2_1x7')(tower77)
	tower77 = BatchNormalization()(tower77)
	tower77 = Conv2D(224, kernel_size=(7, 1), padding='same', activation='relu', name='Inception_v4B2_t77_conv3_7x1')(tower77)
	tower77 = BatchNormalization()(tower77)
	tower77 = Conv2D(224, kernel_size=(1, 7), padding='same', activation='relu', name='Inception_v4B2_t77_conv4_1x7')(tower77)
	tower77 = BatchNormalization()(tower77)
	tower77 = Conv2D(256, kernel_size=(7, 1), padding='same', activation='relu', name='Inception_v4B2_t77_conv5_7x1')(tower77)
	tower77 = BatchNormalization()(tower77)

	output = concatenate([tower77, tower7, tower1, towerap], axis=3, name='Inception_v4_B2')
	return output

def Inception_v4_ReduxB(in_tensor):
	towermp = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='Inception_v4_ReduxB_tmp_mp1_3x3V/2')(in_tensor)

	tower3 = Conv2D(192, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4_ReduxB_t3_conv1_1x1')(in_tensor)
	tower3 = BatchNormalization()(tower3)
	tower3 = Conv2D(192, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu', name='Inception_v4_ReduxB_t3_conv2_3x3V/2')(tower3)
	tower3 = BatchNormalization()(tower3)

	tower73 = Conv2D(256, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4_ReduxB_t73_conv1_1x1')(in_tensor)
	tower73 = BatchNormalization()(tower73)
	tower73 = Conv2D(256, kernel_size=(1, 7), padding='same', activation='relu', name='Inception_v4_ReduxB_t73_conv2_1x7')(tower73)
	tower73 = BatchNormalization()(tower73)
	tower73 = Conv2D(320, kernel_size=(7, 1), padding='same', activation='relu', name='Inception_v4_ReduxB_t73_conv3_7x1')(tower73)
	tower73 = BatchNormalization()(tower73)
	tower73 = Conv2D(320, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu', name='Inception_v4_ReduxB_t73_conv4_3x3V/2')(tower73)
	tower73 = BatchNormalization()(tower73)

	output = concatenate([tower73, tower3, towermp], axis=3, name='Inception_v4_ReduxB')
	return output

def Inception_v4_C1(in_tensor):
	towerap = AveragePooling2D(pool_size=(1, 1), padding='same', name='Inception_v4C1_tap_ap1_1x1')(in_tensor)
	towerap = Conv2D(256, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4C1_tap_conv2_1x1')(towerap)
	towerap = BatchNormalization()(towerap)

	tower1 = Conv2D(256, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4C1_t1_conv1_1x1')(in_tensor)
	tower1 = BatchNormalization()(tower1)

	tower3 = Conv2D(192, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4C1_t3_conv1_1x1')(in_tensor)
	tower3 = BatchNormalization()(tower3)
	tower3a = Conv2D(256, kernel_size=(1, 3), padding='same', activation='relu', name='Inception_v4C1_t3a_conv2_1x3')(tower3)
	tower3a = BatchNormalization()(tower3a)
	tower3b = Conv2D(256, kernel_size=(3, 1), padding='same', activation='relu', name='Inception_v4C1_t3b_conv2_3x1')(tower3)
	tower3b = BatchNormalization()(tower3b)

	tower5 = Conv2D(384, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4C1_t5_conv1_1x1')(in_tensor)
	tower5 = BatchNormalization()(tower5)
	tower5 = Conv2D(448, kernel_size=(1, 3), padding='same', activation='relu', name='Inception_v4C1_t5_conv2_1x3')(tower5)
	tower5 = BatchNormalization()(tower5)
	tower5 = Conv2D(512, kernel_size=(3, 1), padding='same', activation='relu', name='Inception_v4C1_t5_conv3_1x3')(tower5)
	tower5 = BatchNormalization()(tower5)
	tower5a = Conv2D(256, kernel_size=(3, 1), padding='same', activation='relu', name='Inception_v4C1_t5a_conv4_3x1')(tower5)
	tower5a = BatchNormalization()(tower5a)
	tower5b = Conv2D(256, kernel_size=(1, 3), padding='same', activation='relu', name='Inception_v4C1_t5b_conv4_1x3')(tower5)
	tower5b = BatchNormalization()(tower5b)

	output = concatenate([tower5b, tower5a, tower3b, tower3a, tower1, towerap], axis=3, name='Inception_v4_C1')
	return output

def Inception_v4_C2(in_tensor):
	towerap = AveragePooling2D(pool_size=(1, 1), padding='same', name='Inception_v4C2_tap_ap1_1x1')(in_tensor)
	towerap = Conv2D(256, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4C2_tap_conv2_1x1')(towerap)
	towerap = BatchNormalization()(towerap)

	tower1 = Conv2D(256, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4C2_t1_conv1_1x1')(in_tensor)
	tower1 = BatchNormalization()(tower1)

	tower3 = Conv2D(192, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4C2_t3_conv1_1x1')(in_tensor)
	tower3 = BatchNormalization()(tower3)
	tower3a = Conv2D(256, kernel_size=(1, 3), padding='same', activation='relu', name='Inception_v4C2_t3a_conv2_1x3')(tower3)
	tower3a = BatchNormalization()(tower3a)
	tower3b = Conv2D(256, kernel_size=(3, 1), padding='same', activation='relu', name='Inception_v4C2_t3b_conv2_3x1')(tower3)
	tower3b = BatchNormalization()(tower3b)

	tower5 = Conv2D(384, kernel_size=(1, 1), padding='same', activation='relu', name='Inception_v4C2_t5_conv1_1x1')(in_tensor)
	tower5 = BatchNormalization()(tower5)
	tower5 = Conv2D(448, kernel_size=(1, 3), padding='same', activation='relu', name='Inception_v4C2_t5_conv2_1x3')(tower5)
	tower5 = BatchNormalization()(tower5)
	tower5 = Conv2D(512, kernel_size=(3, 1), padding='same', activation='relu', name='Inception_v4C2_t5_conv3_1x3')(tower5)
	tower5 = BatchNormalization()(tower5)
	tower5a = Conv2D(256, kernel_size=(3, 1), padding='same', activation='relu', name='Inception_v4C2_t5a_conv4_3x1')(tower5)
	tower5a = BatchNormalization()(tower5a)
	tower5b = Conv2D(256, kernel_size=(1, 3), padding='same', activation='relu', name='Inception_v4C2_t5b_conv4_1x3')(tower5)
	tower5b = BatchNormalization()(tower5b)

	output = concatenate([tower5b, tower5a, tower3b, tower3a, tower1, towerap], axis=3, name='Inception_v4_C2')
	return output

def AvgPoolFC(in_tensor):
	output = AveragePooling2D(pool_size=(int(in_tensor.shape[1]), int(in_tensor.shape[2])), padding='same', name='AFC_ap')(in_tensor)
	output = Flatten(name='AFC_Flatten')(output)
	output = Dropout(rate=0.35, name='AFC_Dropout')(output)
	output = Dense(128, activation='relu', name='AFC_FC')(output)
	output = BatchNormalization()(output)
	#output = Lambda(lambda  x: K.l2_normalize(x,axis=1), name='AFC_L2Norm')(output)

	return output

def create_model(input_img):
	input_img = Input(shape=input_img)
	out = stem(input_img)
	out = Inception_v4_A1(out)
	out = Inception_v4_A2(out)
	out = Inception_v4_ReduxA(out)
	out = Inception_v4_B1(out)
	out = Inception_v4_B2(out)
	out = Inception_v4_ReduxB(out)
	out = Inception_v4_C1(out)
	out = Inception_v4_C2(out)
	out = AvgPoolFC(out)
	model = Model(inputs = input_img, outputs = out)
	return model

'''
# Input image shape
shape = (250, 250, 3)
model = getInceptionV4(shape)
print(model.summary())
'''