'''
Recognize faces in a video feed from the camera

Place at least one image of each person to recognize in the people directory
with the person's name (John.jpg)
'''

# Standard library
import os
import glob

# Packages
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array

# Custom
from openface import AlignDlib

# Trained from scratch
from models import ang, ang2, inceptionV4, nn2

# Pretrained
from models import openface_nn4sv1


# Dlib Align
file_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(file_dir,'models')
dlib_model_dir = os.path.join(model_dir, 'dlib')
align = AlignDlib(os.path.join(dlib_model_dir, "shape_predictor_68_face_landmarks.dat"))


# ANG
# dimen=96
# model = ang.create_model((dimen, dimen, 3))
# model.load_weights('./weights/ang_weights.h5')

# ANG2
# dimen=96
# model = ang2.create_model((dimen, dimen, 3))
# model.load_weights('./weights/ang2_weights.h5')

# Inception V4
# dimen = 250
# model = inceptionV4.create_model((dimen, dimen, 3))
# model.load_weights("./weights/iv4_weights.h5")

# NN2
# dimen = 250
# model = nn2.create_model((250, 250, 3))
# model.load_weights("./weights/nn2_weights.h5")

# OPENFACE NN4smallV1 - Pretrained model
dimen = 96
model = openface_nn4sv1.create_model()
model.load_weights('weights/openface_nn4sv1_weights.h5')

font = cv2.FONT_HERSHEY_SIMPLEX

def idFace(face):

	face = cv2.resize(face, (dimen, dimen))
	face = face.astype("float")/255.0
	face_matrix = img_to_array(face, data_format='channels_last')
	face_encoding = model.predict(np.expand_dims(face_matrix, axis=0))

	min_dist = 100
	identity = None
	
	# Loop over the database dictionary's names and encodings.
	for (name, db_encoding) in database.items():
		
		# Compute L2 distance between the target "encoding" and the current "emb" from the database.
		dist = np.linalg.norm(db_encoding - face_encoding)

		#print('distance for %s is %s' %(name, dist))

		# If this distance is less than the min_dist, then set min_dist to dist, and identity to name
		if dist < min_dist:
			min_dist = dist
			identity = name
	
	if min_dist > 1.00:
		return "None"
	else:
		return str(identity)

def detection(database):

	vc = cv2.VideoCapture(0)

	face_cascade = cv2.CascadeClassifier('./haar_cascades/haar_frontalface_default.xml')
	temp = "None"
	while vc.isOpened():
		_, frame = vc.read()
		img = frame
		roi_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		if(faces is not None):	
			for (x, y, w, h) in faces:
				roi_face = roi_color[y:y+h, x:x+w]
				aligned = align.align(dimen, roi_face, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
				if(aligned is not None):
					aligned = cv2.resize(aligned, (dimen, dimen))
					roi_face = aligned
				IDname = idFace(roi_face)
				if(IDname!="None" and IDname!=temp):
					temp = IDname
				cv2.putText(img, IDname, (x+7, y+25), font, 0.8, (60, 76, 231), 2, cv2.LINE_AA)
				cv2.rectangle(img, (x,y), (x+w, y+h), (60, 76, 231), 2)
				cv2.rectangle(img, (x-2,y-2), (x+w+2, y+h+2), (255, 255, 255), 2)
		
		key = cv2.waitKey(100)
		cv2.imshow("preview", img)

		if key == 27: # exit on ESC
			break
	cv2.destroyWindow("preview")


def prepare_database():
	database = {}
	# load all the images of individuals to recognize into the database

	for file in glob.glob("people/*"):
		identity = os.path.splitext(os.path.basename(file))[0]
		database[identity] = img_to_enc(file)

	return database

def img_to_enc(path):
	image = cv2.imread(path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	temp = align.align(dimen, image, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
	if temp is not None:
		image = temp
	image = cv2.resize(image, (dimen, dimen))
	image = image.astype("float")/255.0
	image = img_to_array(image, data_format='channels_last')
	enc = model.predict(np.expand_dims(image, axis=0))
	return enc

if __name__ == '__main__':
	database = prepare_database()
	detection(database)