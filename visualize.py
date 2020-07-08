'''
Visualize with the benchmarks faces how the 4 models trained from scratch
perform against the pretrained Openface NN4smallV1 model
'''

# Standard library
import os

# Packages
import cv2
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

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

# Openface
openface_model = openface_nn4sv1.create_model()
openface_weights = 'weights/openface_nn4sv1_weights.h5'

# Ang
ang_model = ang.create_model((96, 96, 3))
ang_weights = './weights/ang_weights.h5'

# Ang2
ang2_model = ang2.create_model((96, 96, 3))
ang2_weights = './weights/ang2_weights.h5'

# Inception V4
iv4_model = inceptionV4.create_model((250, 250, 3))
iv4_weights = "./weights/iv4_weights.h5"

# NN2
nn2_model = nn2.create_model((250, 250, 3))
nn2_weights = "./weights/nn2_weights.h5"


def load_image(path):
	img = cv2.imread(path, 1)
	# OpenCV loads images with color channels
	# in BGR order. So we need to reverse them
	return img[...,::-1]

def align_image(img, dimen):
	return align.align(dimen, img, align.getLargestFaceBoundingBox(img), 
						   landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

class IdentityMetadata():
	def __init__(self, base, name, file):
		# dataset base directory
		self.base = base
		# identity name
		self.name = name
		# image file name
		self.file = file

	def __repr__(self):
		return self.image_path()

	def image_path(self):
		return os.path.join(self.base, self.name, self.file) 
	
def load_metadata(path):
	metadata = []
	for i in os.listdir(path):
		for f in os.listdir(os.path.join(path, i)):
			# Check file extension. Allow only jpg/jpeg' files.
			ext = os.path.splitext(f)[1]
			if ext == '.jpg' or ext == '.jpeg':
				metadata.append(IdentityMetadata(path, i, f))
	return np.array(metadata)

def get_embeddings(metadata, model, dimen):
	embedded = np.zeros((metadata.shape[0], 128))

	for i, m in enumerate(metadata):
		img = load_image(m.image_path())
		img = align_image(img, dimen)
		# scale RGB values to interval [0,1]
		img = (img / 255.).astype(np.float32)
		# obtain embedding vector for image
		embedded[i] = model.predict(np.expand_dims(img, axis=0))[0]
	return embedded
	 

def visualize(embedded, targets):
	x_embedded = TSNE(n_components=2).fit_transform(embedded)

	for i, t in enumerate(set(targets)):
		idx = targets == t
		plt.scatter(x_embedded[idx, 0], x_embedded[idx, 1], label=t)   

	#plt.legend(bbox_to_anchor=(1, 1))

def visualize_models(metadata, targets):

	# ANG
	embedded = get_embeddings(metadata, ang_model, dimen=96)
	plt.subplot(4, 2, 1)
	plt.title('[ANG] Model 1 - 3 Conv Stem, 3 Inception Blocks')
	visualize(embedded, targets)

	plt.subplot(4, 2, 2)
	plt.title('10 hours of training')
	ang_model.load_weights(ang_weights)
	embedded = get_embeddings(metadata, ang_model, dimen=96)
	visualize(embedded, targets)

	# ANG2
	embedded = get_embeddings(metadata, ang2_model, dimen=96)
	plt.subplot(4, 2, 3)
	plt.title('[ANG2] Model 2 - 3 Conv Stem, 7 Inception Blocks')
	visualize(embedded, targets)

	plt.subplot(4, 2, 4)
	plt.title('20 hours of training')
	ang2_model.load_weights(ang2_weights)
	embedded = get_embeddings(metadata, ang2_model, dimen=96)
	visualize(embedded, targets)

	# Inception V4
	embedded = get_embeddings(metadata, iv4_model, dimen=250)
	plt.subplot(4, 2, 5)
	plt.title('[Inception V4] Inception V4 - 10 Conv Stem, 5 Inception Blocks')
	visualize(embedded, targets)

	plt.subplot(4, 2, 6)
	plt.title('20 hours of training')
	iv4_model.load_weights(iv4_weights)
	embedded = get_embeddings(metadata, iv4_model, dimen=250)
	visualize(embedded, targets)

	# NN2
	embedded = get_embeddings(metadata, nn2_model, dimen=250)
	plt.subplot(4, 2, 7)
	plt.title('[NN2] Original FaceNet Model - 3 Conv Stem, 11 Inception Blocks')
	visualize(embedded, targets)

	plt.subplot(4, 2, 8)
	plt.title('50 hours of training')
	nn2_model.load_weights(nn2_weights)
	embedded = get_embeddings(metadata, nn2_model, dimen=250)
	visualize(embedded, targets)

	plt.show()

def visualize_openface(metadata, targets):
	embedded = get_embeddings(metadata, openface_model, dimen=96)
	plt.subplot(1, 2, 1)
	plt.title('[OPENFACE] NN4SmallV1 - 3 Conv Stem, 7 Inception Blocks')
	visualize(embedded, targets)

	plt.subplot(1, 2, 2)
	plt.title('3600+ hours of training')
	openface_model.load_weights(openface_weights)
	embedded = get_embeddings(metadata, openface_model, dimen=96)
	visualize(embedded, targets)

	plt.show()

if __name__ == '__main__':
	metadata = load_metadata('benchmark_faces')
	targets = np.array([m.name for m in metadata])
	visualize_models(metadata, targets)
	visualize_openface(metadata, targets)
	
