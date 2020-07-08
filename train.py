'''
Train and save the weights for any of the 4 model architectures

Download the face image dataset separately and set it as the training_data directory
'''

# Standard library
import time

# Packages
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Trained from scratch
from models import ang, ang2, inceptionV4, nn2

''' Comment/uncomment appropriate lines to select the model to train '''
# ANG
# dimen = 96
# model = ang.create_model((dimen, dimen, 3))
# weights = "./weights/ang_weights.h5"

# ANG2
# dimen = 96
# model = ang2.create_model((dimen, dimen, 3))
# weights = "./weights/ang2_weights.h5"

# Inception V4
# dimen = 250
# model = inceptionV4.create_model((dimen, dimen, 3))
# weights = "./weights/iv4_weights.h5"

# NN2
dimen = 250
model = nn2.create_model((dimen, dimen, 3))
weights = "./weights/nn2_weights.h5"


''' Directory name containing training data '''
# Refer to https://keras.io/api/preprocessing/image/
# for subdirectory structure of training_data directory
training_data = "MSC_red"

''' Hyperparameters controlling model training '''
epochs = 500 # Total number of epochs to train on
batch_size = 80 # No of images per batch
batches_per_epoch = 30 # AKA steps_per_epoch
margin = 0.6 # dist(anchor, negative) - dist(anchor, positive)


# Triplet loss
def tripletLoss(margin):
    def fromModel(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [batch_size]), dtype=tf.int32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        m = tf.cast(margin, dtype=tf.float32)
        loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels=y_true, embeddings=y_pred, margin=m)
        return tf.ones([batch_size, 1]) * loss
    return fromModel

# Data Generator
train_datagen = ImageDataGenerator(
        rotation_range = 20,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        shear_range = 0.15,
        zoom_range = 0.05,
        horizontal_flip = True,
        fill_mode = 'nearest',
        data_format = "channels_last")

train_generator = train_datagen.flow_from_directory(
        directory = training_data,  # this is the target directory
        target_size = (dimen, dimen),  # all images will be resized to dimen X dimen
        batch_size = batch_size,
        shuffle = False,
        class_mode = 'sparse')

# Get model
model.compile(loss=tripletLoss(margin=margin), optimizer='adam', metrics=['accuracy'])

# To use a previous training checkpoint as a start point
# model.load_weights(weights)

# Generator Fit
start_time = time.time()
model.fit_generator(train_generator, steps_per_epoch=batches_per_epoch, epochs=epochs)
total_time = time.time() - start_time

print(f"{epochs} Epochs | {batch_size} Images per epoch | {batches_per_epoch} Steps per epoch")

# # Serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# print("Saved model JSON to disk")

# Serialize weights to HDF5
model.save_weights(weights, overwrite=True)
print(f"Saved {weights} to disk")

# Print time taken
hrs = int(total_time//3600)
mins = int((total_time%3600)//60)
secs = int(total_time%60)
print(f"Time Taken = {hrs} Hours, {mins} Mins, {secs} Secs")