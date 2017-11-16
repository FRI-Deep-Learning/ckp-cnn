import sys

args = sys.argv[1:]

if len(args) < 3:
    print("[!] Not enough arguments!\nUsage: python train_siamese_model.py <image_db> <num_training_pairs> <num_validation_pairs> [<ram_max_gb>]")
    exit(1)

IMAGE_DB_FILENAME = args[0]
NUM_TRAINING_PAIRS = int(args[1])
NUM_VALIDATION_PAIRS = int(args[2])
MAX_RAM = int(args[3]) if len(args) == 4 else -1

IMG_SIZE = 64
NUM_EPOCHS = 30
BATCH_SIZE = 128

print("> Initializing Tensorflow and Keras...")

import keras
import gzip
import umsgpack
from tqdm import tqdm
import random
import numpy as np
import uuid

model_uuid = uuid.uuid1()

random.seed()

print("> Loading training images...")

image_db = gzip.open(IMAGE_DB_FILENAME, "rb")

num_subjects = umsgpack.unpack(image_db)

subjects = []

for i in tqdm(range(0, num_subjects)):
    num_images = umsgpack.unpack(image_db)
    images = []

    for ii in range(0, num_images):
        images.append(umsgpack.unpack(image_db))
    
    subjects.append(images)

average_image = umsgpack.unpack(image_db)
image_db.close()

NUM_TRAINING_PAIRS_HALF = NUM_TRAINING_PAIRS // 2
NUM_VALIDATION_PAIRS_HALF = NUM_VALIDATION_PAIRS // 2

training_pairs_x_l = []
training_pairs_x_r = []
training_pairs_y = []

validation_pairs_x_l = []
validation_pairs_x_r = []
validation_pairs_y = []

def transform_image(image):
    image = np.array(image)
    image = image - average_image
    image /= 255
    return np.expand_dims(image, axis=2)

def pick_pairs(num_pairs_half, x_l, x_r, y):
    for i in range(0, num_pairs_half):
        subject_idx = i % num_subjects
        image_a = transform_image(random.choice(subjects[subject_idx]))
        image_b = transform_image(random.choice(subjects[subject_idx]))
        x_l.append(image_a)
        x_r.append(image_b)
        y.append(1)
    
    for i in range(0, num_pairs_half):
        subject_idx = i % num_subjects
        other_subject_idx = random.randrange(num_subjects)
        image_a = transform_image(random.choice(subjects[subject_idx]))
        image_b = transform_image(random.choice(subjects[other_subject_idx]))
        x_l.append(image_a)
        x_r.append(image_b)
        y.append(0)

print("> Picking training pairs...")
pick_pairs(NUM_TRAINING_PAIRS_HALF, training_pairs_x_l, training_pairs_x_r, training_pairs_y)
print("> Picking validation pairs...")
pick_pairs(NUM_VALIDATION_PAIRS_HALF, validation_pairs_x_l, validation_pairs_x_r, validation_pairs_y)

print("> Compiling model...")

from keras.layers import Dropout, Flatten, Average, Concatenate, MaxPooling2D, Input, Conv2D, Dense, Activation
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

training_pairs_x_l = np.array(training_pairs_x_l)
training_pairs_x_r = np.array(training_pairs_x_r)
training_pairs_y = np.array(training_pairs_y)
validation_pairs_x_l = np.array(validation_pairs_x_l)
validation_pairs_x_r = np.array(validation_pairs_x_r)
validation_pairs_y = np.array(validation_pairs_y)

# print("===", training_pairs_x_l.shape, training_pairs_x_r.shape, training_pairs_y.shape)

# training_pairs_y = np_utils.to_categorical(training_pairs_y, 2)
# validation_pairs_y = np_utils.to_categorical(validation_pairs_y, 2)

def build_twin(prefix, l_input):
    l_c1 = Conv2D(64, (3, 3), activation="relu", name=prefix+"c1")(l_input)
    l_c2 = Conv2D(128, (3, 3), activation="relu", name=prefix+"c2")(l_c1)
    l_p1 = MaxPooling2D(pool_size=(2, 2), name=prefix+"p1")(l_c2)

    l_c3 = Conv2D(256, (3, 3), activation="relu", name=prefix+"c3")(l_p1)
    l_c4 = Conv2D(256, (3, 3), activation="relu", name=prefix+"c4")(l_c3)
    l_p2 = MaxPooling2D(pool_size=(2, 2), name=prefix+"p2")(l_c4)

    l_c5 = Conv2D(512, (3, 3), activation="relu", name=prefix+"c5")(l_p2)
    l_c6 = Conv2D(512, (3, 3), activation="relu", name=prefix+"c6")(l_c5)
    l_c7 = Conv2D(512, (3, 3), activation="relu", name=prefix+"c7")(l_c6)
    l_p3 = MaxPooling2D(pool_size=(2, 2), name=prefix+"p3")(l_c7)

    return l_p3

left_input = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
left_twin = build_twin("l_", left_input)
right_input = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
right_twin = build_twin("r_", right_input)

# l_cc1 = Concatenate(axis=1)([left_twin, right_twin])
l_cc1 = Average()([left_twin, right_twin])
# l_fl1 = Flatten()(l_cc1)
# l_fc1 = Dense(1024)(l_fl1)
# l_do1 = Dropout(0.5)(l_fc1)
# l_fc2 = Dense(128)(l_do1)
l_ffc1 = Conv2D(1024, (3, 3))(l_cc1)
l_ffc2 = Conv2D(512, (1, 1))(l_ffc1)
l_fl1 = Flatten()(l_ffc2)
l_fc3 = Dense(32)(l_fl1)
l_fc4 = Dense(1)(l_fc3)
l_sm1 = Activation("softmax")(l_fc4)

model = Model(inputs=[left_input, right_input], outputs=l_sm1)

model.summary()

preds = model.predict([np.ones((1, 64, 64, 1)),np.ones((1, 64, 64, 1))])
print(preds.shape)

# from https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
def get_model_memory_usage(batch_size, model):
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    total_memory = 4*(batch_size/1024**3)*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = round(total_memory, 3)
    return float(gbytes)

memory_usage = get_model_memory_usage(BATCH_SIZE, model)
print("> Memory Usage: ", memory_usage, "GB")

if MAX_RAM == -1:
    answer = input("> Continue? y/n: ")
    if not answer.lower().startswith("y"):
        exit(0)
else:
    if memory_usage > MAX_RAM:
        print("[!] Memory usage above acceptable limit! Stopping!")
        exit(1)
    else:
        print("> Memory usage is acceptable. Continuing.")

print("> Compiling...")

sgd = SGD(lr=0.01, momentum=0.9, decay=0.001)

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print("> Training...")

print(training_pairs_x_l.shape, training_pairs_x_r.shape, training_pairs_y.shape)

model.fit([training_pairs_x_l, training_pairs_x_r], training_pairs_y,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              validation_data=([validation_pairs_x_l, validation_pairs_x_r], validation_pairs_y),
              shuffle=True,
              callbacks=[ModelCheckpoint("best_siamese_model_{}.h5".format(str(model_uuid)), save_best_only=True)])