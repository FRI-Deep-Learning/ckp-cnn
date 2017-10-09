import sys

if len(sys.argv) != 3:
    print("> Usage: python train_embedding.py <number of anchors per subject> <model file>")
    exit(1)

from keras.models import load_model
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import uuid
from tqdm import tqdm
import gzip
import umsgpack
import random
import numpy as np

num_epoch = 20
batch_size = 129

L = int(sys.argv[1])

model_uuid = uuid.uuid1()

print("> Loading data...")

input_file_name = "preprocessed_db.gz"

input_file = gzip.open(input_file_name, "rb")

num_images = umsgpack.unpack(input_file)

subject_image_map = umsgpack.unpack(input_file)

subject_take_map = {}

for subject in subject_image_map:
    num_subject_images = subject_image_map[subject]

    image_nums = [i for i in range(0, num_subject_images)]
    random.shuffle(image_nums)

    to_take = image_nums[:min(len(image_nums), L)]

    subject_take_map[subject] = to_take

subject_image_id_map = {s: 0 for s in subject_take_map}

subject_anchor_map = {s: [] for s in subject_take_map}

print(">   Gathering anchor images...")
for i in tqdm(range(0, num_images)):
    (subject, image) = umsgpack.unpack(input_file)

    current_id = subject_image_id_map[subject]
    subject_image_id_map[subject] += 1

    if current_id in subject_take_map[subject]:
        subject_anchor_map[subject].append(image)

total_triplets = []

print(">   Creating triplets...")
for subject in tqdm(subject_anchor_map):
    subject_images = subject_anchor_map[subject]
    
    total_pairs = []
    for i in range(0, len(subject_images)): # For each anchor image
        for j in range(0, len(subject_images)): # Get all other images (positives)
            if i == j: # Unless they are the same image
                continue
            
            other_subject = random.choice([p for p in subject_anchor_map if p != subject])
            other_image = random.choice(subject_anchor_map[other_subject])

            total_triplets.append(subject_images[i])
            total_triplets.append(subject_images[j])
            total_triplets.append(other_image)

input_file.close()
del subject_anchor_map
print("> Triplet Total:", len(total_triplets))

print("> Loading model...")
model = load_model(sys.argv[2])

# Freeze all layers
for layer in model.layers:
    layer.trainable = False

model.pop() # Removes softmax layer
model.pop() # Removes last fully-connected layer

model.add(Dense(1024, name="embedding"))
model.add(Activation("softmax", name="softmax"))

alpha = 0.2

def triplet_loss(y_true, y_pred):
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]

    losses = tf.maximum(0.0, tf.reduce_sum((a - p)**2, 1) - tf.reduce_sum((a - n)**2, 1) + 0.2)

    return tf.reduce_sum(losses)


sgd = SGD(lr=0.25, momentum=0.9)

model.compile(loss=triplet_loss,
              optimizer=sgd)

# from https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
def get_model_memory_usage(bs, model):
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

    total_memory = 4*(bs/1024**3)*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = round(total_memory, 3)
    return gbytes

model.summary()

print("Memory usage:", get_model_memory_usage(batch_size, model), "GB")
print("Press enter if that's okay. If not, type NO and then press enter.")
if input() == "NO":
    exit()

model.fit(np.array(total_triplets)[:,:,:,None], np.zeros([len(total_triplets), 1024]),
              batch_size=batch_size,
              epochs=num_epoch,
              shuffle=True,
              callbacks=[ModelCheckpoint("best_triplet_model_{}.h5".format(str(model_uuid)), save_best_only=True)])