"""
    This script splits the images into two views.

    View 1 is used for development. It contains mutually exclusive training and testing sets.
    Its training set has 80% of the images, the testing set has 20% of the images.

    View 2 uses all of the images.
"""
import gzip
import umsgpack
import random
from tqdm import tqdm

subject_map = {}
id_counter = 0

in_file = gzip.open("occluded_db.gz", "rb")

num_images = umsgpack.unpack(in_file)

subject_image_map = umsgpack.unpack(in_file)

training_map = {}
testing_map = {}

print("> Choosing images...")
for subject in tqdm(subject_image_map):
    subject_images = [i for i in range(0, subject_image_map[subject])]
    random.shuffle(subject_images)
    num_training_images = int(subject_image_map[subject] * 0.8)

    training_images = subject_images[:num_training_images]
    testing_images = subject_images[num_training_images:]

    training_map[subject] = training_images
    testing_map[subject] = testing_images

training_total = 0
testing_total = 0

for subject in training_map:
    training_total += len(training_map[subject])

for subject in training_map:
    testing_total += len(testing_map[subject])
    
print("> Training images:", training_total)
print("> Testing images:", testing_total)

training_out = open("v1_training.dat", "wb")
testing_out = open("v1_testing.dat", "wb")

training_out.write(umsgpack.packb(training_total))
testing_out.write(umsgpack.packb(testing_total))

subject_image_current_id = {s: 0 for s in subject_image_map}

def one_hot(sid):
    oh = [0 for i in range(0, len(subject_image_map))]
    oh[sid] = 1

    return oh

print("> Writing data...")
for i in tqdm(range(0, num_images)):
    (subject, image) = umsgpack.unpack(in_file)

    if subject not in subject_map:
        subject_map[subject] = id_counter
        id_counter += 1

    image_id = subject_image_current_id[subject]
    subject_image_current_id[subject] += 1

    if image_id in training_map[subject]:
        training_out.write(umsgpack.packb((image, subject_map[subject])))
    else:
        testing_out.write(umsgpack.packb((image, subject_map[subject])))

mean_image = umsgpack.unpack(in_file)

training_out.write(umsgpack.packb(mean_image))
testing_out.write(umsgpack.packb(mean_image))