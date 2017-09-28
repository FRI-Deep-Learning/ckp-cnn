import umsgpack
from tqdm import tqdm
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras import backend as K

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
    return gbytes

# Constants
train_file_name = "v1_training.dat"
test_file_name = "v1_testing.dat"

batch_size = 50
num_classes = 123
num_epoch = 100

img_rows = 64
img_cols = 64

# Load images
train_in = open(train_file_name, "rb")
test_in = open(test_file_name, "rb")

num_train = umsgpack.unpack(train_in)
num_test = umsgpack.unpack(test_in)

x_train = []
y_train = []

def expand_image(image):
    ret = []
    for row in image:
        ret_row = []
        for col in row:
            ret_row.append([col])
        ret.append(ret_row)
    return ret

print("> Loading training data...")
for i in tqdm(range(0, num_train)):
    (image, one_hot) = umsgpack.unpack(train_in)
    x_train.append(expand_image(image))
    y_train.append(one_hot)

x_test = []
y_test = []

print("> Loading testing data...")
for i in tqdm(range(0, num_test)):
    (image, one_hot) = umsgpack.unpack(test_in)
    x_test.append(expand_image(image))
    y_test.append(one_hot)

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

mean_image = umsgpack.unpack(test_in)

mean_image = np.array(mean_image)

print(mean_image.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

mean_image = expand_image(mean_image)

mean_image = np.array(mean_image)

print(mean_image.shape)

x_train -= mean_image
x_test -= mean_image

x_train /= 255
x_test /= 255

# Set up output data in categorical matrices

y_train = np_utils.to_categorical(y_train, num_classes)
y_test  = np_utils.to_categorical(y_test, num_classes)

# Build the network model

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=(64, 64, 1)))
model.add(Activation("relu"))

model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation("relu"))

model.add(Conv2D(256, (3, 3)))
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512, (3, 3)))
model.add(Activation("relu"))

model.add(Conv2D(512, (3, 3)))
model.add(Activation("relu"))

model.add(Conv2D(512, (3, 3)))
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(4096, (1, 1)))
model.add(Activation("relu"))

model.add(Flatten())
model.add(Dense(num_classes))

model.add(Activation("softmax"))

preds = model.predict(np.ones((1, 64, 64, 1)))
print(preds.shape)

# Compile the model and put data between 0 and 1

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print("Memory usage:", get_model_memory_usage(batch_size, model), "GB")
print("Press enter if that's okay. If not, type NO and then press enter.")
if input() == "NO":
    exit()

print(x_train.shape)
print(y_train.shape)

# Train the model

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=num_epoch,
              validation_data=(x_test, y_test),
              shuffle=True)

model.save("finished_model.hdf5")