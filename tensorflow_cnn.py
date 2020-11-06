# import tensorflow library
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# import matplotlib library
import matplotlib.pyplot as plt


# download and prepare the CIFAR10 dataset

# the CIFAR10 dataset contains 60,000 colour images in 10 classes, with 6,000 images in each class
# the dataset is divided into 50,000 training images and 10,000 testing images
# the classes are mutually exclusive and there is no overlap between between them
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images = train_images / 255.0 #Normalise pixel values to be between 0 and 1
test_images = test_images / 255.0 #Normalise pixel values to be between 0 and 1


# verify the data

# to verify the dataset looks correct, let's plot the first 25 images from the training set and display the class name below each image
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(10, 10))

for each_index in range (25):
    plt.subplot(5, 5, each_index + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[each_index], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[each_index][0]]) # the CIFAR10 labels happen to be arrays which is why we need the extra index

plt.show()


# create the convolutional base

# as input, a CNN takes tensors of shape (image_height, image_width, colour_channels), ignoring the batch size
# colour_channels refers to (red, green, blue)
# we will configure our CNN to process inputs of shape (32, 32, 3) which is the format of CIFAR images
# we can do this by passing the argument `input_shape` to our first layer
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary() # display the architecture of our model

"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 30, 30, 32)        896
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 13, 13, 64)        18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 4, 4, 64)          36928
=================================================================
Total params: 56,320
Trainable params: 56,320
Non-trainable params: 0
_________________________________________________________________
"""

# we can see that the output of every Conv2D and MaxPooling2D is a 3D tensor of shape (height, width, channels)
# the width and height dimensions tend to shrink as we go deeper in the network
# the number of output channels for each Conv2D layer is controlled by the first argument (e.g. 32 or 64)
# as the width and height shrink, we can afford (computationally) to add more output channels in each Conv2D layer


# add dense layers on top

# to complete our model, we will feed the last output tensor from the convolutional base of shape (4, 4, 64) into one or more Dense layers to perform classification
# Dense layers take vectors as input which are 1D, while the current output is a 3D tensor
# first, we will flatten (or unroll) the 3D output to 1D, then add one or more Dense layers on top
# CIFAR has 10 output classes, so we will use a final Dense layer with 10 outputs and a softmax activation
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary() # display the complete architecture of our model

"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 30, 30, 32)        896
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 13, 13, 64)        18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 4, 4, 64)          36928
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0
_________________________________________________________________
dense (Dense)                (None, 64)                65600
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650
=================================================================
Total params: 122,570
Trainable params: 122,570
Non-trainable params: 0
_________________________________________________________________
"""

# our (4, 4, 64) outputs were flattened into vectors of shape (1,024) before going through two Dense layers


# compile and train the model

num_epochs = 10
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=num_epochs, validation_data=(test_images, test_labels))

"""
Epoch 1/10
1563/1563 [==============================] - 61s 39ms/step - loss: 1.5153 - accuracy: 0.4438 - val_loss: 1.2795 - val_accuracy: 0.5360
Epoch 2/10
1563/1563 [==============================] - 62s 40ms/step - loss: 1.1354 - accuracy: 0.5962 - val_loss: 1.0688 - val_accuracy: 0.6230
Epoch 3/10
1563/1563 [==============================] - 62s 39ms/step - loss: 0.9872 - accuracy: 0.6508 - val_loss: 1.0157 - val_accuracy: 0.6471
Epoch 4/10
1563/1563 [==============================] - 64s 41ms/step - loss: 0.8915 - accuracy: 0.6875 - val_loss: 0.9017 - val_accuracy: 0.6870
Epoch 5/10
1563/1563 [==============================] - 61s 39ms/step - loss: 0.8117 - accuracy: 0.7173 - val_loss: 0.8681 - val_accuracy: 0.7050
Epoch 6/10
1563/1563 [==============================] - 63s 40ms/step - loss: 0.7512 - accuracy: 0.7364 - val_loss: 0.8764 - val_accuracy: 0.6949
Epoch 7/10
1563/1563 [==============================] - 62s 40ms/step - loss: 0.6925 - accuracy: 0.7568 - val_loss: 0.8613 - val_accuracy: 0.7063
Epoch 8/10
1563/1563 [==============================] - 61s 39ms/step - loss: 0.6444 - accuracy: 0.7759 - val_loss: 0.8759 - val_accuracy: 0.7050
Epoch 9/10
1563/1563 [==============================] - 62s 40ms/step - loss: 0.6021 - accuracy: 0.7875 - val_loss: 0.8617 - val_accuracy: 0.7209
Epoch 10/10
1563/1563 [==============================] - 61s 39ms/step - loss: 0.5557 - accuracy: 0.8034 - val_loss: 0.8971 - val_accuracy: 0.7062
"""


# evaluate the model

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

"""
313/313 - 3s - loss: 0.8971 - accuracy: 0.7062
"""
