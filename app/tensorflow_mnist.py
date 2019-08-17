import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import keras
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

### configuring tensorboard
NAME = 'data-model-%d' % int(time.time())
tensorboard = TensorBoard(log_dir= 'logs/%s' % NAME)

### loading dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=13)

### normalizing dataset
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
val_images = val_images.reshape(val_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
val_images = val_images.astype('float32') / 255

test = {"values": [test_images[0].tolist()]}

print(test)
### creating model
input_shape = (28, 28, 1)
model = keras.Sequential([
    keras.layers.Conv2D(28, kernel_size=(3,3), input_shape=input_shape),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', 'sparse_categorical_crossentropy'])
model.summary()

### training and saving model
history = model.fit(train_images, train_labels, batch_size=256, epochs=14,
                    validation_data=(val_images, val_labels), verbose=1, callbacks=[tensorboard])
model.save('model.h5')
#!tar -zcvf model.tgz model.h5

### loading model
# model = keras.models.load_model('model.h5')

### testing model
test_loss, test_acc, test_acc2 = model.evaluate(test_images, test_labels)

print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
