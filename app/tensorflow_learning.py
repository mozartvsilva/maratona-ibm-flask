import tensorflow as tf
import sklearn
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import time

NAME = 'data-model-%d' % int(time.time())

tensorboard = TensorBoard(log_dir= 'logs/%s' % NAME)
print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=13)

### normalize dataset
original_test_images = test_images
original_test_labels = test_labels
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
val_images = val_images.reshape(val_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
val_images = val_images.astype('float32') / 255

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
# plt.savefig('demo.png')


def create_model(optimizer = 'adam'):
    model = keras.Sequential([
        keras.layers.Conv2D(28, kernel_size=(3,3), input_shape=input_shape),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'sparse_categorical_crossentropy'])
    return model


def plot_history(histories, key='sparse_categorical_crossentropy'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    plt.savefig('history.png')


model = create_model()
model.summary()

### loading model
# model = keras.models.load_model('my_model.h5')

### training model
history = model.fit(train_images, train_labels, batch_size=256, epochs=14,
                    validation_data=(val_images, val_labels), verbose=1, callbacks=[tensorboard])

### training model with data augmentation
# gen = keras.preprocessing.image.ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
#                                height_shift_range=0.08, zoom_range=0.08)
# batches = gen.flow(train_images, train_labels, batch_size=256)
# val_batches = gen.flow(val_images, val_labels, batch_size=256)

# history = model.fit_generator(batches, steps_per_epoch=48000//256, epochs=50,
#                     validation_data=val_batches, validation_steps=12000//256, use_multiprocessing=True)

### grid search model
# optimizers = ['rmsprop', 'adam']
# # inits = ['glorot_uniform', 'normal', 'uniform']
# epochs = np.array([5])
# batches = np.array([256])

# model = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=1)

# param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches)
# grid = sklearn.model_selection.GridSearchCV(estimator=model, param_grid=param_grid)
# grid_result = grid.fit(train_images, train_labels)
# print('best_score_', grid_result.best_score_)
# print('best_params_', grid_result.best_params_)

### saving model
model.save('my_model.h5')

plot_history([('baseline', history)])

### evaluate model
test_loss, test_acc, test_acc2 = model.evaluate(test_images, test_labels)

print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

img = test_images[12]
img = (np.expand_dims(img, 0))
print(img.shape)

predictions_single = model.predict(img)
prediction_result = np.max(predictions_single[0])
prediction_label = np.argmax(predictions_single[0])
print(predictions_single[0])
print(prediction_result)
print(class_names[prediction_label])


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


num_rows = 10
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, original_test_labels, original_test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, original_test_labels)
plt.savefig('demo.png')
