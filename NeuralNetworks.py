import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#import data
data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#shrink pixel image value to manageable numbers
train_images = train_images/255
test_images = test_images/255

#show images each train image is just an array of pixel values (28x28)
#plt.imshow(train_images[7], cmap=plt.cm.binary)
#plt.show()

#build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), #input layer, flatten input into one-level array
    keras.layers.Dense(128, activation='relu'), #128 neurons in next layer (Dense = fully connected) (relu = rectify linear unit. this is arbitrary, choose any you want)
    keras.layers.Dense(10, activation='softmax') #10 neurons in output layer, (softmax = pick values for each neurons so that all those values add up to one)

])

#setup parameters for model (optimizers, loss functions, metrics)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#train model. Epochs work like if the neural network receives a certain subset of images in a certain, it will tweak its parameters accordingly (ex. see 10 images shirt first, etc.)
# higher epoch != more accuracy, epoch number is arbitrary
model.fit(train_images, train_labels, epochs=10) #epoch = how many times the model is going to see the info (train iimages, and train labels)

#evaluate model to test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Accuracy: ', test_acc) #after last epoch, it shows accuracy on test data