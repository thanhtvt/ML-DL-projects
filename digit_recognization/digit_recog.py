# Import lib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

# Reshape data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Label encoding
no_of_class = 10    # There are total 10 class from 0 to 9

y_train = np_utils.to_categorical(y_train, no_of_class)
y_val = np_utils.to_categorical(y_val, no_of_class)
y_test = np_utils.to_categorical(y_test, no_of_class)

# Modeling
model = Sequential()

# Add Conv layer with 32 kernels (filters) size 3x3
# Use sigmoid function as activation function
model.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=(28, 28, 1)))

model.add(Conv2D(32, (3, 3), activation='sigmoid'))

# Add Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add Flatten layer
model.add(Flatten())

# Add Fully Connected layer with 128 nodes using sigmoid function as activation
model.add(Dense(128, activation='sigmoid'))

# Add Output layer with 10 nodes using softmax function to calculate probability
model.add(Dense(no_of_class, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model
hist = model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1,
                          validation_data=(X_val, y_val))
print('Model has been trained successfully!')

# Save result in another file
model.save('mnist.h5')
print('Saving model as mnist.h5')

# Plot loss, accuracy of training set and validation set
no_of_epochs = 10
fig = plt.figure()

plt.plot(np.arange(0, no_of_epochs), hist.history['loss'], label='training loss')
plt.plot(np.arange(0, no_of_epochs), hist.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, no_of_epochs), hist.history['acc'], label='accuracy')
plt.plot(np.arange(0, no_of_epochs), hist.history['val_acc'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()

plt.show()

# Evaluate model
score = model.evaluate(X_test, y_test, verbose=0)
print(score)

# Recognise images
plt.imshow(X_test[0].reshape(28, 28), cmap='gray')

pred = model.predict(X_test[0].reshape(1, 28, 28, 1))
print('This number is: ', np.argmax(pred))
