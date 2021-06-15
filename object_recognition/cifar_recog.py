# Import lib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator  # Data augmentation
from tensorflow.keras.optimizers import SGD


def load_dataset():
    # Load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Label encoding
    no_of_class = 10  # There are total 10 class

    y_train = np_utils.to_categorical(y_train, no_of_class)
    y_val = np_utils.to_categorical(y_val, no_of_class)
    y_test = np_utils.to_categorical(y_test, no_of_class)
    return X_train, y_train, X_val, y_val, X_test, y_test


def prep_pixels(X_train, X_val, X_test):
    # Reshape data
    X_prep_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
    X_prep_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
    X_prep_val = X_val.reshape(X_val.shape[0], 32, 32, 3)

    # Normalizing data
    X_prep_train /= 255.0
    X_prep_test /= 255.0
    X_prep_val /= 255.0

    return X_prep_train, X_prep_val, X_prep_test


def define_model():
    # Modeling use 3 VGGs
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                     padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())     # Add batch normalization to stabilize the learning
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))             # Add Dropout layer to regularize, reduce overfitting
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # Compile model
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def data_augmentation(X_train, y_train, batch_size=64):
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                                 horizontal_flip=True)
    generator = datagen.flow(X_train, y_train, batch_size=batch_size)
    steps = X_train.shape[0] // batch_size
    return steps, generator


def save_model(model):
    model.save('cifar10.h5')
    print('Model saved successfully!')


def training_process_plotting(hist):
    # Plot loss, accuracy of training set and validation set
    plt.figure(figsize=(18, 4))

    plt.subplot(121)
    plt.title('Loss')
    plt.plot(hist.history['loss'], label='training loss')
    plt.plot(hist.history['val_loss'], label='validation loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(122)
    plt.title('Accuracy')
    plt.plot(hist.history['accuracy'], label='training accuracy')
    plt.plot(hist.history['val_accuracy'], label='validation accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('Learning rate.png')
    plt.show()


def evaluate_model(X_test, y_test, model):
    # Evaluate model
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Loss: ', score[0])
    print('Accuracy: ', round(score[1] * 100, 2), '%')


if __name__ == '__main__':
    # Load dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Prepare data
    X_train, X_val, X_test = prep_pixels(X_train, X_val, X_test)

    # Define model
    model = define_model()

    # Generate new data to regularize, expand training dataset
    steps, gen = data_augmentation(X_train, y_train)

    # Train model
    hist = model.fit_generator(gen, steps_per_epoch=steps, epochs=400,
                               validation_data=(X_val, y_val), verbose=0)
    print('Model has been trained successfully!')

    # Save model
    save_model(model)

    # Plotting
    training_process_plotting(hist)

    # Evaluate model
    evaluate_model(X_test, y_test, model)   # Loss: 0.39450207352638245
                                            # Accuracy: 87.4%
