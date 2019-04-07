from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from PIL import Image

size = 200
img_size = (size, size)
image_shape = (size, size, 1)

def createModel(classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=image_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))

    return model



model1 = createModel(10)
batch_size = 256
epochs = 100
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# train_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory('dataset/train')

x_paths = pd.read_csv('dataset/t2_x_train.csv', header=None)
x_imgs = [Image.open("dataset/train/" + fname).resize(img_size).convert('L') for fname in x_paths[0]]
y = np.array(pd.read_csv('dataset/t2_y_train.csv'))

x_test = pd.read_csv('dataset/t2_x_test.csv', header=None)

all_len = len(x_imgs)
valid_len = 400
train_len = all_len - valid_len


x_train = np.array([np.array(x_imgs[i]) for i in range(0, train_len - 1)])
x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
y_train = np.array([y[i] for i in range(0, train_len - 1)])

x_valid = np.array([np.array(x_imgs[i]) for i in range(train_len, all_len - 1)])
x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[1], x_valid.shape[2], 1))
y_valid = np.array([y[i] for i in range(train_len, all_len - 1)])

print("All images  : " + str(all_len))
print("Train images: " + str(train_len) + " " + str(x_train[0].shape))
print("Valid images: " + str(valid_len) + " " + str(x_train[0].shape))


history = model1.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2,
                     validation_data=(x_valid, y_valid)
                     )


#model1.evaluate(test_data, test_labels_one_hot)