import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import color
from sklearn.utils import shuffle

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.optimizers import RMSprop



def load_data(data_dir):
    """Loads a data set and returns two lists:

    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    labels = []
    images = []
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.

    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels


def train_dataset(images,labels):
    # Resize images
    images32 = [skimage.transform.resize(image, (32, 32), mode="constant")
                for image in images]
    display_images_and_labels(images32, labels)
    # 图像已归一化
    for image in images32[:5]:
        print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))
    # 将图像转为numpy array
    labels_a = np.array(labels)
    images_a = np.array(images32)

    # 将图像转为灰度图
    images_a = color.rgb2gray(images_a)
    display_images_and_labels(images_a, labels)
    print("labels: ", labels_a.shape, "\nimages: ", images_a.shape)
    # 数据集增强
    images_a, labels_a = expend_training_data(images_a, labels_a)
    print(images_a.shape, labels_a.shape)
    labels = labels_a.tolist()
    print(len(labels))
    # 展示增强后图片
    plot_agument(images_a, labels)
    return images_a,labels_a


def display_images_and_labels(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()


def expend_training_data(train_x, train_y):
    """
    数据集增强，通过随机的平移和旋转扩充数据
    """
    expanded_images = np.zeros([train_x.shape[0] * 5, train_x.shape[1], train_x.shape[2]])
    expanded_labels = np.zeros([train_x.shape[0] * 5])

    counter = 0
    for x, y in zip(train_x, train_y):

        # register original data
        expanded_images[counter, :, :] = x
        expanded_labels[counter] = y
        counter = counter + 1

        # get a value for the background
        # zero is the expected value, but median() is used to estimate background's value
        bg_value = np.median(x)  # this is regarded as background's value

        for i in range(4):
            # rotate the image with random degree
            angle = np.random.randint(-15, 15, 1)
            new_img = ndimage.rotate(x, angle, reshape=False, cval=bg_value)

            # shift the image with random distance
            shift = np.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img, shift, cval=bg_value)

            # register new training data
            expanded_images[counter, :, :] = new_img_
            expanded_labels[counter] = y
            counter = counter + 1

    return expanded_images, expanded_labels


def plot_agument(images_a, labels):
    plt.figure(figsize=(16, 9))
    unique_labels = set(labels)
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        if i > 3:
            break
        img_index = labels.index(label)
        for j in range(5):
            image = images_a[img_index+j]
            plt.subplot(3, 5, (i-1)*5 + j+1)  # A grid of 8 rows x 8 columns
            plt.axis('off')
            plt.title("Label {0} ({1})".format(label, labels.count(label)))
            _=plt.imshow(image, cmap='gray')
        i += 1
    plt.show()


def shuffle_data(images_a,labels_a):
    """打乱数据，并分为训练集和验证集"""
    indx = np.arange(0, len(labels_a))
    indx = shuffle(indx)
    images_a = images_a[indx]
    labels_a = labels_a[indx]

    print(images_a.shape, labels_a.shape)
    train_x, val_x = images_a[:20000], images_a[20000:]
    train_y, val_y = labels_a[:20000], labels_a[20000:]

    train_y = keras.utils.to_categorical(train_y, 62)
    val_y = keras.utils.to_categorical(val_y, 62)
    print(train_x.shape, train_y.shape)
    print(val_x.shape,val_y.shape)
    return train_x,train_y,val_x,val_y


def build_model(train_x,train_y,val_x,val_y):
    """构造神经网络"""
    model = Sequential()
    model.add(Flatten(input_shape=(32, 32)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(62, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(train_x, train_y,
                        batch_size=128,
                        epochs=20,
                        verbose=2,
                        validation_data=(val_x, val_y))

    # print the keys contained in the history object
    print(history.history.keys())
    model.save('model.h5')
    return history


def plot_training(history):
    """plot the training and validation loss for each epoch"""
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def test_dataset(test_data_dir):
    """Load the test dataset"""
    test_images, test_labels = load_data(test_data_dir)
    # Transform the images, just like we did with the training set.
    test_images32 = [skimage.transform.resize(image, (32, 32),mode = "constant")
                     for image in test_images]

    test_images_a = np.array(test_images32)
    test_labels_a = np.array(test_labels)

    test_images_a = color.rgb2gray(test_images_a)

    display_images_and_labels(test_images_a, test_labels)

    test_x = test_images_a
    test_y = keras.utils.to_categorical(test_labels_a, 62)
    return test_x,test_y,test_labels_a


def display_prediction(test_x,test_labels_a,predicted):
    """Display the predictions and the ground truth visually."""
    fig = plt.figure(figsize=(10, 10))
    j = 1
    for i in range(0, 1000, 50):
        truth = test_labels_a[i]
        prediction = predicted[i]
        plt.subplot(5, 4, j)
        j = j + 1
        plt.axis('off')
        color = 'green' if truth == prediction else 'red'
        plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
                 fontsize=12, color=color)
        plt.imshow(test_x[i], cmap='gray')




