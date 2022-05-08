from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation, GlobalMaxPooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
import pandas as pd
import os

# VGG19 MODEL: https://www.kaggle.com/code/kaledibif/fer-project-with-alexnet-vgg19-and-resnet-18

# VGG Based Model: https://www.kaggle.com/code/vumerenko/vgg-based-architecture-0-70-test-acc

# VGG16 Resized Model: https://github.com/amilkh/cs230-fer/tree/master/models

# https://github.com/CNC-IISER-BHOPAL/Tiny-ImageNet-Visual-Recognition-Challenge-IITK/blob/master/custom_vgg16_for_tiny_imagenet.ipynb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_dir = './data/train'

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   validation_split=0.2,
                                   featurewise_center=False,
                                   featurewise_std_normalization=False,
                                   rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1. / 255,
                                   validation_split=0.2)

train_dataset = train_datagen.flow_from_directory(directory=train_dir,
                                                  target_size=(48, 48),
                                                  class_mode='categorical',
                                                  subset='training',
                                                  batch_size=64)

valid_dataset = valid_datagen.flow_from_directory(directory=train_dir,
                                                  target_size=(48, 48),
                                                  class_mode='categorical',
                                                  subset='validation',
                                                  batch_size=64)

image_shape = (48, 48, 3)
num_classes = 7
epochs = 60
lr = 0.0001

base_model = VGG16(input_shape=image_shape, include_top=False, weights='imagenet')
for layer in base_model.layers[:-4]:
    layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(Dropout(0.5))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(32, kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32, kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32, kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(7, activation='softmax'))
model.summary()

os.makedirs('./weights', exist_ok=True)
os.makedirs('./weights/vgg16', exist_ok=True)
mcp = ModelCheckpoint('./weights/vgg16_model.h5',
                      save_best_only=True,
                      verbose=1,
                      mode='min',
                      moniter='val_loss')

lrd = ReduceLROnPlateau(monitor='val_loss',
                        factor=0.2,
                        patience=6,
                        verbose=1,
                        min_delta=0.0001)
es = EarlyStopping(monitor='val_loss',
                   min_delta=0,
                   patience=3,
                   verbose=1,
                   restore_best_weights=True)

steps_per_epoch = train_dataset.n // train_dataset.batch_size
validation_steps = train_dataset.n // train_dataset.batch_size
sgd = SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=True)
adam = Adam(lr=lr, decay=1e-6)

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x=train_dataset,
                    validation_data=valid_dataset,
                    epochs=epochs,
                    callbacks=[lrd, mcp],
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps)

os.makedirs('./history', exist_ok=True)
os.makedirs('./history/vgg16', exist_ok=True)
file_name = './history/vgg16/vgg16_hist.csv'
with open(file_name, mode='w') as f:
    pd.DataFrame(history.history).to_csv(f)
