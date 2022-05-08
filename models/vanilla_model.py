from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import plot_model
import pandas as pd
import os

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

# https://github.com/atulapra/Emotion-detection/blob/master/src/emotions.py

# Create the model
# Create the model
# model = Sequential()
#
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=image_shape))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(7, activation='softmax'))
# model.summary()

# model = Sequential()
#
# model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=image_shape))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(2, 2))
# model.add(Dropout(0.2))
#
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
# model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01)))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
#
# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.4))
#
# model.add(Dense(num_classes, activation='softmax'))

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=image_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

os.makedirs('./weights', exist_ok=True)
os.makedirs('./weights/vanilla', exist_ok=True)

# plot_model(model, to_file=f'./visuals/vanilla_arch.png', show_shapes=True, show_layer_names=True)

mcp = ModelCheckpoint('./weights/vanilla/vanilla_model.h5',
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
os.makedirs('./history/vanilla', exist_ok=True)
file_name = './history/vanilla/vanilla_hist_trial.csv'
with open(file_name, mode='w') as f:
    pd.DataFrame(history.history).to_csv(f)
