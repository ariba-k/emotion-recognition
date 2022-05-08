from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, \
    Activation, Input, GlobalAveragePooling2D, GaussianNoise
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
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

# https://www.kaggle.com/code/yasserhessein/emotion-recognition-with-resnet50/notebook
# import pretrained imagenet ResNet50 model
# include_top = False ->  fully-connected output layers of the model used to make predictions is not loaded
# allows a new output layer to be added and trained

image_shape = (48, 48, 3)
num_classes = 7
epochs = 60
lr = 0.0001

# base_model = ResNet50(input_shape=image_shape, include_top=False, weights="imagenet")
#
# for layer in base_model.layers[:-4]:
#     layer.trainable=False
#
# model=Sequential()
# model.add(base_model)
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(BatchNormalization())
# model.add(Dense(32,kernel_initializer='he_uniform'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(32,kernel_initializer='he_uniform'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(32,kernel_initializer='he_uniform'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(7,activation='softmax'))

main_input = Input(image_shape)

x = BatchNormalization()(main_input)
x = GaussianNoise(0.01)(x)

base_model = ResNet50(weights=None, input_tensor=x, include_top=False)

flatten = GlobalAveragePooling2D()(base_model.output)
# flatten = layers.Flatten()(base_model.output)

fc = Dense(2048, activation='relu',
           kernel_regularizer=l2(0.001),
           bias_regularizer=l2(0.001),
           )(flatten)
fc = Dropout(0.25)(fc)
fc = Dense(2048, activation='relu',
           kernel_regularizer=l2(0.001),
           bias_regularizer=l2(0.001),
           )(fc)
fc = Dropout(0.5)(fc)

predictions = Dense(num_classes, activation="softmax")(fc)

model = Model(inputs=main_input, outputs=predictions, name='resnet50')
os.makedirs('./weights', exist_ok=True)
os.makedirs('./weights/resnet50', exist_ok=True)
mcp = ModelCheckpoint('./weights/resnet50/resnet50_model.h5',
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
os.makedirs('./history/resnet50', exist_ok=True)
file_name = './history/resnet50/resnet50_hist.csv'
with open(file_name, mode='w') as f:
    pd.DataFrame(history.history).to_csv(f)
