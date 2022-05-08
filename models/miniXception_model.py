from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Conv2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, MaxPooling2D, SeparableConv2D, Add
from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import get_file
import os
import pandas as pd

# https://github.com/kumarnikhil936/face_emotion_recognition_cnn
# https://github.com/XiuweiHe/EmotionClassifier/blob/master/src/cnn.py --> consider doing a tiny ALEXNET
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
regularization = l2(0.01)

def xception_block(input_tensor, num_kernels, l2_reg=0.01):
    """Xception core block.
    # Arguments
        input_tenso: Keras tensor.
        num_kernels: Int. Number of convolutional kernels in block.
        l2_reg: Float. l2 regression.
    # Returns
        output tensor for the block.
    """
    residual = Conv2D(num_kernels, 1, strides=(2, 2),
                      padding='same', use_bias=False)(input_tensor)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(
        num_kernels, 3, padding='same',
        kernel_regularizer=l2(l2_reg), use_bias=False)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(num_kernels, 3, padding='same',
                        kernel_regularizer=l2(l2_reg), use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(3, strides=(2, 2), padding='same')(x)
    x = Add()([x, residual])
    return x

def build_xception(
        input_shape, num_classes, stem_kernels, block_kernels, l2_reg=0.01):
    """Function for instantiating an Xception model.
    # Arguments
        input_shape: List corresponding to the input shape of the model.
        num_classes: Integer.
        stem_kernels: List of integers. Each element of the list indicates
            the number of kernels used as stem blocks.
        block_kernels: List of integers. Each element of the list Indicates
            the number of kernels used in the xception blocks.
        l2_reg. Float. L2 regularization used in the convolutional kernels.
    # Returns
        Tensorflow-Keras model.
    # References
        - [Xception: Deep Learning with Depthwise Separable
            Convolutions](https://arxiv.org/abs/1610.02357)
    """

    x = inputs = Input(input_shape, name='image')
    for num_kernels in stem_kernels:
        x = Conv2D(num_kernels, 3, kernel_regularizer=l2(l2_reg),
                   use_bias=False, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    for num_kernels in block_kernels:
        x = xception_block(x, num_kernels, l2_reg)

    x = Conv2D(num_classes, 3, kernel_regularizer=l2(l2_reg),
               padding='same')(x)
    # x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='label')(x)

    model_name = '-'.join(['XCEPTION',
                           str(input_shape[0]),
                           str(stem_kernels[0]),
                           str(len(block_kernels))
                           ])
    model = Model(inputs, output, name=model_name)
    return model

stem_kernels = [32, 64]
block_data = [128, 128, 256, 256, 512, 512, 1024]
model_inputs = (image_shape, num_classes, stem_kernels, block_data)
# model = build_xception(*model_inputs)

URL = 'https://github.com/oarriaga/altamira-data/releases/download/v0.6/'
filename = 'fer2013_mini_XCEPTION.119-0.65.hdf5'
path = get_file(filename, URL + filename, cache_subdir='../weights')
base_model = load_model(path)
base_model.layers[0]._batch_input_shape = (None, 48, 48, 3)
model = model_from_json(base_model.to_json())
model.summary()

for layer in model.layers:
    try:
        layer.set_weights(base_model.get_layer(name=layer.name).get_weights())
    except:
        print("Could not transfer weights for layer {}".format(layer.name))


os.makedirs('./weights', exist_ok=True)
os.makedirs('./weights/miniXception', exist_ok=True)
mcp = ModelCheckpoint('./weights/miniXception/miniXception_model.h5',
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
os.makedirs('./history/miniXception', exist_ok=True)
file_name = './history/miniXception/miniXception_hist.csv'
with open(file_name, mode='w') as f:
    pd.DataFrame(history.history).to_csv(f)
