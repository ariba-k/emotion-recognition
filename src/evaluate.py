from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from skimage import io, transform
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


test_dir = './data/test'
# X_test_gen = get_datagen(test_dir)
# def get_datagen(dataset):
#     return ImageDataGenerator().flow_from_directory(
#         dataset,
#         target_size=(48, 48),
#         color_mode='grayscale',
#         shuffle=True,
#         class_mode='categorical',
#         batch_size=32)

X_test_gen = ImageDataGenerator().flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    shuffle=True,
    class_mode='categorical',
    batch_size=32)
X_test = np.zeros((len(X_test_gen.filepaths), 48, 48, 3))
Y_test = np.zeros((len(X_test_gen.filepaths), 7))
for i in range(0, len(X_test_gen.filepaths)):
    x = io.imread(X_test_gen.filepaths[i], as_gray=True)
    X_test[i, :] = transform.resize(x, (48, 48, 1))
    Y_test[i, X_test_gen.classes[i]] = 1

model_names = ['vanilla', 'resnet50', 'vgg16', 'miniXception']


# test_datagen = ImageDataGenerator(rescale=1. / 255)
# test_dataset = test_datagen.flow_from_directory(directory=test_dir,
#                                                 target_size=(48, 48),
#                                                 class_mode='categorical',
#                                                 batch_size=64)
#

def creat_hist_plots(name):
    # accuracy
    os.makedirs('./visuals', exist_ok=True)
    hist = pd.read_csv(f"./history/{name}/{name}_hist_2.csv")
    hist = hist.to_dict('list')
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 2)
    plt.plot(hist['acc'])
    plt.plot(hist['val_acc'])
    plt.title(f'{name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'valid'], loc='upper left')
    # loss
    plt.subplot(1, 2, 1)
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.title(f'{name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['train', 'valid'], loc='upper left')
    os.makedirs('./results/performance', exist_ok=True)
    os.makedirs(f'./results/performance/{name}', exist_ok=True)
    plt.savefig(f'./results/performance/{name}/{name}_perf_2.png')


def create_confusion_mat(name, labels, predictions):
    target_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    cm = confusion_matrix(labels, predictions)
    # normalize
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 7))
    sns.heatmap(cmn, annot=True, fmt='.3f', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    os.makedirs('./results/confusion', exist_ok=True)
    os.makedirs(f'./results/confusion/{name}', exist_ok=True)
    plt.savefig(f'./results/confusion/{name}/{name}_confusion_mat_2.png')


def create_evaluation_summary(results):
    data = [[str(metric) for metric in model] for model in results]
    columns = ['Parameters (m)', 'Accuracy', 'Precision', 'Recall']
    rows = ['Vanilla', 'ResNet50', 'VGG-16', 'miniXception']
    rcolors = np.full(len(rows), 'linen')
    ccolors = np.full(len(columns), 'lavender')

    max_vals = [max(x) for x in zip(*results)]
    acc_max = max_vals[1]
    prec_max = max_vals[2]
    rec_max = max_vals[3]

    colors = []

    for model in results:
        model_colors = ['w']
        if model[1] == acc_max:
            model_colors.append('g')
        else:
            model_colors.append('w')

        if model[2] == prec_max:
            model_colors.append('g')
        else:
            model_colors.append('w')

        if model[3] == rec_max:
            model_colors.append('g')
        else:
            model_colors.append('w')
        colors.append(model_colors)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    table = ax.table(
        cellText=data,
        cellColours=colors,
        rowLabels=rows,
        colLabels=columns,
        rowColours=rcolors,
        colColours=ccolors,
        cellLoc='center',
        loc='upper left')

    ax.set_title('Model Test Results')
    fig.tight_layout()
    os.makedirs('./results/metrics', exist_ok=True)
    plt.savefig('./results/metrics/eval_metrics_2.png')
    plt.show()


results = []
for name in model_names:
    # accuracy and loss graphs
    creat_hist_plots(name)



#     model = load_model(f'./weights/{name}_model_2.h5')
#     # architecture
#     # plot_model(model, to_file=f'./visuals/{name}_arch_horz_2.png', show_shapes=True, show_layer_names=True,
#     #            rankdir='LR',
#     #            expand_nested=True)
#
#     # prediction and truth sets
#     y_pred_probs = model.predict(X_test)
#     y_pred = y_pred_probs.argmax(axis=1)
#     y_true = Y_test.argmax(axis=1)
#
#     # confusion matrix
#     create_confusion_mat(name, y_true, y_pred)
#
#     # evaluation metrics
#     trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
#     non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
#     tot_params = (trainable_params + non_trainable_params) / 1000000
#
#     precision = precision_score(y_true, y_pred, average='macro')
#     recall = recall_score(y_true, y_pred, average='macro')
#     acc = accuracy_score(y_true, y_pred)
#     results.append([round(tot_params, 2), round(acc, 4), round(precision, 4), round(recall, 4)])
#
# create_evaluation_summary(results)
