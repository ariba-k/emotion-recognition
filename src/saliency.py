import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from skimage import io, transform
from matplotlib.colors import BoundaryNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib import cm
from PIL import Image
import os

# https://github.com/amilkh/cs230-fer/blob/master/error-analysis.ipynb

os.makedirs('./results/saliency', exist_ok=True)
os.makedirs('./results/saliency/normal', exist_ok=True)
os.makedirs('./results/saliency/heatmap', exist_ok=True)

model = load_model('./weights/vanilla/vanilla_model.h5')

test_dir = './data/test'

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
    X_test[i, :] = transform.resize(x, (48, 48, 3))
    Y_test[i, X_test_gen.classes[i]] = 1


def iter_occlusion(image, size=8):
    occlusion = np.full((size * 5, size * 5, 1), [0.5], np.float32)
    occlusion_center = np.full((size, size, 1), [0.5], np.float32)
    occlusion_padding = size * 2

    # print('padding...')
    image_padded = np.pad(image,
                          ((occlusion_padding, occlusion_padding), (occlusion_padding, occlusion_padding), (0, 0)),
                          'constant', constant_values=0.0)

    for y in range(occlusion_padding, image.shape[0] + occlusion_padding, size):
        for x in range(occlusion_padding, image.shape[1] + occlusion_padding, size):
            tmp = image_padded.copy()

            tmp[y - occlusion_padding:y + occlusion_center.shape[0] + occlusion_padding, \
            x - occlusion_padding:x + occlusion_center.shape[1] + occlusion_padding] \
                = occlusion

            tmp[y:y + occlusion_center.shape[0], x:x + occlusion_center.shape[1]] = occlusion_center

            yield x - occlusion_padding, y - occlusion_padding, \
                  tmp[occlusion_padding:tmp.shape[0] - occlusion_padding,
                  occlusion_padding:tmp.shape[1] - occlusion_padding]


emotions_dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
# len(X_test)
# 0, 30
# 958, 988
# 1069, 1097
# 2093, 2122
# 3867, 3896
# 5097, 5127
# 7128, 7158
for i in range(len(X_test)):
    data = X_test[i]
    correct_class = np.argmax(Y_test[i])
    # img_ = Image.fromarray((data * 255).astype(np.uint8))
    # img_.save(f'./results/saliency/normal/{i}_{emotions_dict[correct_class]}.png')

    # input tensor for model.predict
    inp = data.reshape(1, 48, 48, 3)
    # image data for matplotlib's imshow
    img = data.reshape(48, 48, 3)
    # occlusion
    img_size = img.shape[0]
    occlusion_size = 4
    _ = plt.imshow(img, cmap='gray')
    # print('occluding...')

    heatmap = np.zeros((img_size, img_size), np.float32)
    class_pixels = np.zeros((img_size, img_size), np.int16)

    counters = defaultdict(int)

    for n, (x, y, img_float) in enumerate(iter_occlusion(data, size=occlusion_size)):
        X = img_float.reshape(1, 48, 48, 3)
        out = model.predict(X)
        # print('#{}: {} @ {} (correct class: {})'.format(n, np.argmax(out), np.amax(out), out[0][correct_class]))
        # print('x {} - {} | y {} - {}'.format(x, x + occlusion_size, y, y + occlusion_size))

        heatmap[y:y + occlusion_size, x:x + occlusion_size] = out[0][correct_class]
        class_pixels[y:y + occlusion_size, x:x + occlusion_size] = np.argmax(out)
        counters[np.argmax(out)] += 1

    pred = model.predict(inp)
    # print('Correct class: {}'.format(correct_class))
    # print('Predicted class: {} (prob: {})'.format(np.argmax(pred), np.amax(out)))

    # print('Predictions:')
    # for class_id, count in counters.items():
    #     print('{}: {}'.format(class_id, count))

    heatmap = 1 - heatmap

    # fig = plt.figure(figsize=(8, 8))
    #
    # ax1 = plt.subplot(1, 2, 1, aspect='equal')
    # hm = ax1.imshow(heatmap)
    #
    # ax2 = plt.subplot(1, 2, 2, aspect='equal')
    #
    # vals = np.unique(class_pixels).tolist()
    # bounds = vals + [vals[-1] + 1]  # add an extra item for cosmetic reasons
    #
    # custom = cm.get_cmap('Greens', len(bounds))  # discrete colors
    #
    # norm = BoundaryNorm(bounds, custom.N)
    #
    # cp = ax2.imshow(class_pixels, norm=norm, cmap=custom)
    #
    # divider = make_axes_locatable(ax1)
    # cax1 = divider.append_axes("right", size="5%", pad=0.05)
    # cbar1 = plt.colorbar(hm, cax=cax1)
    #
    # divider = make_axes_locatable(ax2)
    # cax2 = divider.append_axes("right", size="5%", pad=0.05)
    # cbar2 = ColorbarBase(cax2, cmap=custom, norm=norm,
    #                      # place the ticks at the average value between two entries
    #                      # e.g. [280, 300] -> 290
    #                      # so that they're centered on the colorbar
    #                      ticks=[(a + b) / 2.0 for a, b in zip(bounds[::], bounds[1::])],
    #                      boundaries=bounds, spacing='uniform', orientation='vertical')
    #
    # cbar2.ax.set_yticklabels([n for n in np.unique(class_pixels)])
    #
    # fig.tight_layout()

    # plt.show()
    # norm = cm.colors.Normalize(vmin=0, vmax=1)
    plt.figure(figsize=(6, 6))

    plt.imshow(img, cmap=cm.gray)
    plt.pcolormesh(heatmap, cmap=plt.cm.jet, alpha=0.50)
    # plt.pcolormesh(heatmap, cmap=plt.cm.jet, alpha=0.50, norm=norm)

    # v = np.linspace(0, 1, 7, endpoint=True)

    # plt.colorbar(norm=norm).solids.set(alpha=1)
    plt.colorbar().solids.set(alpha=1)
    # os.makedirs('./results/saliency/heatmap_adj', exist_ok=True)
    os.makedirs('./results/saliency/heatmap', exist_ok=True)

    plt.savefig(
        f'./results/saliency/heatmap/{i}_{emotions_dict[correct_class]}_{emotions_dict[np.argmax(pred)]}.png')
    # plt.show()
