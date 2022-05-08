from fer import FER
import cv2
import os
import csv
import tensorflow as tf
#  SCRIPT NOT USED
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

data_path = '../data/'
test_data_path = data_path + 'test/'
train_data_path = data_path + 'train/'
emo_folders = [name for name in os.listdir(train_data_path) if os.path.isdir(os.path.join(train_data_path, name))]
emo_folders.sort()

emo_results = {}
for emotion in emo_folders:
    images_folder = os.listdir(test_data_path + emotion)
    # name dependent since numbers come after 'im' and before '.png'
    # this is for natural sorting
    images_folder.sort(key=lambda x: int(x[2:-4]))
    images_per_emotion = []
    for image_name in images_folder[:5]:
        image_path = os.path.join(test_data_path + emotion, image_name)
        image_results = {'image': image_name}
        image = cv2.imread(image_path)
        # emotion_model = '' -> override given model w/ path to trained model weights (hdf5)
        emo_detector = FER(mtcnn=True)
        # Capture all the emotions on the image
        emotion_info = emo_detector.detect_emotions(image)
        if emotion_info:
            captured_emotions, = emotion_info
            emotions = dict(sorted(captured_emotions['emotions'].items()))
            # Use the top Emotion() function to call for the dominant emotion in the image
            dominant_emotion, emotion_score = emo_detector.top_emotion(image)
        else:
            emotions = {emo: None for emo in emo_folders}
            dominant_emotion = None

        emotions['prediction'] = dominant_emotion
        image_results.update(emotions)
        images_per_emotion.append(image_results)

    emo_results[emotion] = images_per_emotion

os.makedirs('../results', exist_ok=True)
results_path = '../results/'
for emo in emo_results:
    csv_name = '/generic.csv'
    csv_path = results_path + emo + csv_name
    os.makedirs(results_path + emo, exist_ok=True)
    with open(csv_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(emo_results.keys()))
        writer.writeheader()
        writer.writerows([emo_results])

# LOAD MODEL of FER --> https://www.tensorflow.org/guide/keras/save_and_serialize

# DESCRIBES HUMAN PERFORMANCE --> https://arxiv.org/pdf/1307.0414.pdf
