import os
import cv2
import pandas as pd
import json
import numpy as np
import datetime
from models_ocr.pytesseract_model import PytesseractModel
from models_ocr.preprocessing.preprocessor import Preprocessor

data_path = r'raw_data\helvetios_challenge_dataset_training'
results_csv = r'benchmarking.csv'

preprocessor = Preprocessor(deskew=True)
config = r'-l eng -c tessedit_char_blacklist=0123456789 --psm 11'
confidence = 3
model = PytesseractModel(preprocessor, confidence=confidence, custom_config=config)

img_path = data_path + r'\images'
labels_path = data_path + r'\labels\labels.csv'

img_names = os.listdir(img_path)

df_labels = pd.read_csv(labels_path)
df_labels.fillna("[]", inplace=True)
df_labels["legend"] = df_labels["legend"].apply(lambda x: json.loads(x.replace(' ', ', ').replace('\'', '\"')))
labels = np.array(df_labels["legend"])

imgs = []
for img_name in df_labels["sample_name"]:
    img = cv2.imread(os.path.join(img_path,img_name))
    if img is not None:
        imgs.append(img)

predictions = []
t0 = datetime.datetime.now()
for img in imgs:
    predictions.append(model.predict(img))
runtime = datetime.datetime.now() - t0

def simple_model_comparison(list_of_labels, list_of_ocr_results):
    """both inputs as list of lists"""
    scores = []
    for (label_list, result_list) in zip(list_of_labels, list_of_ocr_results):
        detected_count = 0
        for word in label_list:
            detected_count += (word in result_list)
        detected_relative = detected_count / len(label_list) if len(label_list)!=0 else 1
        scores.append(detected_relative)
    return sum(scores)/len(scores)

score = simple_model_comparison(labels, predictions)

print(
    f"""
    Runtime: {runtime} for {len(imgs)} images
    Score: {score}
    """
)

df = pd.DataFrame({"Model": [model.name], "Runtime": [runtime], "Score": [score]})

df.to_csv("model_evaluation.csv", mode='a', index=False, header=False)