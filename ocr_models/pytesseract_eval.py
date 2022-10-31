import numpy as np
import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class PytesseractEval():
    def __init__(self, custom_config=None, confidence=50):
        if custom_config == None:
            self.custom_config = r'-l eng+grc -c tessedit_char_blacklist=0123456789 --psm 11'
        else:
            self.custom_config = custom_config
        self.confidence = confidence
        self.name = "pytesseract_" + self.custom_config + "_" + str(self.confidence)

    def predict(self, img):
        img = img[0]
        result = []
        d = pytesseract.image_to_data(img, output_type=Output.DICT, config=self.custom_config)
        n_boxes = len(d['text'])
        # for i in range(n_boxes):
        #     if int(d['conf'][i]) > self.confidence:
        #         (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        #         # (x, y, w, h) = (x-5, y-5, w+10, h+10)
        #         crop_img = img[max(0,y):min(img.shape[1],y+h), max(0,x):min(img.shape[0],x+w)]
        #         result.append(pytesseract.image_to_string(crop_img, config=self.custom_config))
        for i in range(n_boxes):
            if int(d['conf'][i]) > self.confidence:
                result.append(d['text'][i])


        res = [ee for e in result for ee in e.strip().split("\n")]

        return [list(filter(("").__ne__, [e.strip().replace("\n", "") for e in res]))]