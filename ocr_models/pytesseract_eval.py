# from eval_interface import EvalInterface
# from preprocessing.preprocessor import Preprocessor
import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class PytesseractEval():
    def __init__(self, 
                preprocessor=None, 
                custom_config=None, confidence=50):
        if custom_config == None:
            self.custom_config = r'-l eng+grc -c tessedit_char_blacklist=0123456789 --psm 11'
        else:
            self.custom_config = custom_config
        self.confidence = confidence
        self.preprocessor = preprocessor
        self.name = f"pytesseract_{self.custom_config}_{self.confidence}_{self.preprocessor.name}"

    def predict(self, img):
        # img = img[0]
        img = self.preprocessor.preprocess(img)
        result = []
        d = pytesseract.image_to_data(img, output_type=Output.DICT, config=self.custom_config)
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(d['conf'][i]) > self.confidence:
                result.append(d['text'][i])

        res = [ee for e in result for ee in e.strip().split("\n")]

        return list(filter(("").__ne__, [e.strip().replace("\n", "") for e in res]))

    def get_name(self):
        return self.name