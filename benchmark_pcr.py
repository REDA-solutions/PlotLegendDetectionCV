# %%
from ocr_models.pytesseract_eval import PytesseractEval
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os

import pandas as pd
from time import perf_counter

from itertools import product as cart_prod
from tqdm import tqdm
from typing import Sequence, Callable
import torch
from torchvision.transforms.functional import pil_to_tensor

# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor


# def get_word_predictions_from_doctr(result_doc):
#         pages = result_doc.pages
#         words = []
#         for page in pages:
#             blocks = page.blocks
#             words_for_page = []
#             for block in blocks:
#                 for line in block.lines:
#                     for word in line.words:
#                         words_for_page.append(word.value)
#             words.append(words_for_page)

#         return words
# %%

# %%
class ImageDataset(Dataset):
    def __init__(self, image_path, label_df):
        self.image_path = image_path
        self.image_names = os.listdir(image_path)
        self.label_df = label_df

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_names = self.image_names[idx]
        image = Image.open(self.image_path + "/" + image_names)
        # image_arr = pil_to_tensor(image)
        image_arr = np.array(image)

        label = self.label_df.loc[self.label_df["sample_name"] == image_names, "legend"].values[0]
        label = label.split(" ") if not isinstance(label, float) else list()
        label = [word.replace("'", "").replace('"', "").replace("[", "").replace("]", "") for word in label]

        return image_arr, label

# %%
def mycollator(batch):
    return (
        np.stack([image for image, _ in batch]),
        [label for _, label in batch],
    )

# %%

# %%
def get_benchmark_df(
    data_loader: DataLoader,
    pred_funcs: Sequence[Callable],
    post_funcs: Sequence[Callable],
    model_names: Sequence[str]
) -> pd.DataFrame:
    n_batches = len(ds)//batch_size

    benchmark_dict = {
        "name": [],
        "time": [],
        "score": []
    }
    
    for pred_func, post_func, model_name in zip(pred_funcs, post_funcs, model_names):
        for img, labels in tqdm(data_loader, total=n_batches):
            t0 = perf_counter()
            preds = pred_func(img)
            dt = perf_counter() - t0

            preds = post_func(preds)

            scores = []
            for pred, label in zip(preds, labels):
                scores += [sum([1 for word in pred if word in label]) / len(label) if len(label) != 0 else 1]
            
        
            benchmark_dict["name"].append(model_name)
            benchmark_dict["time"].append(dt)
            benchmark_dict["score"].append(np.mean(scores))
            # print("=================================")
            # print(f"model: {model_name}")
            # print(f"time: {dt}")
            # print(f"mean score: {np.mean(scores)}")
            # print()


    return pd.DataFrame(data=benchmark_dict)
if __name__ == '__main__':
    train_image_path = "raw_data\helvetios_challenge_dataset_training\images"
    train_label_path = "raw_data\helvetios_challenge_dataset_training\labels"

    batch_size = 1
    n_workers = 0

# %%
    label_df = pd.read_csv(train_label_path + "\labels.csv")

    ds = ImageDataset(train_image_path, label_df)
    data_loader = DataLoader(ds, batch_size=batch_size, num_workers=n_workers, collate_fn=mycollator, shuffle=False)
    from ocr_models.pytesseract_eval import PytesseractEval
    # %%
    pte = PytesseractEval(confidence=15)
    pred_funcs = [pte.predict]
    post_funcs = [lambda x: x]
    model_names = [pte.name]

    # %%
    
    # DOCTR
    
    # det_archs = [
    #     #"db_resnet34",
    #     "db_mobilenet_v3_large",
    #     "db_resnet50",
    #     #"linknet_resnet18",
    #     #"linknet_resnet34",
    #     #"linknet_resnet50",
    #     "db_resnet50_rotation"
    # ]

    # reco_archs = [
    #     "crnn_vgg16_bn",
    #     "crnn_mobilenet_v3_small", 
    #     #"master",
    #     #"sar_resnet31"
    # ] #

    # for det_arch, reco_arch in cart_prod(det_archs, reco_archs):
    #     print()
    #     print(det_arch, reco_arch)
    #     try:
    #         model = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True, assume_straight_pages=True, detect_orientation=False)#"rotation" not in det_arch
    #     except:
    #         print("scheißdreck..warte mal..")
    #         import ssl
    #         ssl._create_default_https_context = ssl._create_unverified_context
    #         model = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True, assume_straight_pages=True, detect_orientation=False) #"rotation" not in det_arch



    #     # prep_func = lambda path: DocumentFile.from_images(path)
    #     pred_func = lambda img: model(img)
    #     post_func = get_word_predictions_from_doctr

    #     # prep_funcs.append(prep_func)
    #     pred_funcs.append(pred_func)
    #     post_funcs.append(post_func)
    #     model_names.append(f"doctr_{det_arch}_{reco_arch}")

    # %%
    df = get_benchmark_df(data_loader, pred_funcs, post_funcs, model_names)

    df.to_csv("benchmarking.csv", mode='a', index=False, header=False)


