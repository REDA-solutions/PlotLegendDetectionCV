import pandas as pd
import numpy as np
from tqdm import tqdm

data = pd.read_csv(r'generated_data\data3\labels\generated_graph_legends.csv')
path = r'generated_data\data3\labels'

for _, row in tqdm(data.iterrows()):
    x_center_n = (row['xmin']+row['width']/2) / 1024
    y_center_n = (row['ymin']+row['height']/2) / 512
    width_n = row['width'] / 1024
    height_n = row['height'] / 512
    if np.isnan(row['xmin']):
        s = ""
    else:
        s = f"0 {x_center_n} {y_center_n} {width_n} {height_n}"
    with open(path + "\\" + row['sample_name'].replace("png", "txt"), 'w') as f:
        f.write(s)