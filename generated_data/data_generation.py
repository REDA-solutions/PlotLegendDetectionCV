"""
This script generates additional random data for training models.
Besides the legend text, the label-csv also contains information about the bounding box of the legend.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as tra
from matplotlib import font_manager as fm
import io
import random
from PIL import Image
from random_word import RandomWords
import os
import cv2
import string
from pathlib import Path
from tqdm import tqdm

no_samples_to_generate = 5000
path = "data3"

Path(path + "/images").mkdir(parents=True, exist_ok=True)
Path(path + "/labels").mkdir(parents=True, exist_ok=True)

r = RandomWords()

def random_word_list(length):
    single_letter_threshold = random.randint(0,length-1)
    result = []
    for i in range (0,single_letter_threshold):
        result.append(random.choices(string.ascii_uppercase + string.ascii_lowercase)[0])
    
    for i in range(single_letter_threshold, length):
        result.append(r.get_random_word())
    return result

def list_to_string(lst):
    if lst is None:
        return ''
    else:
        return ' '.join(lst)

def noisy(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def draw_random_graph(number_of_graphs, path, rotation):
    legend_data = pd.DataFrame(columns=['sample_name', 'legend', 'xmin', 'ymin', 'width', 'height'])

    for graphs_created in tqdm(range(number_of_graphs)):
        markers = [".", "o", "v", "^", "s", "p", "*", "d", "X", "+", ">"]
        random.shuffle(markers)

        legend_length = random.randint(1,4)
        legend_text = random_word_list(legend_length)


        data_1 = random.choice([-1, 1, 1, 1]) * np.random.randn(random.randint(20,50),2)*random.randint(-50,50)
        data_2 = np.random.randint(random.randint(-50,0), random.randint(1,100), size=(random.randint(20,50),2))
        data_3 = np.random.rand(random.randint(20,50),2)*random.randint(-100,100)
        x = random.randint(-10,10) * np.linspace(-5.0, 5.0, 50)
        y = random.randint(1,10) + np.linspace(-5.0, 5.0, 50)
        data_4 = np.stack((x,y), axis=1)

        all_data = [data_1, data_2, data_3, data_4]
        random.shuffle(all_data)

        fig = plt.figure(figsize=(10,7.5))
        ax = fig.add_subplot()

        font = fm.FontProperties(family= random.choice(['Comic Sans MS', 'MS Gothic', 'SimSun', 'Impact', 'Brush Script MT', 'Cambria', 'Lucida Console']),
                            weight=random.choice(['bold', 'normal']),
                            style='normal', size=30)

        font_2 = fm.FontProperties(family= random.choice(['Comic Sans MS', 'MS Gothic', 'SimSun', 'Impact', 'Brush Script MT', 'Cambria', 'Lucida Console']),
                    weight=random.choice(['bold', 'normal']),
                    style='normal', size=16)

        plt.suptitle(list_to_string(random_word_list(random.randint(1,2))), fontproperties=font)
        plt.title(list_to_string(random_word_list(random.randint(1,4))), fontproperties=font_2)
        plt.ylabel(r.get_random_word(), fontsize=10)
        plt.xlabel(r.get_random_word(), fontsize=10)
        plt.grid(random.choice([True, False]))

        fig.set_size_inches(8, 3.2)
        plt.rcParams['figure.facecolor'] = random.choice(['white', 'white', 'white', 'grey'])

        x, y, w, h = 0, 0, 0, 0

        for i in range(legend_length):
            ax.scatter(all_data[i][:,0], all_data[i][:,1], marker = markers[i], s=random.randint(80, 200),  label = legend_text[i])
            ax.set_facecolor(random.choice(['white', 'white', 'white', 'grey']))

        plt.tight_layout()

        if (random.choice([True, True, True, True, True, True, True, True, True, True, True, True, False])):
            legend = ax.legend(loc=random.randint(0,9), title= random.choice([r.get_random_word(), False]), fontsize = 7, title_fontsize = 7, markerscale = 0.5)
            plt.draw()
            frame = legend.get_frame()
            frame.set_facecolor(random.choice(['white', 'grey', 'lightgrey', 'lightgrey']))
            frame.set_edgecolor(random.choice(['white', 'grey', 'darkgrey']))
            
            x, y, w, h = legend.get_window_extent().bounds
            _, height = fig.canvas.get_width_height()
            y = height - y - h
            x, y, w, h = int(x), int(y), int(w), int(h)

        # plt.show()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)#, bbox_inches="tight")
        buffer.seek(0)

        white = (255,255,255)
        new_image = Image.new("RGB", (1024,512), white)

        plot_img =  Image.open(buffer)

        # changing the scale of the plot - not used because it makes legend postition calculation quite complicated
        # plot_img = plot_img.resize([random.randint(820,835),random.randint(410,430)])

        if rotation:
            rotated_img = plot_img.rotate(random.randint(-4,4), expand = 1, fillcolor = white)
        else:
            rotated_img = plot_img

        x_paste = random.randint(50,80)
        y_paste = random.randint(20,40)
        new_image.paste(rotated_img, (x_paste, y_paste))

        if x!=0 and y!=0:
            x, y = x+x_paste, y+y_paste
        else:
            x = y = w = h = np.nan

        # print(x,y,w,h)

        new_image = np.asarray(new_image)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        noisy_image = noisy(new_image, random.uniform(0, 0.02))

        if np.isnan(w): # == if no legend
            legend_text = []

        image_name = "graph" + str(graphs_created) + ".png"
        data_to_add = pd.DataFrame({'sample_name': [image_name], 'legend': [legend_text], 'xmin': [x], 'ymin': [y], 'width': [w], 'height': [h]}, index = [0])

        legend_data = pd.concat([legend_data, data_to_add])

        cv2.imwrite(path + "/images/" + image_name, noisy_image)

        plt.close()
        # if not np.isnan(x):
        #     plt.imshow(noisy_image[y:y+h, x:x+w])
        #     plt.show()
        # else:
        #     print("None")

    legend_data.to_csv(path + '/labels/generated_graph_legends.csv')
    # return new_image

draw_random_graph(number_of_graphs=no_samples_to_generate, path=path, rotation=False)