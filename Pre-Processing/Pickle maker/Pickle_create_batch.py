import pickle
import gzip
import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import mixed_precision
# import tensorflow_addons as tfa
# import keras_tuner
import cv2
import numpy as np
import re
import os
import torch
from tqdm.notebook import tqdm
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.efficientnet import EfficientNetB7, EfficientNetB0 
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import mixed_precision
from openpyxl import Workbook, load_workbook
import platform
from progress.bar import Bar
import tqdm
import multiprocessing
import concurrent.futures

# tf.config.run_functions_eagerly(True)


## quietly deep-reload tqdm
# import sys
# from IPython.lib import deepreload 
# stdout = sys.stdout
# sys.stdout = open('junk','w')
# deepreload.reload(tqdm)
# sys.stdout = stdout

# tqdm.__version__

## Training using FP16
# mixed_precision.global_policy()
# mixed_precision.set_global_policy('mixed_float16')
# mixed_precision.global_policy()


def cls():
    system = platform.system()
    if system == 'Windows':
        os.system('cls')
    elif system == 'Linux':
        os.system('clear')
    elif system == 'MacOS':
        os.system('clear')


# Files to access:
# validation
vw_dest = "Dataset/excels/ISH_News_Sports final VAL.xlsx"
vo_dest = "Dataset/Pickles/excel_data.dev"
vf_dest = "Dataset/Final folder for frames"

# test
tw_dest = "Dataset/excels/ISH_News_Sports final TEST.xlsx"
to_dest = "Dataset/Pickles/excel_data.test"
tf_dest = "Dataset/Final folder for frames"

# train
w_dest = "Dataset/excels/ISH_News_Sports final TRAIN.xlsx"
o_dest = "Dataset/Pickles/excel_data.train"
f_dest = "Dataset/Final folder for frames"

# Batch Size
batch_size_user = 8192



# ***Initialise the CNN model***
base_model = EfficientNetB7(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
feature_extractor = Model(inputs=base_model.input, outputs=x)



# ***Function for extraction of features***
def get_features_batch(filenames, destination):
    features_listofList = []
    for filename in filenames:
        input_string = filename
        pattern = r'\d+'
        match = re.search(pattern, input_string)
        if match:
            first_match = match.group()
            input_folder = os.path.join(os.getcwd(), destination, first_match, input_string)
            file_paths_frames = [os.path.join(input_folder, file) for file in sorted(os.listdir(input_folder)) if file.endswith(".jpg")]

            batch_images = []
            for frame_file in file_paths_frames:
                image = cv2.imread(frame_file)
                x1, y1 = 535, 0
                x2, y2 = 1385, 1080
                image = image[y1:y2, x1:x2]
                image = cv2.resize(image, (224, 224))
                image = preprocess_input(image)
                image = np.expand_dims(image, axis=0)
                batch_images.append(image)

            batch_images = np.concatenate(batch_images, axis=0)
            spatial_embeddings_batch = feature_extractor.predict(batch_images)

            for spatial_embedding in spatial_embeddings_batch:
                features_listofList.append(spatial_embedding)
        else:
            print("No match found.")
            return None

    return torch.tensor(np.array(features_listofList))




# ***Function to create the pickle file***
def create_pickle_batch(workbook_dest, output_dest, frame_dest):
    # load the excel file
    workbook = load_workbook(workbook_dest)
    sheet = workbook.active

    excel_data = []
    for row in sheet.iter_rows(values_only=True):
        excel_data.append(row)

    # Get the features in batches
    list_of_inputs = []
    batch_size = batch_size_user  # Adjust as needed

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(0, len(excel_data), batch_size):
            batch_filenames = [str(tmp[0]) for tmp in excel_data[i:i+batch_size]]
            features = get_features_batch(batch_filenames, frame_dest)

            for tmp, feature in zip(excel_data[i:i+batch_size], features):
                if feature is not None:
                    data_dict = {
                        'name': tmp[0],
                        'signer': tmp[1],
                        'gloss': tmp[2],
                        'text': tmp[3],
                        'sign': feature + 1e-8
                    }
                    list_of_inputs.append(data_dict)

    with gzip.open(os.path.join(os.getcwd(), output_dest), 'wb') as f:
        pickle.dump(list_of_inputs, f)



# Validation pickle
create_pickle_batch(vw_dest, vo_dest, vf_dest)


# Testing pickle
create_pickle_batch(tw_dest, to_dest, tf_dest)


# Training pickle
create_pickle_batch(w_dest, o_dest, f_dest)


print("Done")