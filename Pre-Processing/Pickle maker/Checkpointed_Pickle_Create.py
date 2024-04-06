import pickle
import gzip
import os
from tensorflow import keras
import tensorflow as tf
import cv2
import numpy as np
import re
import torch
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input
from openpyxl import load_workbook
import platform

# Function to save checkpoints
def save_checkpoint(checkpoint_data, checkpoint_number, output_folder):
    checkpoint_file = os.path.join(output_folder, f"checkpoint_{checkpoint_number}.pkl.gz")
    with gzip.open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)

def get_features(filename, destination, checkpoint_folder):
    input_string = filename
    pattern = r'\d+'
    match = re.search(pattern, input_string)
    if match:
        first_match = match.group()
        input_folder = os.path.join(os.getcwd(), destination, first_match, input_string)
        try:
            file_paths_frames = [file for file in sorted(os.listdir(input_folder)) if file.endswith(".jpg")]
        except:
            return None

        base_model = EfficientNetB7(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        feature_extractor = Model(inputs=base_model.input, outputs=x)

        features_listofList = []
        checkpoint_counter = 0  # Counter for checkpoints
        checkpoint_data = []  # Data to be saved in the checkpoint
        for indx, frame_file in enumerate(file_paths_frames):
            frame_filename = os.path.join(input_folder, frame_file)
            image = cv2.imread(frame_filename)
            image = cv2.resize(image, (600, 600))
            image = preprocess_input(image)
            image = np.expand_dims(image, axis=0)
            spatial_embedding = feature_extractor.predict(image)[0]
            features_listofList.append(spatial_embedding)
            
            # Checkpointing logic
            if len(features_listofList) % 1000 == 0:
                checkpoint_counter += 1
                checkpoint_data.append({
                    'filename': filename,
                    'features': features_listofList
                })
                save_checkpoint(checkpoint_data, checkpoint_counter, checkpoint_folder)
                checkpoint_data = []  # Reset checkpoint data
                features_listofList = []  # Reset features list
            
        # Save any remaining data as a final checkpoint
        if checkpoint_data:
            checkpoint_counter += 1
            save_checkpoint(checkpoint_data, checkpoint_counter, checkpoint_folder)
            
    else:
        print("No match found.")
        return None

# Function to create the pickle file
def create_pickle(workbook_dest, output_dest, frame_dest, checkpoint_folder):
    workbook = load_workbook(workbook_dest)
    sheet = workbook.active
    excel_data = []
    for row in sheet.iter_rows(values_only=True):
        excel_data.append(row)
        
    list_of_inputs = []
    for tmp in excel_data:
        features = get_features(str(tmp[0]), frame_dest, checkpoint_folder)
        if features is not None:
            if len(features):
                data_dict = {
                    'name': tmp[0],
                    'signer': tmp[1],
                    'gloss': tmp[2],
                    'text': tmp[3],
                    'sign': features + 1e-8
                }
                list_of_inputs.append(data_dict)
                
    with gzip.open(output_dest, 'wb') as f:
        pickle.dump(list_of_inputs, f)

# Paths
vw_dest = "Dataset/excels/Validation.xlsx"
vo_dest = "Dataset/Pickles/excel_data.dev.pkl.gz"
vf_dest = "Dataset/Final folder for frames"
vo_checkpoint_folder = "Dataset/Checkpoints/validation"

create_pickle(vw_dest, vo_dest, vf_dest, vo_checkpoint_folder)

