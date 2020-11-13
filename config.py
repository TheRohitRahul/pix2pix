import os
import torch
import shutil

FORCE_CPU = True

'''
Put in the name of the model 
This is just used to create folder of this name
'''
MODEL_NAME = ""
MODEL_SAVE_FOLDER = "./models"
MODEL_SUMMARY_PATH = "./summary"
IMAGE_SAVE_PATH = "./generated_images"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_FOLDER, MODEL_NAME)


TRAIN_RESUME_PATH = ""
TEST_MODEL_PATH = TRAIN_RESUME_PATH

'''
specify the path to the folder containing A images and B images 
This supports multiple datasets and you can control the ratio of images belonging to each dataset in the batch
key here is just to associate a unique id to each dataset. this can be named anything
Ratio is the amount of images that will be present in the batch of a specific dataset
'''
DATASET_DICT = {
    "key1"  :{
        "A_folder" : "",
        "B_folder" : "",
        "ratio" : 0.5
    },
    "key2" : {
        "A_folder" : "",
        "B_folder" : "",
        "ratio" : 0.5
    }
}

BATCH_SIZE = 8
PRINT_EVERY = 20

INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1

NUM_ITERATIONS = 2000

TEST_MODEL_PATH = ""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
if FORCE_CPU:
    DEVICE = "cpu"

if not(os.path.exists(MODEL_SAVE_PATH)):
    os.makedirs(MODEL_SAVE_PATH)

if (os.path.exists(MODEL_SUMMARY_PATH)):
    shutil.rmtree(MODEL_SUMMARY_PATH)
os.makedirs(MODEL_SUMMARY_PATH)

if (os.path.exists(IMAGE_SAVE_PATH)):
    shutil.rmtree(IMAGE_SAVE_PATH)
os.makedirs(IMAGE_SAVE_PATH)

