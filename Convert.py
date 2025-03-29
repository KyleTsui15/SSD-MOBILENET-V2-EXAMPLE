#Setup

#Download and Install labelimg
#Properly annotate all images for the dataset
#Drag and drop the image and xml files into images/all directory

#Download anaconda and ensure the proper folders are listed in environment variables

#Recommended to be run on VS Code with the "Jupyter" extension


#Dependencies (In anaconda prompt, VSCODE Terminal, or Command Prompt)

#conda create -n [ENVNAME] python == 3.9
#conda activate [ENVNAME]

#The following step should already be setup and it should be in the file
#--> Download the tensorflow models github repo [https://github.com/tensorflow/models]


#cd models/research

#sudo apt-get install protobuf-compiler
#protoc object_detection/protos/*.proto --python_out=.          (NOTE: You may encounter an issue saying there is no "setup.py" file. In that case just manually download it from github instead of using "git clone") 
#pip install .                                                  (NOTE: This should download all necesary dependencies including tensorflow itself)

#WORKING AS OF 2025/03/16 
#%%

# import os
# from PIL import Image
# import pillow_heif

# def heic_to_jpg(source_dir, destination_dir):
#     """
#     Converts all .heic files in source_dir to .jpg and saves them to destination_dir.
#     """
#     # Create destination directory if it doesn't exist
#     if not os.path.exists(destination_dir):
#         os.makedirs(destination_dir)
    
#     for filename in os.listdir(source_dir):
#         # Check if the file is a .heic (case-insensitive)
#         if filename.lower().endswith('.heic'):
#             heic_path = os.path.join(source_dir, filename)
            
#             # Open the HEIC image using pillow-heif
#             heif_file = pillow_heif.open_heif(heic_path)
            
#             # Convert HEIF/HEIC file to a Pillow Image
#             img = heif_file.to_pillow()
            
#             # Construct output filename (replace .heic with .jpg)
#             base_name = os.path.splitext(filename)[0]
#             output_filename = base_name + '.jpg'
#             output_path = os.path.join(destination_dir, output_filename)
            
#             # Save as JPEG
#             img.save(output_path, format='JPEG')
#             print(f"Converted: {filename} -> {output_filename}")

# if __name__ == "__main__":
#     source_dir = r"C:\Users\Kyle\Downloads\Recycling_Initiative\RAWHEIC"
#     destination_dir = r"C:\Users\Kyle\Downloads\Recycling_Initiative\Data"
    
#     heic_to_jpg(source_dir, destination_dir)
#     print('exists')
import os

def rename_jpg_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".JPG"):
            old_path = os.path.join(directory, filename)
            new_filename = filename[:-4] + ".jpg"
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

# Example usage:
directory_path = "images/train"
rename_jpg_files(directory_path)

#%%

import os

#Create the train, validation, and test directories
os.makedirs('images/train', exist_ok=True)
os.makedirs('images/validation', exist_ok=True)
os.makedirs('images/test', exist_ok=True)

print("Directories created and files extracted.")
# %%


import urllib.request
import subprocess

# # 1. Download the script using urllib.request
# script_url = "https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/util_scripts/train_val_test_split.py"
# script_filename = "train_val_test_split.py"

# urllib.request.urlretrieve(script_url, script_filename)
# print(f"Downloaded {script_filename}")

# # 2. Run the script using subprocess
# result = subprocess.run(["python", script_filename])
# if result.returncode == 0:
#     print("Script ran successfully.")
# else:
#     print("There was an error running the script.")


#%%
#Defines Classes (Useful for future iterations with multiple classes)


labelmap_path = "labelmap.txt"

# Create (or append to) a file named labelmap.txt, and add the class names
with open(labelmap_path, "a") as f:
    f.write("PencilSharpener\n")
#%%


# Download the create_csv.py and create_tfrecord.py scripts
#create_csv_url = "https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/util_scripts/create_csv.py"
#create_tfrecord_url = "https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/util_scripts/create_tfrecord.py"

#urllib.request.urlretrieve(create_csv_url, 'create_csv.py')
#urllib.request.urlretrieve(create_tfrecord_url, 'create_tfrecord.py')
#print("Scripts downloaded successfully.")
import subprocess

# Run the create_csv.py script
subprocess.run(["python", "create_csv.py"], check=True)

# Run create_tfrecord.py for the training dataset
subprocess.run([
    "python", "create_tfrecord.py",
    "--csv_input", "images/train_labels.csv",
    "--labelmap", "labelmap.txt",
    "--image_dir", "images/train",
    "--output_path", "train.tfrecord"
], check=True)

# Run create_tfrecord.py for the validation dataset
subprocess.run([
    "python", "create_tfrecord.py",
    "--csv_input", "images/validation_labels.csv",
    "--labelmap", "labelmap.txt",
    "--image_dir", "images/validation",
    "--output_path", "val.tfrecord"
], check=True)

print("CSV files created and TFRecord files generated.")


#%%

#region RERUN 
train_record_fname = 'train.tfrecord'
val_record_fname = 'val.tfrecord'
label_map_pbtxt_fname = 'labelmap.pbtxt'


#%%
# Change the chosen_model variable to deploy different models available in the TF2 object detection zoo
chosen_model = 'ssd-mobilenet-v2-fpnlite-320'

MODELS_CONFIG = {
    'ssd-mobilenet-v2': {
        'model_name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',
        'base_pipeline_file': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
    },
    'efficientdet-d0': {
        'model_name': 'efficientdet_d0_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d0_coco17_tpu-32.tar.gz',
    },
    'ssd-mobilenet-v2-fpnlite-320': {
        'model_name': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8',
        'base_pipeline_file': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz',
    },
    # The centernet model isn't working as of 9/10/22
    #'centernet-mobilenet-v2': {
    #    'model_name': 'centernet_mobilenetv2fpn_512x512_coco17_od',
    #    'base_pipeline_file': 'pipeline.config',
    #    'pretrained_checkpoint': 'centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz',
    #}
}

model_name = MODELS_CONFIG[chosen_model]['model_name']
pretrained_checkpoint = MODELS_CONFIG[chosen_model]['pretrained_checkpoint']
base_pipeline_file = MODELS_CONFIG[chosen_model]['base_pipeline_file']

num_steps = 50000
batch_size = 2


#%%
# Set file locations and get number of classes for config file
pipeline_fname = 'models/mymodel/' + base_pipeline_file
fine_tune_checkpoint = 'models/mymodel/' + model_name + '/checkpoint/ckpt-0'
# Set file locations and get number of classes for config file


def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())
num_classes = get_num_classes(label_map_pbtxt_fname)
print('Total classes:', num_classes)


#%%

import os
import re

# Make sure these variables are defined before running:
# pipeline_fname, fine_tune_checkpoint, train_record_fname,
# val_record_fname, label_map_pbtxt_fname, batch_size, num_steps,
# num_classes, chosen_model

# Change working directory (replace path as needed)
# os.chdir("models\mymodel")
print("writing custom configuration file")

# Read the base pipeline file
with open(pipeline_fname, "r") as f:
    s = f.read()

# Perform the various regex replacements
s = re.sub(r'fine_tune_checkpoint: ".*?"',
           f'fine_tune_checkpoint: "{fine_tune_checkpoint}"', s)

s = re.sub(r'(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")',
           f'input_path: "{train_record_fname}"', s)
s = re.sub(r'(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")',
           f'input_path: "{val_record_fname}"', s)

s = re.sub(r'label_map_path: ".*?"',
           f'label_map_path: "{label_map_pbtxt_fname}"', s)

s = re.sub(r'batch_size: [0-9]+',
           f'batch_size: {batch_size}', s)

s = re.sub(r'num_steps: [0-9]+',
           f'num_steps: {num_steps}', s)

s = re.sub(r'num_classes: [0-9]+',
           f'num_classes: {num_classes}', s)

# Convert fine-tune checkpoint type from "classification" to "detection"
s = re.sub(r'fine_tune_checkpoint_type: "classification"',
           'fine_tune_checkpoint_type: "detection"', s)

# If using ssd-mobilenet-v2, reduce the learning rate
if chosen_model == 'ssd-mobilenet-v2':
    s = re.sub(r'learning_rate_base: .8',
               'learning_rate_base: .08', s)
    s = re.sub(r'warmup_learning_rate: 0.13333',
               'warmup_learning_rate: .026666', s)

# If using efficientdet-d0, adjust resizers
if chosen_model == 'efficientdet-d0':
    s = re.sub(r'keep_aspect_ratio_resizer', 'fixed_shape_resizer', s)
    s = re.sub(r'pad_to_max_dimension: true', '', s)
    s = re.sub(r'min_dimension', 'height', s)
    s = re.sub(r'max_dimension', 'width', s)

# Write out the modified pipeline config
with open("pipeline_file.config", "w") as f:
    f.write(s)

print("Done writing pipeline_file.config")


#%%
pipeline_file = 'pipeline_file.config'
model_dir = 'training/'


#%%
import subprocess

# Launch TensorBoard as a background subprocess
tensorboard_process = subprocess.Popen([
    "tensorboard",
    "--logdir", "training/train",
    "--port", "6007"
])

print("TensorBoard is running")
print("http://localhost:6007")



#%%
cmd = [
    "python",
    "models/research/object_detection/model_main_tf2.py",
    "--pipeline_config_path=pipeline_file.config",
    "--model_dir=training/",
    "--alsologtostderr",
    "--num_train_steps=50000",
    "--sample_1_of_n_eval_examples=1"
]

process = subprocess.run(cmd, check=False, capture_output=True, text=True)
print("Return code:", process.returncode)
print("Standard Output:\n", process.stdout)
print("Error Output:\n", process.stderr)

#%%

output_directory = '/home/KyleTsui/Recycling_Initiative/custom_model_lite'

# Path to training directory (the conversion script automatically chooses the highest checkpoint file)
last_model_path = '/home/KyleTsui/Recycling_Initiative/training'

import subprocess

cmd = [
    "python",
    "models/research/object_detection/export_tflite_graph_tf2.py",
    "--trained_checkpoint_dir", last_model_path,
    "--output_directory", output_directory,
    "--pipeline_config_path", pipeline_file
]

process = subprocess.run(cmd, capture_output=True, text=True)
print("Return code:", process.returncode)
print("Stdout:\n", process.stdout)
print("Stderr:\n", process.stderr)



#%%
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model('custom_model_lite/saved_model')
tflite_model = converter.convert()

with open('custom_model_lite/detect.tflite', 'wb') as f:
  f.write(tflite_model)


#%%
import os
import cv2
import numpy as np
import sys
import glob
import random
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter

import matplotlib
import matplotlib.pyplot as plt

### Define function for inferencing with TFLite model and displaying results

def tflite_detect_images(modelpath, imgpath, lblpath, min_conf=0.5, num_test_images=10, savepath='/content/results', txt_only=False):

  # Grab filenames of all images in test folder
  images = glob.glob(imgpath + '/*.jpg') + glob.glob(imgpath + '/*.JPG') + glob.glob(imgpath + '/*.png') + glob.glob(imgpath + '/*.bmp')

  # Load the label map into memory
  with open(lblpath, 'r') as f:
      labels = [line.strip() for line in f.readlines()]

  # Load the Tensorflow Lite model into memory
  interpreter = Interpreter(model_path=modelpath)
  interpreter.allocate_tensors()

  # Get model details
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]

  float_input = (input_details[0]['dtype'] == np.float32)

  input_mean = 127.5
  input_std = 127.5

  # Randomly select test images
  images_to_test = random.sample(images, num_test_images)

  # Loop over every image and perform detection
  for image_path in images_to_test:

      # Load image and resize to expected shape [1xHxWx3]
      image = cv2.imread(image_path)
      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      imH, imW, _ = image.shape
      image_resized = cv2.resize(image_rgb, (width, height))
      input_data = np.expand_dims(image_resized, axis=0)

      # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
      if float_input:
          input_data = (np.float32(input_data) - input_mean) / input_std

      # Perform the actual detection by running the model with the image as input
      interpreter.set_tensor(input_details[0]['index'],input_data)
      interpreter.invoke()

      # Retrieve detection results
      boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
      classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
      scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

      detections = []

      # Loop over all detections and draw detection box if confidence is above minimum threshold
      for i in range(len(scores)):
          if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

              # Get bounding box coordinates and draw box
              # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
              ymin = int(max(1,(boxes[i][0] * imH)))
              xmin = int(max(1,(boxes[i][1] * imW)))
              ymax = int(min(imH,(boxes[i][2] * imH)))
              xmax = int(min(imW,(boxes[i][3] * imW)))

              cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

              # Draw label
              object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
              label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
              labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
              label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
              cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
              cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

              detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])


      # All the results have been drawn on the image, now display the image
      if txt_only == False: # "text_only" controls whether we want to display the image results or just save them in .txt files
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12,16))
        plt.imshow(image)
        plt.show()

      # Save detection results in .txt files (for calculating mAP)
      elif txt_only == True:

        # Get filenames and paths
        image_fn = os.path.basename(image_path)
        base_fn, ext = os.path.splitext(image_fn)
        txt_result_fn = base_fn +'.txt'
        txt_savepath = os.path.join(savepath, txt_result_fn)

        # Write results to text file
        # (Using format defined by https://github.com/Cartucho/mAP, which will make it easy to calculate mAP)
        with open(txt_savepath,'w') as f:
            for detection in detections:
                f.write('%s %.4f %d %d %d %d\n' % (detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))

  return

#%%

# Set up variables for running user's model
PATH_TO_IMAGES='images/test'   # Path to test images folder
PATH_TO_MODEL='custom_model_lite/detect.tflite'   # Path to .tflite model file
PATH_TO_LABELS='labelmap.txt'   # Path to labelmap.txt file
min_conf_threshold=0.5   # Confidence threshold (try changing this to 0.01 if you don't see any detection results)
images_to_test = 2   # Number of images to run detection on

# Run inferencing function!
tflite_detect_images(PATH_TO_MODEL, PATH_TO_IMAGES, PATH_TO_LABELS, min_conf_threshold, images_to_test)

















#region Testing
# %%
import pandas as pd

df = pd.read_csv("images/train_labels.csv")
print(df.head())  # Print first few rows
print(f"Total records: {len(df)}")


# %%
import os

train_annotations = os.listdir("images/train")
print(f"Total annotation files: {len(train_annotations)}")
print(train_annotations[:5])  # Print first few files

# %%
os.getcwd()
# %%
import tensorflow as tf

# Check which physical devices TensorFlow sees
physical_devices = tf.config.list_physical_devices('GPU')
print("GPUs: ", physical_devices)

# If there is at least one GPU listed, TensorFlow is aware of it.
if physical_devices:
    print("TensorFlow can see your GPU!")
else:
    print("No GPU found by TensorFlow.")

# %%
import tensorflow as tf
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Built with GPU support:", tf.test.is_built_with_gpu_support())

# %%
import tensorflow as tf

tfrecord_path = "train.tfrecord"

def parse_tfexample_fn(example_proto):
    # Define your feature structure.
    # The keys and shapes must match how you wrote the TFRecord originally.
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        # ... etc. if you have class/label IDs
    }

    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    # Convert from tf.sparse to tf.dense if needed
    xmin = tf.sparse.to_dense(parsed_example['image/object/bbox/xmin'], default_value=0.0)
    xmax = tf.sparse.to_dense(parsed_example['image/object/bbox/xmax'], default_value=0.0)
    ymin = tf.sparse.to_dense(parsed_example['image/object/bbox/ymin'], default_value=0.0)
    ymax = tf.sparse.to_dense(parsed_example['image/object/bbox/ymax'], default_value=0.0)

    class_text = tf.sparse.to_dense(parsed_example['image/object/class/text'], default_value=b'')

    # Return everything you want to inspect
    return {
        'image/filename': parsed_example['image/filename'],
        'image/encoded': parsed_example['image/encoded'],
        'xmin': xmin,
        'xmax': xmax,
        'ymin': ymin,
        'ymax': ymax,
        'class_text': class_text,
    }

dataset = tf.data.TFRecordDataset([tfrecord_path])
dataset = dataset.map(parse_tfexample_fn)

for idx, record in enumerate(dataset.take(5)):  # look at first 5
    filename = record['image/filename'].numpy().decode('utf-8')
    xmin_vals = record['xmin'].numpy()
    xmax_vals = record['xmax'].numpy()
    ymin_vals = record['ymin'].numpy()
    ymax_vals = record['ymax'].numpy()
    class_texts = record['class_text'].numpy()

    print(f"Record {idx}, filename: {filename}")
    print("  Boxes:")
    for i in range(len(xmin_vals)):
        print(f"    class: {class_texts[i].decode('utf-8')}, "
              f"xmin={xmin_vals[i]:.4f}, xmax={xmax_vals[i]:.4f}, "
              f"ymin={ymin_vals[i]:.4f}, ymax={ymax_vals[i]:.4f}")

    # Optionally load the image to see if shape is correct
    image_data = record['image/encoded'].numpy()
    # e.g. decode
    # import cv2
    # import numpy as np
    # nparr = np.frombuffer(image_data, np.uint8)
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # print("  image shape:", img.shape)

    # You can add checks here:
    #   - Assert that xmin < xmax, ymin < ymax, etc.

    print("")

