import os
import io
import argparse
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--csv_input', required=True, help='Path to the CSV input')
parser.add_argument('--labelmap', required=True, help='Path to the labelmap file')
parser.add_argument('--image_dir', required=True, help='Path to the image directory')
parser.add_argument('--output_path', required=True, help='Path to output TFRecord')
args = parser.parse_args()

def split(df, group):
    """Splits CSV data into groups based on filename."""
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path, label_map_dict):
    """Creates a TensorFlow Example from image annotation data."""
    img_path = os.path.join(path, group.filename)
    
    # Read image file
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'

    xmins, xmaxs, ymins, ymaxs, classes_text, classes = [], [], [], [], [], []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(label_map_dict[row['class']])  # Use dictionary lookup

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def load_label_map(labelmap_path):
    """Loads label map from labelmap.txt file and creates labelmap.pbtxt."""
    with open(labelmap_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    
    # Create a dictionary for label lookup
    label_map_dict = {label: idx + 1 for idx, label in enumerate(labels)}

    # Generate labelmap.pbtxt
    pbtxt_path = os.path.join(os.path.dirname(labelmap_path), "labelmap.pbtxt")
    with open(pbtxt_path, 'w') as f:
        for i, label in enumerate(labels):
            f.write(f"item {{\n  id: {i + 1}\n  name: '{label}'\n}}\n\n")

    print(f"Successfully created labelmap.pbtxt at: {pbtxt_path}")
    return label_map_dict

def main():
    label_map_dict = load_label_map(args.labelmap)
    
    # Create TFRecord file
    writer = tf.io.TFRecordWriter(args.output_path)
    examples = pd.read_csv(args.csv_input)

    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, args.image_dir, label_map_dict)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print(f'Successfully created the TFRecord file: {args.output_path}')

if __name__ == '__main__':
    main()
