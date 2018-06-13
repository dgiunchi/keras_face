#!/usr/bin/python3
import os
import sys
import random
import math
import numpy as np
import skimage.io
from datetime import datetime
import urllib.request
from keras_face.library.siamese import SiameseFaceNet

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

if len(sys.argv) > 1:
    print("# Load image from url in command line")
    input_file, headers = urllib.request.urlretrieve(str(sys.argv[1]))

image = skimage.io.imread(os.path.join(IMAGE_DIR, input_file))
results = model.detect([image], verbose=1)

print("Visualize results")
r = results[0]

visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])
output_file = "{}/{}.png".format("/iexec", datetime.now().strftime('%Y%m%d_%H%M%S'))
print("Save results to ", output_file)
plt.savefig(output_file, bbox_inches='tight', pad_inches='-0.1')



def main():
    fnet = SiameseFaceNet()

    model_dir_path = './models'
    image_dir_path = "./data/images"
    fnet.load_model(model_dir_path)

    database = dict()
    database["danielle"] = [fnet.img_to_encoding(image_dir_path + "/danielle.png")]
    database["younes"] = [fnet.img_to_encoding(image_dir_path + "/younes.jpg")]
    database["tian"] = [fnet.img_to_encoding(image_dir_path + "/tian.jpg")]
    database["andrew"] = [fnet.img_to_encoding(image_dir_path + "/andrew.jpg")]
    database["kian"] = [fnet.img_to_encoding(image_dir_path + "/kian.jpg")]
    database["dan"] = [fnet.img_to_encoding(image_dir_path + "/dan.jpg")]
    database["sebastiano"] = [fnet.img_to_encoding(image_dir_path + "/sebastiano.jpg")]
    database["bertrand"] = [fnet.img_to_encoding(image_dir_path + "/bertrand.jpg")]
    database["kevin"] = [fnet.img_to_encoding(image_dir_path + "/kevin.jpg")]
    database["felix"] = [fnet.img_to_encoding(image_dir_path + "/felix.jpg")]
    database["benoit"] = [fnet.img_to_encoding(image_dir_path + "/benoit.jpg")]
    database["arnaud"] = [fnet.img_to_encoding(image_dir_path + "/arnaud.jpg")]

    fnet.verify(image_dir_path + "/camera_0.jpg", "younes", database)
    fnet.verify(image_dir_path + "/camera_2.jpg", "kian", database)
    fnet.who_is_it(image_dir_path + "/camera_0.jpg", database)
    fnet.who_is_it(image_dir_path + "/younes.jpg", database)


if __name__ == '__main__':
    main()