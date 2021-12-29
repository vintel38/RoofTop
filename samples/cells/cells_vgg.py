"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/cells/")

class CellConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "cell"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 3 + 1  # Background + cell

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
#############################################################

class CellEvalConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "cell"
 
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1 # 1
 
    # Number of classes (including background)
    NUM_CLASSES = 3 + 1# Background,
    # typically after labeled, class can be set from Dataset class
    # if you want to test your model, better set it corectly based on your trainning dataset
 
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
 
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    USE_MINI_MASK = False
    # https://github.com/matterport/Mask_RCNN/issues/2474

###########################################################################   
class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NAME = 'cell'
    # https://github.com/matterport/Mask_RCNN/issues/709
    NUM_CLASSES =  3 + 1 
    # https://github.com/matterport/Mask_RCNN/issues/410



############################################################
#  Dataset
############################################################

class CellDataset(utils.Dataset):

    def load_cell(self, dataset_dir, subset):
        """Load a subset of the Cell dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        # https://github.com/matterport/Mask_RCNN/issues/639
        self.add_class("cell", 1, "shsy5y")
        self.add_class("cell", 2, "astro")
        self.add_class("cell", 3, "cort")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "cell",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a cell dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "cell":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        classid = np.zeros([mask.shape[-1]], dtype=np.int32)
                        
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
            if 'classid' in p:
                classid[i] = [self.class_info[idx]['name']==p['classid'] for idx in range(len(self.class_info))].index(True) # + 1 
            else:
                classid[i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.array(classid, dtype=np.int32)
        # https://github.com/matterport/Mask_RCNN/issues/639

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cell":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, epochs=30):
    """Train the model."""
    # Training dataset.
    dataset_train = CellDataset()
    dataset_train.load_cell(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CellDataset()
    dataset_val.load_cell(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs,
                layers='heads')
                
def fix_overlap(msk):
    """
    Args:
        mask: multi-channel mask, each channel is an instance of cell, shape:(520,704,None)
    Returns:
        multi-channel mask with non-overlapping values, shape:(520,704,None)
    """
    msk = np.array(msk)
    msk = np.pad(msk, [[0,0],[0,0],[1,0]])
    ins_len = msk.shape[-1]
    msk = np.argmax(msk,axis=-1)
    msk = tf.keras.utils.to_categorical(msk, num_classes=ins_len)
    msk = msk[...,1:]
    msk = msk[...,np.any(msk, axis=(0,1))]
    return msk
                
                
def test(model, image_path = None, video_path=None, savedfile=None, classname = None):
    assert image_path or video_path
 
     # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # https://github.com/matterport/Mask_RCNN/issues/1435
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        
        # masks = fix_overlap(r['rois'])
        masks = r['rois']
        # https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/279995
                
        # Colorful
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
        visualize.get_display_instances_pic(image, boxes=masks, masks=r['masks'], 
            class_ids = r['class_ids'], class_number=model.config.NUM_CLASSES,ax = ax,
            class_names=classname,scores=r['scores'], show_mask=True, show_bbox=True)
        # Save output
        if savedfile == None:
            file_name = "test_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        else:
            file_name = savedfile
        plt.savefig(file_name)
        #skimage.io.imsave(file_name, testresult)
    elif video_path:
        pass
    print("Saved to ", os.getcwd() , file_name)
    
def evaluate(dataset, config, image_id):
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, config, image_id)
    scaled_image = modellib.mold_image(image, config) # transfo graphique lambda sur l'image : substract mean pixels to main image
    sample = np.expand_dims(scaled_image, 0)
    r = model.detect(sample, verbose=0)[0]
    # https://github.com/matterport/Mask_RCNN/issues/1285
    AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
    AR, positive_ids = utils.compute_recall(r["rois"], gt_bbox, iou=0.5)
    ARs.append(AR)
    F1_scores.append((2* (np.mean(precisions) * np.mean(recalls)))/(np.mean(precisions) + np.mean(recalls)))
    APs.append(AP)
    return APs, ARs, F1_scores
    
############################################################
#  Detection
############################################################

def detect(model, dataset_dir, class_names):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)
    import matplotlib.pyplot as plt
    import csv
    # Save to csv file
    file_path = os.path.join(submit_dir, "submission.csv")
    with open(file_path, "w") as f:
        header = ['id', 'predicted']
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        for image_name in os.listdir(dataset_dir):
            name, _ = image_name.split('.')
            print("Running on {}".format(os.path.join(dataset_dir, image_name)))
            # Load image and run detection
            image = skimage.io.imread(os.path.join(dataset_dir, image_name))
            # https://github.com/matterport/Mask_RCNN/issues/1435
            if image.ndim != 3:
                image = skimage.color.gray2rgb(image)
            # If has an alpha channel, remove it for consistency
            if image.shape[-1] == 4:
                image = image[..., :3]
            # Detect objects
            r = model.detect([image], verbose=1)[0]
            # Encode image to RLE. Returns a string of multiple lines
            source_id = name
            rle = mask_to_rle(source_id, r["masks"], r["scores"])
            row = [name, rle]
            writer.writerow(row)
            
            # Save image with masks
            _, ax = plt.subplots()
            visualize.get_display_instances_pic(
                image, boxes = r['rois'],masks = r['masks'], class_ids = r['class_ids'],
                class_number=model.config.NUM_CLASSES, ax = ax,
                class_names = class_names, scores=r['scores'],
                show_bbox=False, show_mask=False)
            plt.savefig("{}/{}.png".format(submit_dir, name))

    print("Saved to ", submit_dir)
    
    
############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)
    

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect cells.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash' or 'test'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/cell/dataset/",
                        help='Directory of the Cell dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--classnum', required=False,
                        metavar="class number of your detect model",
                        help="Class number of your detector.")
    parser.add_argument('--epochs', required=False,
                        metavar="Number of epochs for the training phase",
                        help="Nb of training phases")
    # https://stackoverflow.com/questions/32761999/how-to-pass-an-entire-list-as-command-line-argument-in-python/32763023
    parser.add_argument('--classnames', nargs="*", type=str, default=['BG'],required=False,
                        metavar="Names of classes to be detected",
                        help="Names of classes")
    # https://stackoverflow.com/questions/23566970/using-argparse-to-create-output-file
    parser.add_argument("-o", "--output", required=False,
                        help="Output folder of your choice")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash" or args.command == "test":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CellConfig()
    elif args.command == "eval":
        config = CellEvalConfig()
        dataset_val = CellDataset()
        dataset_val.load_cell(args.dataset)
        dataset_val.prepare()
        config.NUM_CLASSES = len(dataset_val.class_info)
    else:
        config = InferenceConfig()
        config.NUM_CLASSES = int(args.classnum)+1 # add backgrouond
        
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights
              
    if args.command == "train":
        # Load weights
        print("Loading weights ", weights_path)
        if args.weights.lower() == "coco":
            # Exclude the last layers because they require a matching
            # number of classes if we change the backbone?
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)
        print("Loading weights finished")
        # Train or evaluate
        print("Start Training !")
        train(model, epochs=int(args.epochs))
        
    elif args.command == "test":
        # we test all models trained on the dataset in different stage
        print(os.getcwd())
        filenames = os.listdir(args.weights)
        for filename in filenames:
            if filename.endswith(".h5"):
                print("Load weights from {filename} ".format(filename=filename))
                model.load_weights(os.path.join(args.weights,filename),by_name=True)
                savedfile_name = os.path.splitext(filename)[0] + ".jpg"
                test(model, image_path=args.image,video_path=args.video, savedfile=savedfile_name, classname = args.classnames)
                
    elif args.command == "detect":
        detect(model, args.dataset, args.classnames)
    
    elif args.command == "eval":
      # https://github.com/matterport/Mask_RCNN/issues/2474
        APs = list(); 
        ARs = list();
        F1_scores = list();
        if os.path.isfile(args.weights):
            model.load_weights(args.weights,by_name=True)
            for image_id in dataset_val.image_ids:     
                APs, ARs, F1_scores = evaluate(dataset_val, config, image_id)
            mAP = np.mean(APs)
            mAR = np.mean(ARs)
            print("mAP is {}, mAR is {} and F1_scores are {}".format(mAP, mAR, F1_scores))
        elif os.path.isfile(args.weights):
            weights = os.listdir(args.weights)
            for weight in weights:
                path_weight = os.path.join(args.weights, weight)
                model.load_weights(path_weight,by_name=True)
                for image_id in dataset_val.image_ids:     
                    APs, ARs, F1_scores = evaluate(dataset_val, config, image_id)
                mAP = np.mean(APs)
                mAR = np.mean(ARs)
                print("{} weight : mAP is {}, mAR is {} and F1_scores are {}".format(weight, mAP, mAR, F1_scores))
        
    else:
        print("'{}' is not recognized.Use 'train' or 'test'".format(args.command))
