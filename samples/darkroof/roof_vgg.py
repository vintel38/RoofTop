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
from tqdm import tqdm

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


class RoofConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "roof"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + roof

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
#############################################################

class RoofEvalConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "roof"
 
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1 # 1
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1# Background,
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
    NAME = 'roof'
    # https://github.com/matterport/Mask_RCNN/issues/709
    NUM_CLASSES =  1 +1 
    # https://github.com/matterport/Mask_RCNN/issues/410



############################################################
#  Dataset
############################################################

class RoofDataset(utils.Dataset):

    def load_roof(self, dataset_dir, subset):
        """Load a subset of the Roof dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("roof", 1, "roof")

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
                "roof",
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
        # If not a roof dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "roof":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "roof":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, epochs=30):
    """Train the model."""
    # Training dataset.
    dataset_train = RoofDataset()
    dataset_train.load_roof(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = RoofDataset()
    dataset_val.load_roof(args.dataset, "val")
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
                
                
def test(model, image_path = None, video_path=None, savedfile=None, classname = None):
    assert image_path or video_path
 
     # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        name, _ = args.image.split('.')
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Colorful
        import matplotlib.pyplot as plt
        
        _, ax = plt.subplots()
        visualize.get_display_instances_pic(image, boxes=r['rois'], masks=r['masks'], 
            class_ids = r['class_ids'], class_number=model.config.NUM_CLASSES,ax = ax,
            class_names=classname,scores=None, show_mask=True, show_bbox=True)
        # Save output
        plt.savefig(os.path.join(args.logs, name + '_d.png'))
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
    F1_score=(2* (np.mean(precisions) * np.mean(recalls)))/(np.mean(precisions) + np.mean(recalls))
    return AP, AR, F1_score

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect roofs.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash' or 'test'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/roof/dataset/",
                        help='Directory of the Roof dataset')
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
    parser.add_argument('--output', required=False,
                        metavar="output folder to store output file of the program",
                        help='Output folder of the Mask RCNN program')
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
        config = RoofConfig()
    elif args.command == "eval":
        config = RoofEvalConfig()
        dataset_val = RoofDataset()
        dataset_val.load_roof(args.dataset, 'test')
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
        train(model, epochs=args.epochs)
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
    
    elif args.command == "eval":
      # https://github.com/matterport/Mask_RCNN/issues/2474
        APs = list(); 
        ARs = list();
        F1_scores = list();
        if os.path.isfile(args.weights):
            # tester un seul fichier de poids
            weights = [args.weights]
        elif os.path.isdir(args.weights):
            # tester plusieurs fichiers de poids contenus dans un dossier 
            weights = os.listdir(args.weights)        
        for weight in weights:
            if len(weights)>1:
                path_weight = os.path.join(args.weights, weight)
            else:
                path_weight = weight
            model.load_weights(path_weight,by_name=True)
            if args.image:
                name, ext = args.image.split('.')
                # inférer le modèle sur une seule image 
                lst = os.listdir(os.path.join(args.dataset, 'test'))
                # evaluate function and also print the detected photo
                image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, config, lst.index(args.image))
                scaled_image = modellib.mold_image(image, config) # transfo graphique lambda sur l'image : substract mean pixels to main image
                sample = np.expand_dims(scaled_image, 0)
                r = model.detect(sample, verbose=0)[0]
                # https://github.com/matterport/Mask_RCNN/issues/1285
                AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
                AR, positive_ids = utils.compute_recall(r["rois"], gt_bbox, iou=0.5)
                ARs=AR
                F1_scores=(2* (np.mean(precisions) * np.mean(recalls)))/(np.mean(precisions) + np.mean(recalls))
                APs=AP
                # APs, ARs, F1_scores = evaluate(dataset_val, config, lst.index(args.image))
                # Run model detection and generate the color splash effect
                print("Running on {}".format(args.image))
                # Read image
                image = skimage.io.imread(os.path.join(args.dataset, 'test', args.image))
                # Colorful
                import matplotlib.pyplot as plt
                _, ax = plt.subplots()
                visualize.get_display_instances_pic(image, boxes=r['rois'], masks=r['masks'], 
                    class_ids = r['class_ids'], class_number=model.config.NUM_CLASSES,ax = ax,
                    class_names=args.classnames,scores=r["scores"], show_mask=True, show_bbox=True)
                # Save output
                plt.savefig(os.path.join(args.logs, name + '_d.png'))
                np.save(os.path.join(args.logs, name + 'mask.npy'), r)
            else:
                # inférer le modèle sur tout un dataset chargé
                for image_id in tqdm(dataset_val.image_ids, desc='dataset_val.image_ids'):
                    AP, AR, F1_score = evaluate(dataset_val, config, image_id)
                    APs.append(AP)
                    ARs.append(AR)
                    F1_scores.append(F1_score)
            mAP = np.mean(APs)
            mAR = np.mean(ARs)
            print("{} weight : mAP is {}, mAR is {} and F1_scores are {}".format(weight, mAP, mAR, F1_scores))
                
    else:
        print("'{}' is not recognized.Use 'train' or 'test'".format(args.command))
