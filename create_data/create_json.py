import datetime
import json
import os
import re
import fnmatch
from PIL import Image
from itertools import groupby
import numpy as np
from skimage import measure
from pycococreatortools import pycococreatortools
from pycocotools import mask
from tqdm import tqdm

ROOT_DIR = '/home/potato/workplace/cocoapi/PythonAPI'
data_dir = "/home/potato/workplace/dataset/val_data_random_surface"
IMAGE_DIR = os.path.join(data_dir, "rgb_img")
LABEL_DIR = os.path.join(data_dir, "label_img")  # label_img class
SEG_DIR = os.path.join(data_dir, "ints_img")  # seg_img inst
ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")
f = open("/home/potato/workplace/cocoapi/PythonAPI/name_list_val.txt")
name = []
for line in f:
    name.append(line.strip())
f.close()
INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "wuzhenyu",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'banana',
        'supercategory': 'shape',
    },
    {
        'id': 2,
        'name': 'bowl',
        'supercategory': 'shape',
    },
    {
        'id': 3,
        'name': 'box',
        'supercategory': 'shape',
    },
    {
        'id': 4,
        'name': 'can',
        'supercategory': 'shape',
    },
    {
        'id': 5,
        'name': 'cup',
        'supercategory': 'shape',
    },
    {
        'id': 6,
        'name': 'marker',
        'supercategory': 'shape',
    },
    {
        'id': 7,
        'name': 'pear',
        'supercategory': 'shape',
    },
    {
        'id': 8,
        'name': 'sugar',
        'supercategory': 'shape',
    },
]


def filter_for_jpeg(root, files):
    file_types = ['*.png', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle
def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files
def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour
def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    for name_one in tqdm(name):
        image_path = os.path.join(IMAGE_DIR, name_one)
        image = Image.open(image_path)
        image_info = pycococreatortools.create_image_info(
            image_id, name_one, image.size)
        coco_output["images"].append(image_info)
        segImg_path = os.path.join(SEG_DIR, name_one)
        label_path = os.path.join(LABEL_DIR, name_one)
        segImg = Image.open(segImg_path)
        label = Image.open(label_path)
        segImg = np.asarray(segImg)
        label = np.asarray(label)
        obj_list = np.unique(segImg)
        tolerance = 0 #2
        for obj_id in obj_list[1:]:
            y, x = np.where(segImg == obj_id)
            xmin = x.min()
            ymin = y.min()
            xmax = x.max()
            ymax = y.max()
            class_id = label[y[0], x[0]]

            category_info = {'id': class_id.tolist(), 'is_crowd': 0}
            # bounding_box
            w = xmax - xmin
            h = ymax - ymin
            bounding_box = np.asarray([xmin, ymin, w, h])
            # binary_map
            patch = segImg.copy()
            patch[patch != obj_id] = 0
            patch[patch == obj_id] = 1
            area_sum = np.sum(patch)
            if area_sum > 1000:
                binary_mask_encoded = mask.encode(np.asfortranarray(patch.astype(np.uint8)))
                area = mask.area(binary_mask_encoded)
                #rle = binary_mask_to_rle(patch.astype(np.uint8))
                segmentation = binary_mask_to_polygon(patch.astype(np.uint8), tolerance)
                #print(bounding_box)
                annotation_info = {
                    "id": segmentation_id,
                    "image_id": image_id,
                    "category_id": category_info["id"],#category_info["id"]
                    "iscrowd": category_info["is_crowd"],#tegory_info["is_crowd"]
                    "area": area.tolist(), #area.tolist()
                    "bbox": bounding_box.tolist(), # bounding_box
                    "segmentation": segmentation,
                    "width": patch.shape[1],
                    "height": patch.shape[0],
                }
                coco_output["annotations"].append(annotation_info)
                segmentation_id = segmentation_id + 1
        image_id = image_id + 1



            # filter for jpeg images
    with open('{}/instances_val2017.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    """
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)
        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)



            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:

                    print(annotation_filename)
                    # class_id 需要重新确认
                    # class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                                             .convert('1')).astype(np.uint8)

                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1
    with open('{}/instances_shape_train2018.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    """

if __name__ == "__main__":
    main()