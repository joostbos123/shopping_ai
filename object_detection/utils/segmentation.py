from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import json
import datetime
from skimage import measure
from shapely.geometry import Polygon
import random
import boto3


def create_segmented_image(results, class_color_mapping):
    # Create list of detected classes, where each element is one detected class/mask
    detected_classes = [tensor.item() for tensor in results[0].boxes.cls]

    # Initialize array to store the segmented image (background color is white)
    height_in, width_in = results[0].masks.masks[0].shape
    annotations = np.ones((height_in, width_in, 3))*255

    # Iterate over detected classes/masks
    for idx, detected_class in enumerate(detected_classes):
        # Get color that corresponds to the class
        class_color = class_color_mapping.get(int(detected_class))
        # Check if class is in the class_color_mapping
        if class_color is not None:
            # Give pixels in the mask the color of the corresponding class
            mask = results[0].masks.masks.cpu().numpy().astype(int)[idx]
            annotations[np.where(mask==1)] = np.array(class_color)
        else:
            print('Segment is not an animal nor person')

        # Create image from annotations array
        annotations = np.uint8(annotations)
        segmented_image = Image.fromarray(annotations, 'RGB')

    return segmented_image


def resize_segmented_image(segmented_image, class_color_mapping, width=1280, height=720):
    # Resize image (note that will also change the color of pixels)
    segmented_image = segmented_image.resize((width, height))

    # make sure that pixel values are in the color mapping
    colors = list(class_color_mapping.values())
    colors.append([255, 255, 255])
    for x in range(0, width):
        for y in range(0, height):
            pixel_list = list(segmented_image.getpixel((x,y)))
            if pixel_list not in colors:
                # Create dict where key is color index and value the sum of absolute differnces between the colors
                pixel_difference = {idx: sum(map(abs, (np.array(color) - np.array(pixel_list)))) for idx, color in enumerate(colors)}
                # Get index of the most similar color
                idx_min = min(pixel_difference, key=pixel_difference.get)
                # Change pixel color to the one of the most similar color
                segmented_image.putpixel((x, y), tuple(colors[idx_min]))

    return segmented_image


def create_segmented_images_from_base_model(path_to_images, path_to_segmented_images, class_color_mapping, width, height, model_name, conf=0.5):

    # Create directory when it does not exist
    if not os.path.exists(path_to_segmented_images):
        os.makedirs(path_to_segmented_images)
    else:
        for file in os.scandir(path_to_segmented_images):
            os.remove(file.path)

    model = YOLO(model_name)

    for file in os.listdir(path_to_images):
        print(file)
        image_path = f'{path_to_images}/{file}'
        results = model(image_path, conf=conf)
    
        # Check if the model has found any masks
        if results[0].masks is not None:
            # Create annotated image where each color represents a class 
            segmented_image = create_segmented_image(results=results, class_color_mapping=class_color_mapping)

            segmented_image = resize_segmented_image(segmented_image=segmented_image, class_color_mapping=class_color_mapping, width=width, height=height)
        else:
            # Create white image
            segmented_image = Image.new("RGB", size=(1280, 720), color=(255, 255, 255))

        segmented_image.save(f'{path_to_segmented_images}/{file}')
        print(f'Created segmented image {path_to_segmented_images}/{file}')


def rgb_to_hex(color):
    # Convert list with rgb color to string of hex color
   r, g, b = tuple(color)
   hex_color = '#{:02x}{:02x}{:02x}'.format(r, g, b)

   return hex_color


def create_manifest(s3_bucket, prefix_to_label_dataset_annotated_images, path_to_segmented_images, class_color_mapping_manifest):

   autolabeling_job = 'debleeding-segmentation-autolabel'
   manifest_file = f'{autolabeling_job}.manifest'

   internal_color_map = {"0": {"class-name": "BACKGROUND",
                               "confidence": 0,
                               "hex-color": "#ffffff"}}

   idx = 1
   for class_name, color in class_color_mapping_manifest.items():
      hex_color = rgb_to_hex(color)
      internal_color_map[str(idx)] = {'class-name': class_name,
                                      'confidence': 0,
                                      'hex-color': hex_color}
      idx += 1

   # Initialize manifest file
   open(manifest_file, 'w').close()

   for file in os.listdir(path_to_segmented_images):

      manifest_part = {
         "source-ref": f's3://{s3_bucket}/{s3_bucket}/{file}',
         "autolabel-ref": f's3://{s3_bucket}/{prefix_to_label_dataset_annotated_images}/{file}',
         "autolabel-ref-metadata": {
            "internal-color-map": internal_color_map,
         "type": "groundtruth/semantic-segmentation",
         "human-annotated": "no",
         "creation-date": str(datetime.datetime.now()).replace(' ', 'T'),
         "job-name": f'labeling-job/{autolabeling_job}',
         }
      }

      # Append to manifest file
      with open(manifest_file, 'a') as f:
         json.dump(manifest_part, f)
         f.write(os.linesep)


def create_sub_masks(image, class_color_mapping: dict):
    width, height = image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    # Create a sub-mask (one bit per pixel) and add to the dictionary
    # Note: we add 1 pixel of padding in each direction
    # because the contours module doesn't handle cases
    # where pixels bleed to the edge of the image
    sub_masks = {class_name: Image.new('1', (width+2, height+2)) for class_name in class_color_mapping}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = list(image.getpixel((x, y))[:3])

            # If the pixel is not black...
            if pixel != [255, 255, 255]:
                # Check to see if we've created a sub-mask...
                class_name = [class_name for class_name, color in class_color_mapping.items() if color == pixel][0]

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[class_name].putpixel((x+1, y+1), 1)

    return sub_masks


def create_annotations(sub_masks, polygon_area_threshold=100):
    annotations = {}
    # Iterate over the different classes
    for class_name, sub_mask in sub_masks.items():
        # Find contours (boundary lines) around each sub-mask
        # Note: there could be multiple contours if the object
        # is partially occluded. (E.g. an elephant behind a tree)
        contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation='low')

        segmentations = []
        for contour in contours:
            # Flip from (row, col) representation to (x, y)
            # and subtract the padding pixel
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)

            # Make a polygon and simplify it
            poly = Polygon(contour)
            poly = poly.simplify(1.0, preserve_topology=False)
            if (poly.area > polygon_area_threshold):
                segmentation = list(poly.exterior.coords)
                segmentations.append(segmentation)

        annotations[class_name] = segmentations

    return annotations


def create_label_txt(annotations, file_name='temp/label.txt', image_width=1280, image_height=720):
    first_line = True
    # Make sure that txt file is empty
    open(file_name, 'w').close()
    # Iterate over classes
    for class_id, segmentations in annotations.items():
        # Iterate over segmentations (i.e.) polygons
        for segmentation in segmentations:
            # Create list of x and y coordinates (as strings)
            segmentation_list = []
            for point in segmentation:
                x = round(point[0] / image_width, 4) # Get x coordinate normalized to image size
                y = round(point[1] / image_height, 4) # Get y coordinate normalized to image size
                segmentation_list.append(str(x))
                segmentation_list.append(str(y))
            # Write annotation to file
            with open(file_name, 'a') as f:
                if first_line:
                    f.write(f'{class_id} {" ".join(segmentation_list)}')
                    first_line = False
                else:
                    f.write(f'\n{class_id} {" ".join(segmentation_list)}')


def upload_yolo_segmentation_dataset(bucket: str, s3_prefix_to_label_dataset: str, s3_prefix_yolo_dataset: str, image_annotation_mapping: dict, class_color_mapping: dict, labeling_job_name: str, num_val_videos: int):

    s3_client = boto3.client('s3')

    # Create temp directory when it does not exist
    if not os.path.exists('temp'):
        os.makedirs('temp')

    # Create list containing all videos (numbers) that are assigned randomly to the train set
    image_names = image_annotation_mapping.keys()
    image_names_train = random.sample(image_names, len(image_annotation_mapping.keys()) - num_val_videos)
    print('Image names assigned to the train set are:', image_names_train)

    # Iterate over all videos
    for image_name, annotation_name in image_annotation_mapping.items():

        set_type = 'train' if image_name in image_names_train else 'val'

        # Download image
        s3_client.download_file(bucket, f'{s3_prefix_to_label_dataset}/{image_name}', 'temp/image.png')

        # Download image with annotations
        s3_client.download_file(bucket, f'{s3_prefix_to_label_dataset}/{labeling_job_name}/annotations/consolidated-annotation/output/{annotation_name}', 'temp/annotations.png')
        
        # Convert image (based on colors) to polygons
        img = Image.open('temp/annotations.png')
        img = img.convert('RGB')
        sub_masks = create_sub_masks(img, class_color_mapping)
        annotations = create_annotations(sub_masks)

        # Create txt file with labels (i.e. polygons)
        create_label_txt(annotations, 'temp/label.txt')
                
        # Upload txt file to S3
        s3_client.upload_file('temp/label.txt', bucket, f"{s3_prefix_yolo_dataset}/{set_type}/labels/{labeling_job_name}_{image_name.replace('.png', '')}.txt")
            
        # Upload image to S3
        s3_client.upload_file('temp/image.png', bucket, f"{s3_prefix_yolo_dataset}/{set_type}/images/{labeling_job_name}_{image_name}")
                    
        print(f"Image {image_name} is added to the {set_type} set")
    
    return annotations
