import boto3
import os
import json
import random
from PIL import Image

def get_manifest(bucket: str, key: str, local_filename='output.manifest'):
    # Download manifest from S3
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket, key, local_filename)

    # Put manifest in list
    manifest = []
    with open(local_filename, 'r') as f:
        for line in f:
            manifest.append(json.loads(line))
        
    return manifest


def convert_box_coordinates(top, left, height, width, image_size={"height": 720, "width": 1280}):
    # Converts (top,left,height,width) to (normalised) yolo format (x_center,y_center,width,height)            
    dw = 1/float(image_size['width']) #normalisation factor x-axis
    dh = 1. / float(image_size['height']) #normalisation factor y-axis
    x_center = round((left + width/2) *dw, 4)
    y_center = round((top + height/2) * dh, 4)
    w_normalised = round(width * dw, 4)
    h_normalised = round(height * dh, 4)

    return x_center, y_center, w_normalised, h_normalised


# Only for video (bounding box) labeling jobs!
def get_video_annotation_mapping(manifest: list, labeling_job_name: str):

    video_annotation_mapping = {}
    for line in manifest:
        video_annotation_mapping[line["source-ref"].split("/")[-2]] = line[f"{labeling_job_name}-ref"].split("/")[-2]
        
    return video_annotation_mapping


# Only for video (bounding box) labeling jobs!
def create_yolo_dataset_from_video_bounding_box(bucket: str, s3_prefix_to_label_dataset: str, s3_prefix_yolo_dataset: str, yolo_folder: str, video_annotation_mapping: str, labeling_job_name: str, num_val_videos: int, perc_no_label_use=1):

    s3_client = boto3.client('s3')

    # Create temp directory when it does not exist
    if not os.path.exists('temp'):
        os.makedirs('temp')

    for set_type in ['train', 'val']:
        for file_type in ['images', 'labels']:
            # Create directory when it does not exist
            if not os.path.exists(f'{yolo_folder}/{set_type}/{file_type}'):
                os.makedirs(f'{yolo_folder}/{set_type}/{file_type}')
            # # Empty folder if it already exists
            # else:
            #     for file in os.scandir(f'{yolo_folder}/{set_type}/{file_type}'):
            #         os.remove(file.path)

    # Set seed for reproducibility
    random.seed(1)

    # Create list containing all videos (numbers) that are assigned randomly to the train set
    annotation_numbers = range(len(video_annotation_mapping.keys()))
    annotation_numbers_train = random.sample(annotation_numbers, len(video_annotation_mapping.keys()) - num_val_videos)
    print('video numbers assigned to the train set are:', annotation_numbers_train)

    # Iterate over all videos
    for video_name, annotation_name in video_annotation_mapping.items():

        set_type = 'train' if int(annotation_name) in annotation_numbers_train else 'val'

        jsonFile = 'temp/annotations.json'
        s3loc_json = f'{s3_prefix_to_label_dataset}/{labeling_job_name}/annotations/consolidated-annotation/output/{annotation_name}/SeqLabel.json'
        with open(jsonFile, 'wb') as f:
            s3_client.download_fileobj(bucket, s3loc_json, f)
        
        # Read json file containing the annotations
        with open(jsonFile, 'rb') as f:
            seqLabel = json.load(f)
                
        # Get list with name of all frames in video
        objects = s3_client.list_objects_v2(Bucket=bucket, Prefix =f'{s3_prefix_to_label_dataset}/{video_name}')['Contents']
        frame_names = [object['Key'].replace('.jpeg', '').replace(f'{s3_prefix_to_label_dataset}/{video_name}/', '') for object in objects if object['Key'].endswith('.jpeg')]
        
        # Iterate over all frames in video
        for frame_name in frame_names:
            
            # Get labels from json file with annotations
            labels = [label for label in seqLabel['detection-annotations'] if label['frame'] == f'{frame_name}.jpeg']
            
            # Ignore percentage of not labeled frames for training set
            if labels or (random.random() <= perc_no_label_use) or set_type=='val':

                # Download image from S3
                s3_prefix_image = f'{s3_prefix_to_label_dataset}/{video_name}/{frame_name}.jpeg'
                local_path_image = f'{yolo_folder}/{set_type}/images/{labeling_job_name}_{video_name.replace(".mp4", "")}_{frame_name}.jpeg'
                s3_client.download_file(bucket, s3_prefix_image, local_path_image)

                # Get image size
                img = Image.open(local_path_image)
                image_size = {"height": img.height, "width": img.width}

                # Initialize empty txt file to store labels
                local_path_label = f'{yolo_folder}/{set_type}/labels/{labeling_job_name}_{video_name.replace(".mp4", "")}_{frame_name}.txt'
                with open(local_path_label, 'w') as f:

                    # Add annotations to txt file if they exist
                    if labels:
                        label = labels[0]
                        # Add annotations to txt file
                        for box in label["annotations"]:
                            height = box["height"]
                            width = box["width"]
                            top = box["top"]
                            left = box["left"]
                            x_center,y_center,w_normalised,h_normalised = convert_box_coordinates(top, left, height, width, image_size)
                            f.write(f"{box['class-id']} {x_center} {y_center} {w_normalised} {h_normalised} \n")

        print(f"Video {video_name} is added to the {set_type} set")


# Only for image bounding box labeling jobs!
def get_image_video_mapping(manifest):

    image_video_mapping = {}
    for line in manifest:
        image_name = line['source-ref'].split('/')[-1]
        video_name = '_'.join(image_name.split('_')[:-1])
        image_video_mapping[image_name] = video_name

    return image_video_mapping


# Only for image bounding box labeling jobs!
def create_yolo_dataset_from_image_bounding_box(bucket: str, s3_prefix_to_label_dataset: str, yolo_folder: str, labeling_job_name: str, manifest: list, image_video_mapping: dict, perc_val_images: float, image_ext='.jpg'):

    s3_client = boto3.client('s3')

    for set_type in ['train', 'val']:
        for file_type in ['images', 'labels']:
            # Create directory when it does not exist
            if not os.path.exists(f'{yolo_folder}/{set_type}/{file_type}'):
                os.makedirs(f'{yolo_folder}/{set_type}/{file_type}')
            # # Empty folder if it already exists
            # else:
            #     for file in os.scandir(f'{yolo_folder}/{set_type}/{file_type}'):
            #         os.remove(file.path)

    # Set seed for reproducibility
    random.seed(1)

    # Assign videos to train and val set
    videos = set(image_video_mapping.values())
    num_training = round((1-perc_val_images)*len(videos))
    videos_training = random.sample(videos, num_training)

    print('videos assigned to the train set are:', videos_training)

    # Download image and create .txt file with labels
    for image_name, video_name in image_video_mapping.items():

        set_type = 'train' if video_name in videos_training else 'val'

        try:
            # Download image from S3
            s3_prefix_image = f'{s3_prefix_to_label_dataset}/{image_name}'
            local_path_image = f'{yolo_folder}/{set_type}/images/{labeling_job_name}_{image_name}'
            s3_client.download_file(bucket, s3_prefix_image, local_path_image)

            # Get image size
            img = Image.open(local_path_image)
            image_size = {"height": img.height, "width": img.width}

            # Retrieve annotations of labeling job from manifest
            annotations = [line[labeling_job_name]['annotations'] for line in manifest if line["source-ref"].split('/')[-1] == image_name][0]
            if annotations == []:
                print(f'No objects in image {image_name}')

            # Initialize empty txt file to store labels
            local_path_label = f'{yolo_folder}/{set_type}/labels/{labeling_job_name}_{image_name.replace(image_ext, ".txt")}'
            with open(local_path_label, 'w') as f:
                # Add annotations to txt file
                for box in annotations:
                    height = box["height"]
                    width = box["width"]
                    top = box["top"]
                    left = box["left"]
                    x_center,y_center,w_normalised,h_normalised = convert_box_coordinates(top, left, height, width, image_size)
                    f.write(f"{box['class_id']} {x_center} {y_center} {w_normalised} {h_normalised} \n")

            print(f"Image {image_name} is added to the {set_type} set")
        
        except:
            print(f'Not able to create labels for image {image_name}, probably the Labeling jobs has not been fully completed yet')


def map_label_classes(old_to_new_class_mapping: dict, labels_folder_path: dict, temp_file_path: dict):

    # Get all files in labels folder
    files = os.listdir(labels_folder_path)

    for file in files:
        if file.endswith('.txt'):
            # Create path to file
            file_path = f'{labels_folder_path}/{file}'
            # Open the input file in read mode and the temporary file in write mode
            with open(file_path, 'r') as input_file, open(temp_file_path, 'w') as temp_file:
                # Iterate over each line in the input file
                for line in input_file:
                    # Replace the desired value in the line
                    class_id = line.split(' ')[0]
                    if class_id in old_to_new_class_mapping.keys():
                        if old_to_new_class_mapping[class_id] != '':
                            modified_line = old_to_new_class_mapping[class_id] + ' ' + ' '.join(line.split(' ')[1:])
                            # Write the modified line to the temporary file
                            temp_file.write(modified_line)
                    else:
                        modified_line = line
                        # Write the modified line to the temporary file
                        temp_file.write(modified_line)

            # Replace the original file with the temporary file
            os.replace(temp_file_path, file_path)
    
    print(f'Succesfully adjusted the class numbers of all labels in folder {labels_folder_path}')