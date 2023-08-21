import os
import cv2
import boto3
import pandas as pd


def get_image_labels_local(yolo_folder, dataset_type):

    image_labels = {}
    label_files_path = f'{yolo_folder}/{dataset_type}/labels'

    for labels_file in os.listdir(label_files_path):
        # Open .txt file containing the labels
        with open(f'{label_files_path}/{labels_file}') as f:
            lines = f.readlines()

        if lines == []:
            labels = [{'label': '', 'box': []}]
        else:
            labels = []
            for line in lines:
                line = line.replace(' \n', '')
                label = {'label': line.split(' ')[0], 'box': line.split(' ')[1:]}
                labels.append(label)
        
        # Add labels to dict
        image_name = labels_file.replace('.txt', '')
        image_labels[image_name] = labels
    
    return image_labels


def plot_yolo_labels(yolo_folder: str, yolo_plotting_folder: str):

    # Create directory when it does not exist
    if not os.path.exists(yolo_plotting_folder):
        os.makedirs(yolo_plotting_folder)
    # Empty folder if it already exists
    else:
        for file in os.scandir(yolo_plotting_folder):
            os.remove(file.path)

    for set_type in ['train', 'val']:
        for image_name in os.listdir(f'{yolo_folder}/{set_type}/images'):

            image_file_path = f'{yolo_folder}/{set_type}/images/{image_name}'
            image = cv2.imread(image_file_path)

            # Get height and width of image
            height = image.shape[0]
            width = image.shape[1]

            image_ext = image_name.split('.')[-1]
            label_file_path = f'{yolo_folder}/{set_type}/labels/{image_name.replace(f".{image_ext}", ".txt")}'
            with open(label_file_path) as f:
                lines = f.readlines()

            lines = [line.replace(' \n', '') for line in lines]
            labels = [line.split(' ') for line in lines]

            for label in labels:
                c1_rect = int(width*(float(label[1]) - float(label[3])/2)), int(height*(float(label[2]) - float(label[4])/2))
                c2_rect = int(width*(float(label[1]) + float(label[3])/2)), int(height*(float(label[2]) + float(label[4])/2))

                x_text = int(width*float(label[1]))
                y_text = int(height*float(label[2]))

                # Plot inference results
                cv2.rectangle(image, c1_rect, c2_rect, (255, 0, 0), thickness=1)
                cv2.putText(image, label[0], (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,  cv2.LINE_AA)

            cv2.imwrite(f'{yolo_plotting_folder}/{image_name}', image)


# def get_image_labels_remote(s3_bucket, s3_prefix, dataset_type, label_ext='.txt'):

#     s3 = boto3.resource('s3')
#     bucket = s3.Bucket(s3_bucket)

#     image_labels = {}

#     for object in bucket.objects.filter(Prefix=f'{s3_prefix}/{dataset_type}/labels'):
#         if object.key.endswith(label_ext):
#             annotation_path = object.key.replace(label_ext, '')
#             image_name = annotation_path.split('/')[-1]
#             body = object.get()['Body'].read()
#             annotations = body.decode("utf-8")
#             labels = []
#             for annotation in annotations.split(' \n')[:-1]:
#                 label = {'label': annotation.split(' ')[0], 'box': annotation.split(' ')[1:]}
#                 labels.append(label)
#             image_labels[image_name] = labels
    
#     return image_labels


def print_dataset_statistics(image_labels_set: dict, set_type: str, class_mapping: dict):

    # Print some statistics of the train set
    print(f'{set_type.capitalize()} set consists of {len(set(image_labels_set.keys()))} images')

    index_mapping = class_mapping.copy()
    index_mapping[''] = 'Geen annotations'

    image_class_labels = []
    for image_labels in image_labels_set.values():
        if image_labels:
            for value in image_labels:
                image_class_labels.append(value['label'])

    image_class_labels_dist = pd.Series(image_class_labels).value_counts().rename(index=index_mapping)
    print('The labels are distributed as follows:\n', image_class_labels_dist)