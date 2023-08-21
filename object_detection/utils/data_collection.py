import boto3
import os
import cv2
import pandas as pd
import torch
import json
import datetime
import math
from ultralytics import YOLO
import numpy as np


def download_folder_from_s3(bucket, prefix, local):
    """
    params:
    - bucket: s3 bucket with target contents
    - prefix: pattern to match in s3
    - local: local path to folder in which to place files
    """
    s3_client = boto3.client("s3")

    keys = []
    dirs = []
    next_token = ""
    base_kwargs = {
        "Bucket": bucket,
        "Prefix": prefix,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != "":
            kwargs.update({"ContinuationToken": next_token})
        results = s3_client.list_objects_v2(**kwargs)
        contents = results.get("Contents")
        for i in contents:
            k = i.get("Key")
            if k[-1] != "/":
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get("NextContinuationToken")
    for d in dirs:
        dest_pathname = os.path.join(local, d)
        dest_pathname = dest_pathname.replace(prefix, "")

        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
    for k in keys:
        dest_pathname = os.path.join(local, k)
        dest_pathname = dest_pathname.replace(prefix, "")

        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        s3_client.download_file(bucket, k, dest_pathname.replace(prefix, ""))


def create_video_selections(
    videos_folder: str,
    videos_selections_folder: str,
    video_selections_range: dict,
    overwrite=True
):

    # Create directory when it does not exist
    if not os.path.exists(videos_selections_folder):
        os.makedirs(videos_selections_folder)
    elif overwrite:
        for file in os.scandir(videos_selections_folder):
            os.remove(file.path)

    for video_name, time_range in video_selections_range.items():
        n_clips = int(len(time_range) / 2)
        for i_clip in range(n_clips):
            start_seconds = f"{time_range[i_clip*2] % 60 :02d}"
            start_minuts = f"{math.floor(time_range[i_clip*2] / 60) :02d}"
            end_seconds = f"{(time_range[i_clip*2+1] + 1) % 60 :02d}"
            end_minuts = f"{math.floor((time_range[i_clip*2+1] + 1) / 60) :02d}"  # Increase by one to include the second of the selected end time
            end_time = f"{time_range[i_clip*2+1] + 1:02d}"  # Increase by one to include the second of the selected end time
            if i_clip < 1:
                os.system(
                    f"ffmpeg -i {videos_folder}/{video_name}.mp4 -ss 00:{start_minuts}:{start_seconds} -to 00:{end_minuts}:{end_seconds} -c:v libx264 -crf 30 {videos_selections_folder}/{video_name}.mp4 >/dev/null 2>&1"
                )
            else:
                os.system(
                    f"ffmpeg -i {videos_folder}/{video_name}.mp4 -ss 00:{start_minuts}:{start_seconds} -to 00:{end_minuts}:{end_seconds} -c:v libx264 -crf 30 {videos_selections_folder}/{video_name}_{i_clip}.mp4 >/dev/null 2>&1"
                )
            print(
                f"Created new video with only the selected time range for {video_name}.mp4"
            )


def lower_fps_of_videos(
    videos_folder: str, videos_lower_fps_folder: str, fps="2"
):
    # Create directory when it does not exist
    if not os.path.exists(videos_lower_fps_folder):
        os.makedirs(videos_lower_fps_folder)
    else:
        for file in os.scandir(videos_lower_fps_folder):
            os.remove(file.path)

    for video_name in os.listdir(videos_folder):
        if video_name.endswith(".mp4"):  # or .avi, .mpeg, whatever.
            file_path_video = f"{videos_folder}/{video_name}"
            file_path_video_lower_fps = (
                f"{videos_lower_fps_folder}/{video_name}"
            )
            os.system(
                f"ffmpeg -i {file_path_video}  -filter:v fps={fps} {file_path_video_lower_fps} >/dev/null 2>&1"
            )
            print(f"fps of video {video_name} is lowered to {fps}")
        else:
            continue


def get_frames_from_videos(videos_folder: str, images_folder: str, fps="2", overwrite=True):
    # Create directory when it does not exist
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    elif overwrite:
        for file in os.scandir(images_folder):
            os.remove(file.path)

    for video_name in os.listdir(videos_folder):
        if video_name.endswith(".mp4"):
            file_path_video = f"{videos_folder}/{video_name}"
            file_path_image = (
                f'{images_folder}/{video_name.replace(".mp4", "")}_%03d.jpg'
            )
            os.system(
                f"ffmpeg -i {file_path_video}  -vf fps={fps} {file_path_image} >/dev/null 2>&1"
            )
            print(
                f"Frames are extracted from video {video_name} with fps {fps}"
            )
        else:
            continue


def get_inference_dataframe_yolov5(images_folder: str, model_path: str):
    df_results = pd.DataFrame()

    # Load model
    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)

    for image in os.listdir(images_folder):
        image_path = f"{images_folder}/{image}"
        # Inference
        im = cv2.imread(image_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to("cpu")

        im = np.expand_dims(im, 0)

        results = model(im[0])
        df_result = results.pandas().xyxy[0]

        df_result["image"] = image
        df_results = df_results.append(df_result)

    return df_results


def get_inference_dataframe_yolov8(
    images_folder: str, model_path: str, class_mapping: dict
):
    # Load model
    model = YOLO(model_path)

    results = []
    for image in os.listdir(images_folder):
        image_path = f"{images_folder}/{image}"

        # Inference
        pred = model.predict(image_path, iou=0.5)

        for pred_object in pred:
            boxes = pred_object.boxes
            for box in boxes:
                xmin, ymin, xmax, ymax = [
                    tensor.item() for tensor in box.xyxy[0]
                ]
                confidence = float(box.conf)
                class_number = int(box.cls)
                class_name = class_mapping[str(class_number)]
                results.append(
                    {
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "confidence": confidence,
                        "class": class_number,
                        "name": class_name,
                        "image": image,
                    }
                )

    df_results = pd.DataFrame.from_records(results)
    return df_results


def plot_inference_results(
    images_folder: str,
    images_labeled_folder: str,
    df_results: pd.DataFrame,
    colors: dict,
):
    # Create directory when it does not exist
    if not os.path.exists(images_labeled_folder):
        os.makedirs(images_labeled_folder)
    else:
        for file in os.scandir(images_labeled_folder):
            os.remove(file.path)

    for image_name in os.listdir(images_folder):
        df_result = df_results[df_results["image"] == image_name]
        image_path = f"{images_folder}/{image_name}"

        # Plot inference results
        image = cv2.imread(image_path)
        for _, row in df_result.iterrows():
            c1_rect = int(row["xmin"]), int(row["ymin"])
            c2_rect = int(row["xmax"]), int(row["ymax"])
            x_text = int((row["xmin"] + row["xmax"]) / 2)
            y_text = int((row["ymin"] + row["ymax"]) / 2)

            cv2.rectangle(
                image, c1_rect, c2_rect, colors[row["class"]], thickness=1
            )
            cv2.putText(
                image,
                row["name"],
                (x_text, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"{round(row['confidence'], 2)}",
                (x_text, y_text + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imwrite(f"{images_labeled_folder}/{image_name}", image)


def create_bounding_box_manifest(
    s3_bucket,
    prefix_to_label_dataset,
    images_folder,
    df_results,
    autolabeling_job,
    class_mapping,
):
    # Create directory when it does not exist
    if not os.path.exists("temp"):
        os.makedirs("temp")

    manifest_file = f"{autolabeling_job}.manifest"

    # Initialize manifest file
    open(f"temp/{manifest_file}", "w").close()

    images = os.listdir(images_folder)  # df_results['image'].unique()

    for index, image_name in enumerate(images):
        df_result = df_results[df_results["image"] == image_name]

        annotations = []
        for _, row in df_result.iterrows():
            left = int(row["xmin"])
            top = int(row["ymin"])
            width = int(row["xmax"] - row["xmin"])
            height = int(row["ymax"] - row["ymin"])
            if str(row["class"]) in class_mapping.keys():
                annotation = {
                    "class_id": row["class"],
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                }
                annotations.append(annotation)

        manifest_part = {
            "source-ref": f"s3://{s3_bucket}/{prefix_to_label_dataset}/{image_name}",
            autolabeling_job: {
                "image_size": [{"width": 1280, "height": 720, "depth": 3}],
                "annotations": annotations,
            },
            f"{autolabeling_job}-metadata": {
                "objects": [{"confidence": 0} for _ in annotations],
                "class-map": class_mapping,
                "type": "groundtruth/object-detection",
                "human-annotated": "no",
                "creation-date": str(datetime.datetime.now()).replace(
                    " ", "T"
                ),
                "job-name": f"labeling-job/{autolabeling_job}",
            },
        }

        # Append to manifest file
        with open(f"temp/{manifest_file}", "a") as f:
            json.dump(manifest_part, f)
            if index < len(images) - 1:
                f.write(os.linesep)


def create_bounding_box_manifest_from_yolo_dataset(
    s3_bucket,
    s3_prefix_to_label_dataset,
    dataset_prefix_yolo_old,
    images_to_label,
    autolabeling_job,
    class_mapping,
    ignore_classes=[],
    image_ext=".png",
):
    # Create manifest file for relabeling images Vion Tilburg offloading
    manifest_file = f"{autolabeling_job}.manifest"

    # Initialize manifest file
    open(f"temp/{manifest_file}", "w").close()

    for set_type in ["train", "val"]:
        images = os.listdir(f"{dataset_prefix_yolo_old}/{set_type}/images")
        images = [image.replace(image_ext, "") for image in images]

        for index, image_name in enumerate(images):
            if image_name in images_to_label:
                image = cv2.imread(
                    f"{dataset_prefix_yolo_old}/{set_type}/images/{image_name}{image_ext}"
                )

                # Get height and width of image
                image_height = image.shape[0]
                image_width = image.shape[1]

                try:
                    label_file_path = f"{dataset_prefix_yolo_old}/{set_type}/labels/{image_name}.txt"
                    with open(label_file_path) as f:
                        lines = f.readlines()

                    lines = [line.replace(" \n", "") for line in lines]
                    labels = [line.split(" ") for line in lines]

                    annotations = []
                    for label in labels:
                        class_id = int(label[0])
                        x_norm = float(label[1])
                        y_norm = float(label[2])
                        width_norm = float(label[3])
                        height_norm = float(label[4])

                        left = int((x_norm - width_norm / 2) * image_width)
                        top = int((y_norm - height_norm / 2) * image_height)
                        width = int(width_norm * image_width)
                        height = int(height_norm * image_height)

                        if (
                            class_id
                            in [int(id) for id in class_mapping.keys()]
                        ) and (class_id not in ignore_classes):
                            annotation = {
                                "class_id": class_id,
                                "left": left,
                                "top": top,
                                "width": width,
                                "height": height,
                            }
                            annotations.append(annotation)
                except:
                    annotations = []
                    print(f"labels for image {image_name} could not be found")

                manifest_part = {
                    "source-ref": f"s3://{s3_bucket}/{s3_prefix_to_label_dataset}/{image_name}{image_ext}",
                    autolabeling_job: {
                        "image_size": [
                            {
                                "width": image_width,
                                "height": image_height,
                                "depth": 3,
                            }
                        ],
                        "annotations": annotations,
                    },
                    f"{autolabeling_job}-metadata": {
                        "objects": [{"confidence": 0} for _ in annotations],
                        "class-map": class_mapping,
                        "type": "groundtruth/object-detection",
                        "human-annotated": "no",
                        "creation-date": str(datetime.datetime.now()).replace(
                            " ", "T"
                        ),
                        "job-name": f"labeling-job/{autolabeling_job}",
                    },
                }

                # Append to manifest file
                with open(f"temp/{manifest_file}", "a") as f:
                    json.dump(manifest_part, f)
                    if index < len(images) - 1:
                        f.write(os.linesep)


def upload_folder_to_s3(bucket: str, prefix: str, local_folder: str):
    s3_client = boto3.client("s3")

    # Enumerate local files recursively
    for root, dirs, files in os.walk(local_folder):
        for file_name in files:
            local_path = f"{root}/{file_name}"
            s3_prefix = f'{prefix}{root.replace(local_folder, "")}/{file_name}'
            s3_client.upload_file(local_path, bucket, s3_prefix)

    print(
        f"Uploaded all local files from {local_folder} to S3 location {bucket}/{prefix}"
    )
