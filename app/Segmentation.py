import os
from dataclasses import dataclass

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import cv2

torch.use_deterministic_algorithms(False)

import supervision as sv
from groundingdino.util.inference import Model
from segment_anything import SamPredictor

import numpy as np
from shapely.geometry import Polygon
from autodistill.detection import CaptionOntology

from autodistill_grounded_sam.helpers import (
    combine_detections,
    load_grounding_dino,
    load_SAM,
)

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Segmentation:
    ontology: CaptionOntology
    grounding_dino_model: Model
    sam_predictor: SamPredictor
    box_threshold: float
    text_threshold: float

    def __init__(self, class_mapping: dict):
        self.class_mapping = class_mapping
        self.ontology = CaptionOntology(class_mapping)
        self.grounding_dino_model = load_grounding_dino()
        self.sam_predictor = load_SAM()

    def predict(
        self, image: np.ndarray, box_threshold=0.35, text_threshold=0.25
    ) -> sv.Detections:
        # GroundingDINO predictions
        detections_list = []

        for i, description in enumerate(self.ontology.prompts()):
            # detect objects
            detections = self.grounding_dino_model.predict_with_classes(
                image=image,
                classes=[description],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            detections_list.append(detections)

        detections = combine_detections(
            detections_list, overwrite_class_ids=range(len(detections_list))
        )

        # Only keep highers confidence detection when the overlap is too high
        detections = self.remove_overlapping_detections(detections)

        # SAM Predictions
        xyxy = detections.xyxy

        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box, multimask_output=False
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])

        detections.mask = np.array(result_masks)

        # separate in supervision to combine detections and override class_ids
        return detections

    def remove_overlapping_detections(
        self, detections: sv.Detections, iou_threshold=0.8
    ) -> sv.Detections:
        detection_indeces_to_keep = list()

        # Get matrix of IOU between all detections
        iou_matrix = sv.box_iou_batch(detections.xyxy, detections.xyxy)

        for index, detection in enumerate(detections):
            conf_overlapping_detections = detections[
                iou_matrix[index] > iou_threshold
            ].confidence

            # Check if detection has the highest confidence of all overlapping detections
            if (
                detection[2] >= conf_overlapping_detections.max()
            ):  # detection[2] contains the confidence
                detection_indeces_to_keep.append(index)

        detections = detections[detection_indeces_to_keep]

        return detections

    def get_images_per_class(
        self, image: np.ndarray, detections: sv.Detections
    ) -> dict():
        items = list()

        for detection in detections:
            class_name = self.class_mapping[
                list(self.class_mapping.keys())[detection[3]]
            ]  # detection[3] contains the class_id

            image_detection = image.copy()
            # Create image of the detected object with white background (based on mask)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if detection[1][i][j] == False:  # detection[1] contains the mask
                        image_detection[i][j] = (255, 255, 255)

            # Crop the image to the bounding box of the detected object
            # x1, y1, x2, y2 = detection[0]  # detectin[0] contains the bounding box
            # image_detection = np.ascontiguousarray(image_detection[
            #     int(y1) : int(y2), int(x1) : int(x2)
            # ])

            # Create polygon of the bounding box of the detected object
            polygons = sv.mask_to_polygons(
                detection[1]
            )  # detection[1] contains the mask

            # Only keep the largest polygon
            polygon_areas = [Polygon(polygon).area for polygon in polygons]
            polygon = polygons[np.argmax(polygon_areas)]

            items.append(
                {"class": class_name, "image": image_detection, "polygon": polygon}
            )

        return items
