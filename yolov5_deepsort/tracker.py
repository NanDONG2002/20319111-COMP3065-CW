import cv2
import torch
import numpy as np

from deep_sort.utils.parser import get_config  # import configuration program
from deep_sort.deep_sort import DeepSort  # import DeepSort

# Get config from config file
cfg = get_config()
cfg.merge_from_file("./deep_sort/configs/deep_sort.yaml")
CALL_NAME = 0

list_centers = []

# Initialize DeepSort object with configuration parameters
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE,
                    n_init=cfg.DEEPSORT.N_INIT,
                    nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

# Function to draw bounding boxes, accepts image, list of bounding boxes, and line thickness as input
def draw_bboxes(image, bboxes, line_thickness):

    # Compute line thickness
    line_thickness = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) * 0.5) + 1

    list_center = []
    # Process each bounding box
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        color = (0, 255, 0)  # Bounding box color

        # Draw rectangle bounding box
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

        # Calculate position for the text label and draw it
        font_thickness = max(line_thickness - 1, 1)
        t_size = cv2.getTextSize(cls_id, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # Filled rectangle
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, line_thickness / 3,
                    [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)  # Draw text

        t_x, t_y = (x1+x2)//2, (y1+y2)//2

        list_center.append((t_x, t_y))
        # Draw points
    list_centers.append(list_center)

    color = (0, 0, 255)  # Point color, here it is red
    thickness = -1  # Fill the entire circle, -1 means filled
    radius = 5  # Radius of the point
    try:
        for i in list_centers:
            for j in i:
                cv2.circle(image, (j[0], j[1]), radius, color, thickness)
    except:
        pass
    if len(list_centers) == 20:
        list_centers.pop(0)

    return image  # Return the image with drawn bounding boxes

# Function to update tracking information
def update(bboxes, image):
    # 1. Use DeepSort tracker to track objects by passing the center coordinates, dimensions, and confidence of the targets.
    # 2. The tracker processes the input data and returns the updated tracking results.
    # 3. Convert the tracking results into bounding box information ready for drawing, usually including coordinates, category, and tracking ID of the targets.
    # Return a list of bounding box information to draw and display tracking effects on the image.

    bbox_xywh = []  # Store bounding box center coordinates and dimensions
    confs = []  # Store confidence levels
    bboxes2draw = []  # Store bounding boxes to be drawn

    if len(bboxes) > 0:
        # Extract bounding box information
        for x1, y1, x2, y2, lbl, conf in bboxes:
            obj = [
                int((x1 + x2) * 0.5), int((y1 + y2) * 0.5),  # Center coordinates
                x2 - x1, y2 - y1  # Width and height
            ]
            bbox_xywh.append(obj)  # Add to the list
            confs.append(conf)  # Add confidence level

        xywhs = torch.Tensor(bbox_xywh)  # Convert to PyTorch tensor
        confss = torch.Tensor(confs)  # Convert to PyTorch tensor

        outputs = deepsort.update(xywhs, confss, image)  # Update tracking

        for x1, y1, x2, y2, track_id in list(outputs):
            bboxes2draw.append((x1, y1, x2, y2, 'person', track_id))  # Add to the draw list

    return bboxes2draw  # Return the list of bounding boxes to draw

