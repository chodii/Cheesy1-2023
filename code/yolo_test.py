# -*- coding: utf-8 -*-
"""
Created on Mon May  8 09:39:01 2023

@author: Jan Chodora, MD Rakibul Islam, Johana Melander

@references:
    Redmon, Joseph, Santosh Divvala, Ross Girshick, and Ali Farhadi. “You Only Look Once: Unified, Real-Time Object Detection.” arXiv, May 9, 2016. https://doi.org/10.48550/arXiv.1506.02640.

"""

import os
import torch
from matplotlib import pyplot as plt

# YOLOv5s: +s small, +m medium, +l large, +x extream large, +n new_test; +6 on COCO as additional dataset
# YOLOv5: 'yolov5s', 'yolov5s6', 'yolov5l', 'yolov5l6', 'yolov5m', 'yolov5m6', 'yolov5x', 'yolov5x6', 'yolov5n', 'yolov5n6'
def detect():
    types_of_model = ['yolov5s', 'yolov5s6', 'yolov5n6']
    for model_type in types_of_model:
        model = torch.hub.load('ultralytics/yolov5', model_type)
        detect_objects_in_dataset(model, model_type)

def detect_objects_in_dataset(model, model_type, dataset_path = './data/testdata/'):
    results_dir = 'results/' + model_type + '/'
    images = []
    for filename in os.listdir(dataset_path):
        img = os.path.join(dataset_path, filename)
        images.append(img)
        
    results = model(images)
    fig, ax = plt.subplots(figsize=(16,12))
    ax.imshow(results.render()[0])
    plt.show()
    results.save(save_dir=results_dir)

if __name__ == "__main__":
    detect()
