# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
model= YOLO('yolov8n.pt')

model.train(data= 'C:/Users/arnit/OneDrive/Desktop/coding projects/underwater-debris-yolov8-gan/data/data.yaml', epochs=50)
