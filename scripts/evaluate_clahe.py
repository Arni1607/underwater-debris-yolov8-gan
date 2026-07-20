# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

model = YOLO(r"C:\Users\arnit\OneDrive\Desktop\coding projects\underwater-debris-yolov8-gan\models\best_baseline.pt")
metrics = model.val(data=r"C:\Users\arnit\OneDrive\Desktop\coding projects\underwater-debris-yolov8-gan\data\data_clahe.yaml")
