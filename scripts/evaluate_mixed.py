import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

model = YOLO(r"C:\Users\arnit\OneDrive\Desktop\coding projects\underwater-debris-yolov8-gan\models\best_mixed_train.pt")
metrics = model.val(data=r"C:\Users\arnit\OneDrive\Desktop\coding projects\underwater-debris-yolov8-gan\data\data_enhanced.yaml")