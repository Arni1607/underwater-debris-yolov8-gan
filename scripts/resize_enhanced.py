# -*- coding: utf-8 -*-

# py libs
import os
from PIL import Image

input_folder = r"C:\Users\arnit\OneDrive\Desktop\coding projects\underwater-debris-yolov8-gan\data\valid_enhanced\images"
output_folder = r"C:\Users\arnit\OneDrive\Desktop\coding projects\underwater-debris-yolov8-gan\data\valid_enhanced_resized\images"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    inp_img = Image.open(os.path.join(input_folder, filename))
    resized = inp_img.resize((640, 640))
    resized.save(os.path.join(output_folder, filename))
    
