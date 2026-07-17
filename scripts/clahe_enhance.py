import os
import cv2

input_folder = r"C:\Users\arnit\OneDrive\Desktop\coding projects\underwater-debris-yolov8-gan\data\valid\images"
output_folder = r"C:\Users\arnit\OneDrive\Desktop\coding projects\underwater-debris-yolov8-gan\data\valid_clahe\images"
os.makedirs(output_folder, exist_ok=True)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

for filename in os.listdir(input_folder):
    inp_img = cv2.imread(os.path.join(input_folder, filename))
    lab = cv2.cvtColor(inp_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    merged = cv2.merge((l_clahe, a, b))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    cv2.imwrite(os.path.join(output_folder, filename), result)