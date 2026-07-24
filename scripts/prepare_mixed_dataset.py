import os
import shutil

raw_images = r"C:\Users\arnit\OneDrive\Desktop\coding projects\underwater-debris-yolov8-gan\data\train\images"
raw_labels = r"C:\Users\arnit\OneDrive\Desktop\coding projects\underwater-debris-yolov8-gan\data\train\labels"
gan_images = r"C:\Users\arnit\OneDrive\Desktop\coding projects\underwater-debris-yolov8-gan\data\train_enhanced_resized\images"
gan_labels = r"C:\Users\arnit\OneDrive\Desktop\coding projects\underwater-debris-yolov8-gan\data\train_enhanced_resized\labels"
mixed_images = r"C:\Users\arnit\OneDrive\Desktop\coding projects\underwater-debris-yolov8-gan\data\train_mixed\images"
mixed_labels = r"C:\Users\arnit\OneDrive\Desktop\coding projects\underwater-debris-yolov8-gan\data\train_mixed\labels"

os.makedirs(mixed_images, exist_ok=True)
os.makedirs(mixed_labels, exist_ok=True)

# Copy raw images and labels as-is
for f in os.listdir(raw_images):
    shutil.copy(os.path.join(raw_images, f), os.path.join(mixed_images, f))
for f in os.listdir(raw_labels):
    shutil.copy(os.path.join(raw_labels, f), os.path.join(mixed_labels, f))

# Copy GAN images and labels with 'gan_' prefix to avoid overwriting
for f in os.listdir(gan_images):
    shutil.copy(os.path.join(gan_images, f), os.path.join(mixed_images, 'gan_' + f))
for f in os.listdir(gan_labels):
    shutil.copy(os.path.join(gan_labels, f), os.path.join(mixed_labels, 'gan_' + f))

img_count = len(os.listdir(mixed_images))
lbl_count = len(os.listdir(mixed_labels))
print(f"Mixed dataset ready: {img_count} images, {lbl_count} labels")