import os
import shutil
import random

dataset_dir = r"C:\Users\elbek\bozorlar nazorati-vit-demo-alif\bozor-classification-2-8\train"
output_dir = "dataset_split"

val_ratio = 0.2  # 20% for validation

train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")

for d in [train_dir, val_dir]:
    os.makedirs(d, exist_ok=True)

# Loop over each class folder
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    random.shuffle(images)

    split_idx = int(len(images) * (1 - val_ratio))
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    for img in train_images:
        shutil.copy2(os.path.join(class_path, img),
                     os.path.join(train_dir, class_name, img))
    for img in val_images:
        shutil.copy2(os.path.join(class_path, img),
                     os.path.join(val_dir, class_name, img))

print("Dataset split done! Train/Val folders created in", output_dir)
