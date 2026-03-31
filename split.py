import os
import shutil
import random

src_dir = "COVID-19_Radiography_Dataset"
dst_dir = "dataset/COVID_19_Radiography_Dataset"

train_ratio = 0.8
random.seed(33)

classes = os.listdir(src_dir)

print("发现类别：", classes)

for cls in classes:
    cls_path = os.path.join(src_dir, cls)

    # 只处理文件夹（跳过 metadata）
    if not os.path.isdir(cls_path):
        continue

    # 👉 关键：进入 images 子目录
    img_dir = os.path.join(cls_path, "images")

    if not os.path.exists(img_dir):
        print(f"⚠️ 跳过 {cls}（没有 images 文件夹）")
        continue

    images = os.listdir(img_dir)
    images = [img for img in images if img.endswith(".png")]

    random.shuffle(images)

    split_point = int(len(images) * train_ratio)

    train_imgs = images[:split_point]
    val_imgs = images[split_point:]

    print(f"{cls}: train {len(train_imgs)}, val {len(val_imgs)}")

    # ===== train =====
    for img in train_imgs:
        src_img = os.path.join(img_dir, img)
        dst_img = os.path.join(dst_dir, "train", cls)
        os.makedirs(dst_img, exist_ok=True)
        shutil.copy(src_img, dst_img)

    # ===== val =====
    for img in val_imgs:
        src_img = os.path.join(img_dir, img)
        dst_img = os.path.join(dst_dir, "val", cls)
        os.makedirs(dst_img, exist_ok=True)
        shutil.copy(src_img, dst_img)

print("✅ 数据划分完成！")