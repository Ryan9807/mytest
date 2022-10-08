import os
import json
import glob
import shutil
import tqdm
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

np.random.seed(0)
w, h = (1280, 1024)   # 这是我的图的大小，你看你来改


def labelme2seg(json_files: list, output_path: str):
    for json_file in tqdm.tqdm(json_files, desc="transforming："):
        img_data = np.ones((h, w), dtype=np.uint8) * 27   # 我一共27类，你目标物多少类这里就写多少类，
        with open(json_file, encoding="utf-8") as f:
            json_data = json.load(f)
        labels_data = json_data["shapes"]
        # 将目标物区域像素填充为对应ID号
        for label_data in labels_data:
            # 下面这行，你的label不是数字的话，是汉字或者其它，自己记得稍微改一下，映射成数字，从0开始
            goods_id = int(label_data["label"])
            location = np.asarray(label_data["points"], dtype=np.int32)
            cv2.fillPoly(img_data, [location], color=(goods_id, goods_id, goods_id))

        res_img = Image.fromarray(img_data, mode="P")
        res_img_name = os.path.basename(json_file).replace(".json", ".png")
        res_img.save(os.path.join(output_path, res_img_name))
    return


if __name__ == '__main__':
    labelme_path = r"上面方.jpg和json文件的路径"
    save_path = r"最终保存的路径/my_dataset"

    img_dir_train = os.path.join(save_path, "img_dir", "train")
    img_dir_val = os.path.join(save_path, "img_dir", "val")
    img_dir_test = os.path.join(save_path, "img_dir", "test")

    ann_dit_train = os.path.join(save_path, "ann_dir", "train")
    ann_dir_val = os.path.join(save_path, "ann_dir", "val")
    ann_dir_test = os.path.join(save_path, "ann_dir", "test")

    if not os.path.exists(img_dir_train):
        os.makedirs(img_dir_train)
    if not os.path.exists(img_dir_val):
        os.makedirs(img_dir_val)
    if not os.path.exists(img_dir_test):
        os.makedirs(img_dir_test)

    if not os.path.exists(ann_dit_train):
        os.makedirs(ann_dit_train)
    if not os.path.exists(ann_dir_val):
        os.makedirs(ann_dir_val)
    if not os.path.exists(ann_dir_test):
        os.makedirs(ann_dir_test)

    json_list_path = glob.glob(labelme_path + "/*.json")
    train_path, test_val_path = train_test_split(json_list_path, test_size=0.15)
    test_path, val_path = train_test_split(test_val_path, test_size=0.15)

    # 制作mask：
    labelme2seg(train_path, ann_dit_train)
    labelme2seg(val_path, ann_dir_val)
    labelme2seg(test_path, ann_dir_test)

    # 图复制进对应位置
    for file in tqdm.tqdm(train_path, desc="copy train_img"):
        shutil.copy(file.replace(".json", ".jpg"), img_dir_train)
    for file in tqdm.tqdm(val_path, desc="copy val_img"):
        shutil.copy(file.replace(".json", ".jpg"), img_dir_val)
    for file in tqdm.tqdm(test_path, desc="copy test_img"):
        shutil.copy(file.replace(".json", ".jpg"), img_dir_test)

