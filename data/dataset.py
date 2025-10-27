import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import copy
import torch
import scipy

from .randaug import RandAugment


def build_loader(args):
    train_set, train_loader = None, None
    if args.train_root is not None:
        train_set = ImageDataset(istrain=True, root=args.train_root, data_size=args.data_size, return_index=True)
        train_loader = torch.utils.data.DataLoader(train_set, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size)

    val_set, val_loader = None, None
    if args.val_root is not None:
        val_set = ImageDataset(istrain=False, root=args.val_root, data_size=args.data_size, return_index=True)
        val_loader = torch.utils.data.DataLoader(val_set, num_workers=1, shuffle=True, batch_size=args.batch_size)

    return train_loader, val_loader

def get_dataset(args):
    
    if args.train_root is not None:
        train_set = ImageDataset(istrain=True, root=args.train_root, data_size=args.data_size, return_index=True)
        return train_set   
    return None

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 istrain: bool,
                 root: str,
                 data_size: int,
                 return_index: bool = False):
        # notice that:
        # sub_data_size mean sub-image's width and height.
        """ basic information """
        self.root = root
        self.data_size = data_size
        self.return_index = return_index

        """ declare data augmentation """
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

        # 448:600
        # 384:510
        # 768:
        if istrain:
            # transforms.RandomApply([RandAugment(n=2, m=3, img_size=data_size)], p=0.1)
            # RandAugment(n=2, m=3, img_size=sub_data_size)
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.RandomCrop((data_size, data_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        normalize
                ])
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.CenterCrop((data_size, data_size)),
                        transforms.ToTensor(),
                        normalize
                ])

        """ read all data information """
        self.data_infos = self.getDataInfo(root)


    def getDataInfo(self, root):
        data_infos = []
        folders = os.listdir(root)
        folders.sort() # sort by alphabet
        print("[dataset] class number:", len(folders))
        for class_id, folder in enumerate(folders):
            files = os.listdir(root+folder)
            for file in files:
                data_path = root+folder+"/"+file
                data_infos.append({"path":data_path, "label":class_id})
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        try:
            # 1. 获取样本信息（小样本下精准定位，方便排查）
            info = self.data_infos[index]
            image_path = info["path"]
            label = info["label"]
            required_size = self.data_size  # Swin-Tiny默认输入224x224，从配置读取

            # 2. 前置校验（小样本优先过滤无效文件，节省IO）
            if not os.path.exists(image_path):
                raise IOError(f"文件不存在")
            file_size = os.path.getsize(image_path)
            if file_size < 1024:  # 过滤＜1KB的空文件/残缺文件
                raise IOError(f"文件过小（{file_size} bytes），疑似损坏")
            if not image_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                raise IOError(f"非支持的图片格式")

            # 3. 多方式读取图片（最大化挽救有效样本）
            # 方式1：优先用OpenCV读取（适配大部分图片）
            img = cv2.imread(image_path)
            if img is not None:
                img = img[:, :, ::-1]  # BGR→RGB（Swin-Tiny要求RGB输入）
            else:
                # 方式2：OpenCV失败，用PIL重试（挽救编码异常的图片）
                img = Image.open(image_path).convert("RGB")
                img = np.array(img)  # 转为numpy数组，统一后续处理

            # 4. 图片有效性校验（适配Swin-Tiny输入要求）
            if img.ndim != 3:  # 确保是3通道图片（排除灰度图/单通道图）
                raise IOError(f"非3通道图片（细粒度分类需彩色特征）")
            h, w = img.shape[:2]
            if h < required_size // 2 or w < required_size // 2:  # 尺寸过小无法裁剪到目标大小
                raise IOError(f"图片尺寸过小（{h}x{w}），需至少{required_size//2}x{required_size//2}")

            # 5. 格式转换+数据增强（贴合Swin-Tiny训练需求）
            img = Image.fromarray(img)
            img = self.transforms(img)  # 用之前简化后的轻量增强（避免小样本特征混乱）

            # 6. 按格式返回（小样本下保留索引，方便定位效果好/差的样本）
            if self.return_index:
                return index, img, label
            return img, label

        except Exception as e:
            # 小样本下精准日志：方便你手动清理异常图片，避免重复加载
            # print(f"[样本异常] 索引{index} | 标签{label} | 路径：{image_path} | 原因：{str(e)}")
            # 返回None，由collate_fn过滤（不影响批次训练）
            return None
