import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import copy
import scipy
from torch.utils.data import random_split

from .randaug import RandAugment


def build_loader(args):
    val_ratio = getattr(args, 'val_ratio', 0.2)
    if val_ratio <= 0 or val_ratio >= 1:
        val_ratio = 0.2

    train_full_set, train_loader, val_loader = None, None, None
    if args.train_root is not None and os.path.exists(args.train_root):
        # 1. 加载完整训练集（已提前过滤非图片文件）
        train_full_set = ImageDataset(
            istrain=True, 
            root=args.train_root, 
            data_size=args.data_size, 
            return_index=True
        )
        total_samples = len(train_full_set)
        print(f"[数据集] 完整训练集有效样本数：{total_samples}")

        # 2. 验证集逻辑
        if args.val_root is not None and os.path.exists(args.val_root):
            val_set = ImageDataset(
                istrain=False, 
                root=args.val_root, 
                data_size=args.data_size, 
                return_index=True
            )
            train_set = train_full_set
            print(f"[数据集] 使用用户提供的验证集，有效样本数：{len(val_set)}")
            # 新增：打印验证集类别分布（小样本场景排查用）
            val_labels = [info["label"] for info in val_set.data_infos]
            print(f"[数据集] 验证集包含类别数：{len(set(val_labels))}")
        else:
            # 自动拆分+类别分布校验
            val_size = max(1, int(total_samples * val_ratio))
            train_size = total_samples - val_size
            print(f"[数据集] 自动拆分：训练集{train_size}个，验证集{val_size}个（比例{val_ratio}）")

            generator = torch.Generator().manual_seed(42)
            train_subset, val_subset = random_split(
                train_full_set, [train_size, val_size], generator=generator
            )

            # 提取拆分后的样本信息
            train_indices = train_subset.indices
            val_indices = val_subset.indices
            train_data_infos = [train_full_set.data_infos[i] for i in train_indices]
            val_data_infos = [train_full_set.data_infos[i] for i in val_indices]

            # 新增：打印拆分后的类别分布（关键！小样本防类别缺失）
            train_labels = [info["label"] for info in train_data_infos]
            val_labels = [info["label"] for info in val_data_infos]
            print(f"[数据集] 训练集包含类别数：{len(set(train_labels))}")
            print(f"[数据集] 验证集包含类别数：{len(set(val_labels))}")
            if len(set(train_labels)) != len(set(val_labels)):
                print(f"[警告] 训练集和验证集类别不一致！可能影响验证效果（小样本建议调整val_ratio）")

            # 构建最终的训练集和验证集
            train_set = ImageDataset(istrain=True, root=args.train_root, data_size=args.data_size, return_index=True)
            train_set.data_infos = train_data_infos  # 替换为训练集子集信息
            val_set = ImageDataset(istrain=False, root=args.train_root, data_size=args.data_size, return_index=True)
            val_set.data_infos = val_data_infos  # 替换为验证集子集信息

        # 3. 构建DataLoader（修复collate_fn空列表问题）
        train_loader = torch.utils.data.DataLoader(
            train_set,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=safe_collate_fn,  # 用安全的collate_fn
            drop_last=True  # 小样本场景：丢弃最后一个不完整批次，避免空批次
        )

        val_loader = torch.utils.data.DataLoader(
            val_set,
            num_workers=min(args.num_workers, 8),
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=safe_collate_fn,
            drop_last=False
        )

    return train_loader, val_loader


# 新增：安全的collate_fn（避免空批次）
def safe_collate_fn(batch):
    """过滤无效样本，若批次为空则抛出警告并返回None（需训练代码兼容）"""
    batch = [x for x in batch if x is not None]
    if not batch:
        print(f"[警告] 当前批次无有效样本，已跳过")
        return None  # 后续训练代码需判断：若batch为None则跳过该批次
    return torch.utils.data.dataloader.default_collate(batch)


def get_dataset(args):
    if args.train_root is not None and os.path.exists(args.train_root):
        train_set = ImageDataset(
            istrain=True, 
            root=args.train_root, 
            data_size=args.data_size, 
            return_index=True
        )
        return train_set   
    return None


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, istrain: bool, root: str, data_size: int, return_index: bool = False):
        self.root = root
        self.data_size = data_size
        self.return_index = return_index

        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

        if istrain:
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

        self.data_infos = self.getDataInfo(root)

    def getDataInfo(self, root):
        data_infos = []
        folders = os.listdir(root)
        folders.sort()
        print("[dataset] 类别数:", len(folders))
        for class_id, folder in enumerate(folders):
            folder_path = os.path.join(root, folder)
            if not os.path.isdir(folder_path):
                print(f"[警告] 跳过非目录：{folder_path}")
                continue
            # 新增：仅保留图片文件（提前过滤，节省IO）
            files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            print(f"[dataset] 类别{folder}（ID：{class_id}）：{len(files)}张图片")
            for file in files:
                data_path = os.path.join(folder_path, file)
                data_infos.append({"path": data_path, "label": class_id})
        print(f"[dataset] 总有效图片数：{len(data_infos)}")
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        try:
            info = self.data_infos[index]
            image_path = info["path"]
            label = info["label"]
            required_size = self.data_size

            if not os.path.exists(image_path):
                raise IOError()
                # raise IOError(f"文件不存在")
            file_size = os.path.getsize(image_path)
            if file_size < 1024:
                raise IOError()
                # raise IOError(f"文件过小（{file_size} bytes）")

            # 多方式读取图片
            img = cv2.imread(image_path)
            if img is not None:
                img = img[:, :, ::-1]
            else:
                img = Image.open(image_path).convert("RGB")
                img = np.array(img)

            # 有效性校验
            if img.ndim != 3:
                raise IOError()
                # raise IOError(f"非3通道图片")
            h, w = img.shape[:2]
            if h < required_size // 2 or w < required_size // 2:
                raise IOError()
                # raise IOError(f"尺寸过小（{h}x{w}），需至少{required_size//2}x{required_size//2}")

            img = Image.fromarray(img)
            img = self.transforms(img)

            if self.return_index:
                return index, img, label
            return img, label

        except Exception as e:
            # print(f"[样本异常] 索引{index} | 标签{label} | 路径：{image_path} | 原因：{str(e)}")
            return None