import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import random_split, DataLoader, Dataset
from typing import List, Dict, Optional
from .randaug import RandAugment

from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------- 异常数据校验工具函数（仅初始化时调用） --------------------------
def is_valid_image(image_path: str) -> bool:
    """校验图片是否为有效文件（仅在数据集初始化时调用一次）"""
    if not os.path.exists(image_path):
        print(f"[Warning] 图片路径不存在，跳过：{image_path}")
        return False
    # 尝试读取图片（快速判断损坏）
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Warning] 图片损坏/无法读取，跳过：{image_path}")
        return False
    return True

# -------------------------- 基础数据集类（优化异常校验逻辑） --------------------------
class BaseImageDataset(Dataset):
    def __init__(self, istrain: bool, root: str, data_size: int, return_index: bool = False):
        self.root = root
        self.data_size = data_size
        self.return_index = return_index
        self.data_infos: List[Dict[str, str]] = []  # 存储 (path, label) 的有效数据，初始化时已过滤
        
        # 图像预处理（保持原有逻辑）
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if istrain:
            self.transforms = transforms.Compose([
                transforms.Resize((510, 510), Image.BILINEAR),
                transforms.RandomCrop((data_size, data_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1), # 保留原有增强
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
        
        # 加载并过滤有效数据（仅初始化时校验一次，训练中不再检查）
        raw_infos = self._get_data_infos(root)
        self.data_infos = self._filter_invalid_data(raw_infos)
        print(f"[Dataset] 有效样本数：{len(self.data_infos)}（原始：{len(raw_infos)}，过滤异常：{len(raw_infos)-len(self.data_infos)}）")

    def _get_data_infos(self, root: str) -> List[Dict[str, str]]:
        """子类实现：获取原始数据信息"""
        raise NotImplementedError("子类需实现 _get_data_infos 方法")

    def _filter_invalid_data(self, raw_infos: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """多线程过滤异常数据（仅在初始化时执行一次）"""
        valid_infos = []
        total = len(raw_infos)
        if total == 0:
            print("[Dataset] 警告：原始数据为空！")
            return valid_infos
        
        print(f"[Dataset] 开始初始化过滤 {total} 个样本...")
        # 线程数：根据CPU核心数动态调整（避免线程竞争）
        max_workers = min(os.cpu_count() or 4, 16)  # 最多16线程，避免过度竞争
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_info = {executor.submit(is_valid_image, info["path"]): info for info in raw_infos}
            for i, future in enumerate(as_completed(future_to_info)):
                info = future_to_info[future]
                try:
                    if future.result():
                        valid_infos.append(info)
                except Exception as e:
                    pass
                #    print(f"[Error] 校验图片 {info['path']} 出错：{str(e)}")
                
                # 打印进度（每1000个样本更新）
                #if (i + 1) % 1000 == 0 or (i + 1) == total:
                #    print(f"[Dataset] 过滤进度：{i+1}/{total}（有效：{len(valid_infos)}）")
        
        return valid_infos

    def __len__(self) -> int:
        return len(self.data_infos)

    def __getitem__(self, index: int):
        """训练中不重复校验（假设初始化时已过滤所有异常）"""
        info = self.data_infos[index]
        image_path = info["path"]
        label = info["label"]
        
        # 读取图片（无try-except，依赖初始化过滤）
        img = cv2.imread(image_path)
        img = img[:, :, ::-1]  # BGR -> RGB
        img = Image.fromarray(img)
        img = self.transforms(img)
        
        if self.return_index:
            return index, img, label
        return img, label

# -------------------------- 子类实现（普通图片数据集） --------------------------
class ImageDataset(BaseImageDataset):
    def _get_data_infos(self, root: str) -> List[Dict[str, str]]:
        raw_infos = []
        folders = os.listdir(root)
        folders.sort()  # 按字母序排序
        print(f"[Dataset] 类别数：{len(folders)}")
        
        for class_id, folder in enumerate(folders):
            folder_path = os.path.join(root, folder)
            if not os.path.isdir(folder_path):
                print(f"[Warning] 非目录，跳过：{folder_path}")
                continue
            
            files = os.listdir(folder_path)
            for file in files:
                file_path = os.path.join(folder_path, file)
                # 仅保留常见图片格式
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    raw_infos.append({"path": file_path, "label": class_id})
        return raw_infos

# -------------------------- 数据加载器构建（核心优化点） --------------------------
def build_loader(args):
    train_set, val_set = None, None
    train_loader, val_loader = None, None

    # 选择数据集类
    dataset_class = ImageDataset  # 若有finegrain可扩展

    # -------------------------- 构建训练集 --------------------------
    if args.train_root is not None and os.path.exists(args.train_root):
        full_train_set = dataset_class(
            istrain=True,
            root=args.train_root,
            data_size=args.data_size,
            return_index=True
        )
        
        # 自动拆分训练集/验证集（无独立验证集时）
        if args.val_root is None or args.val_root == "~":
            val_ratio = 0.1
            total_size = len(full_train_set)
            val_size = int(total_size * val_ratio)
            train_size = total_size - val_size
            
            train_set, val_set = random_split(
                full_train_set,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            print(f"[Loader] 自动拆分：{train_size} 训练样本 + {val_size} 验证样本")
        
        # 独立训练集
        else:
            train_set = full_train_set
            print(f"[Loader] 训练集样本数：{len(train_set)}")

    # -------------------------- 构建独立验证集 --------------------------
    if args.val_root is not None and args.val_root != "~" and os.path.exists(args.val_root):
        val_set = dataset_class(
            istrain=False,
            root=args.val_root,
            data_size=args.data_size,
            return_index=True
        )
        print(f"[Loader] 独立验证集样本数：{len(val_set)}")

    # -------------------------- 配置DataLoader（核心优化） --------------------------
    # 1. 计算合理的num_workers（避免线程竞争）
    cpu_count = os.cpu_count() or 4
    # 训练集workers：CPU核心数的1/2 ~ 2/3（平衡负载）
    train_workers = max(min(cpu_count // 2, 16), 2)  # 至少2，最多16
    # 验证集workers：训练集的1/2（避免验证时占用过多资源）
    val_workers = max(train_workers // 2, 1)  # 至少1
    
    # 覆盖用户传入的不合理配置（可选）
    if args.num_workers is not None:
        train_workers = max(min(args.num_workers, 16), 2)
        val_workers = max(train_workers // 2, 1)
    print(f"[Loader] 训练集workers：{train_workers}，验证集workers：{val_workers}")

    # 2. 通用collate_fn（无需过滤None，因初始化已校验）
    def collate_fn(batch):
        return torch.utils.data.dataloader.default_collate(batch)

    # 3. 训练集DataLoader（启用persistent_workers）
    if train_set is not None and len(train_set) > 0:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=train_workers,
            pin_memory=torch.cuda.is_available(),  # 仅GPU时启用
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=True,  # 启用worker持久化
        )

    # 4. 验证集DataLoader（保持合理比例workers）
    if val_set is not None and len(val_set) > 0:
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
            drop_last=False,
            persistent_workers=True,  # 验证集也启用持久化
        )

    print(f"[Loader] 加载完成：训练集{bool(train_loader)}，验证集{bool(val_loader)}")
    return train_loader, val_loader

# -------------------------- 辅助函数 --------------------------
def get_dataset(args):
    if args.train_root is None or not os.path.exists(args.train_root):
        print(f"[Warning] 训练集路径不存在：{args.train_root}")
        return None
    
    dataset_class = ImageDataset  # 可扩展finegrain
    train_set = dataset_class(
        istrain=True,
        root=args.train_root,
        data_size=args.data_size,
        return_index=True
    )
    print(f"[get_dataset] 训练集有效样本：{len(train_set)}")
    return train_set