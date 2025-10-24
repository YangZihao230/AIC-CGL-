import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import random_split, DataLoader, Dataset
from typing import List, Dict, Optional
from .randaug import RandAugment  # 保留原有依赖

from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------- 新增：异常数据校验工具函数 --------------------------
def is_valid_image(image_path: str) -> bool:
    """
    校验图片是否为有效文件（存在性 + 可读取性）
    返回 True：有效图片；False：无效/损坏图片
    """
    # 1. 先判断文件是否存在
    if not os.path.exists(image_path):
        print(f"[Warning] 图片路径不存在，跳过：{image_path}")
        return False
    # 2. 尝试用 cv2 读取（快速判断是否损坏）
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Warning] 图片损坏/无法读取，跳过：{image_path}")
        return False
    # 3. 可选：校验图片尺寸（若需过滤过小图片，可添加此处）
    # if img.shape[0] < 32 or img.shape[1] < 32:
    #     print(f"[Warning] 图片尺寸过小，跳过：{image_path}")
    #     return False
    return True

# -------------------------- 基类优化：减少代码冗余（可选但推荐） --------------------------
class BaseImageDataset(Dataset):
    """基础图片数据集类，抽取公共逻辑（避免 ImageDataset 和 ImageDataset_FG 重复代码）"""
    def __init__(self, istrain: bool, root: str, data_size: int, return_index: bool = False):
        self.root = root
        self.data_size = data_size
        self.return_index = return_index
        self.data_infos: List[Dict[str, str]] = []  # 存储 (path, label) 的有效数据
        
        # 统一的图像预处理（与原代码一致）
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        
        # 加载并过滤有效数据（核心：调用子类实现的 _get_data_infos 并过滤）
        raw_infos = self._get_data_infos(root)
        self.data_infos = self._filter_invalid_data(raw_infos)
        print(f"[Dataset] 有效样本数：{len(self.data_infos)}（原始样本数：{len(raw_infos)}，过滤异常数：{len(raw_infos)-len(self.data_infos)}）")

    def _get_data_infos(self, root: str) -> List[Dict[str, str]]:
        """子类必须实现：获取原始数据信息（未过滤）"""
        raise NotImplementedError("子类需实现 _get_data_infos 方法")

    def _filter_invalid_data(self, raw_infos: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """多线程过滤异常数据：并行校验图片有效性"""
        valid_infos = []
        total = len(raw_infos)
        print(f"[Dataset] 开始多线程过滤 {total} 个样本...")

        # 多线程并行处理（线程数建议设为CPU核心数，如8核设8）
        max_workers = min(16, os.cpu_count() or 4)  # 限制最大线程数，避免资源耗尽
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有校验任务：key=未来对象，value=原始信息
            future_to_info = {executor.submit(is_valid_image, info["path"]): info for info in raw_infos}
            
            # 异步获取结果，按完成顺序处理
            for i, future in enumerate(as_completed(future_to_info)):
                info = future_to_info[future]
                try:
                    if future.result():  # 若校验通过
                        valid_infos.append(info)
                except Exception as e:
                    print(f"[Error] 校验图片 {info['path']} 时出错：{str(e)}")
                
                # 打印进度（每1000个样本更新一次）
                if (i + 1) % 1000 == 0 or (i + 1) == total:
                    print(f"[Dataset] 过滤进度：{i+1}/{total}（有效样本：{len(valid_infos)}）")

        return valid_infos

    def __len__(self) -> int:
        return len(self.data_infos)

    def __getitem__(self, index: int):
        """读取数据（添加 try-except 兜底，避免单个异常数据中断训练）"""
        try:
            info = self.data_infos[index]
            image_path = info["path"]
            label = info["label"]
            
            # 读取图片（与原代码一致）
            img = cv2.imread(image_path)
            img = img[:, :, ::-1]  # BGR -> RGB
            img = Image.fromarray(img)
            img = self.transforms(img)
            
            if self.return_index:
                return index, img, label
            return img, label
        except Exception as e:
            print(f"[Warning] 读取索引 {index} 数据失败（路径：{self.data_infos[index]['path']}），错误：{str(e)}")
            # 返回一个占位值（后续通过 collate_fn 过滤）
            return None

# -------------------------- 子类实现：保留原有数据集逻辑 --------------------------
class ImageDataset(BaseImageDataset):
    """普通图片数据集（对应原 ImageDataset）"""
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
                # 仅保留图片文件（可选：进一步过滤后缀）
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    raw_infos.append({"path": file_path, "label": class_id})
        return raw_infos



# -------------------------- 数据加载器构建：保留验证集逻辑 + 过滤占位值 --------------------------
def build_loader(args):
    train_set, val_set = None, None
    train_loader, val_loader = None, None

    # 选择数据集类（与原逻辑一致）
    dataset_class =ImageDataset

    # -------------------------- 构建训练集（保留自动拆分逻辑） --------------------------
    if args.train_root is not None and os.path.exists(args.train_root):
        # 加载过滤后的训练集（已自动跳过异常数据）
        full_train_set = dataset_class(
            istrain=True,
            root=args.train_root,
            data_size=args.data_size,
            return_index=True
        )
        
        # 无独立验证集：从训练集拆分（与原逻辑一致）
        if args.val_root is None or args.val_root == "~":
            val_ratio = 0.1
            total_size = len(full_train_set)
            val_size = int(total_size * val_ratio)
            train_size = total_size - val_size
            
            # 拆分时使用稳定种子（保证可复现）
            train_set, val_set = random_split(
                full_train_set,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            print(f"[Loader] 自动拆分训练集/验证集：{train_size} 训练样本 + {val_size} 验证样本")
        
        # 有独立验证集：仅保留训练集（后续单独加载验证集）
        else:
            train_set = full_train_set
            print(f"[Loader] 加载训练集：{len(train_set)} 样本（已过滤异常数据）")

    # -------------------------- 构建独立验证集（与原逻辑一致） --------------------------
    if args.val_root is not None and args.val_root != "~" and os.path.exists(args.val_root):
        val_set = dataset_class(
            istrain=False,
            root=args.val_root,
            data_size=args.data_size,
            return_index=True
        )
        print(f"[Loader] 加载独立验证集：{len(val_set)} 样本（已过滤异常数据）")

    # -------------------------- 构建 DataLoader（添加自定义 collate_fn 过滤占位值） --------------------------
    def collate_fn(batch):
        """过滤 batch 中的 None 值（处理 __getitem__ 中的兜底异常）"""
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            raise ValueError("当前 batch 无有效数据，请检查数据集质量或增大 batch_size")
        return torch.utils.data.dataloader.default_collate(batch)

    # 训练集 DataLoader（优化参数：num_workers 建议设为 CPU 核心数的 1/2 ~ 2/3，避免线程竞争）
    if train_set is not None and len(train_set) > 0:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(args.num_workers, 8),  # 16核CPU建议设8（避免IO阻塞）
            pin_memory=True,  # 加速GPU数据传输（若使用GPU）
            collate_fn=collate_fn,  # 过滤无效样本
            drop_last=True  # 丢弃最后一个不完整的batch（避免训练不稳定）
        )

    # 验证集 DataLoader（与原逻辑一致，num_workers=1 避免验证时占用过多资源）
    if val_set is not None and len(val_set) > 0:
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False
        )

    # 日志输出：加载完成信息
    print(f"[Loader] 数据加载完成：训练集 loader {'存在' if train_loader else '不存在'}，验证集 loader {'存在' if val_loader else '不存在'}")
    return train_loader, val_loader

# -------------------------- 保留原有 get_dataset 函数（适配修改后的 Dataset） --------------------------
def get_dataset(args):
    if args.train_root is None or not os.path.exists(args.train_root):
        print(f"[Warning] 训练集路径不存在：{args.train_root}")
        return None
    
    dataset_class = ImageDataset_FG if args.finegrain else ImageDataset
    train_set = dataset_class(
        istrain=True,
        root=args.train_root,
        data_size=args.data_size,
        return_index=True
    )
    print(f"[get_dataset] 加载训练集：{len(train_set)} 有效样本")
    return train_set