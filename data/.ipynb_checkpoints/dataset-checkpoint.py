import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from .randaug import RandAugment  # 假设randaug.py在同级目录


def build_loader(args):
    """构建训练/验证数据加载器，支持自动拆分验证集"""
    train_loader, val_loader = None, None
    val_ratio = getattr(args, 'val_ratio', 0.2)  # 验证集比例，默认10%
    if val_ratio <= 0 or val_ratio >= 1:
        val_ratio = 0.1  # 兜底：确保比例有效

    # 选择数据集类（普通/细粒度）
    DatasetClass = ImageDataset_FG if args.finegrain else ImageDataset

    # 加载训练集并处理验证集
    if args.train_root and os.path.exists(args.train_root):
        # 1. 加载完整训练集元数据（用于拆分）
        full_train_dataset = DatasetClass(
            istrain=True,
            root=args.train_root,
            data_size=args.data_size,
            return_index=True
        )
        total_samples = len(full_train_dataset)
        print(f"[Loader] 训练集总样本数：{total_samples}")

        # 2. 检查验证集路径是否有效
        val_exists = args.val_root and os.path.exists(args.val_root)
        if not val_exists:
            # 自动从训练集拆分验证集
            val_size = max(1, int(total_samples * val_ratio))  # 至少1个样本
            train_size = total_samples - val_size
            print(f"[Loader] 未找到有效验证集路径，自动拆分：{train_size}训练 + {val_size}验证")

            # 固定随机种子，确保拆分可复现
            generator = torch.Generator().manual_seed(42)
            train_subset, val_subset = random_split(
                full_train_dataset,
                [train_size, val_size],
                generator=generator
            )

            # 验证集使用评估模式（无随机增强）
            val_dataset = DatasetClass(
                istrain=False,
                root=args.train_root,
                data_size=args.data_size,
                return_index=True,
                data_infos=[full_train_dataset.data_infos[i] for i in val_subset.indices]
            )
            train_dataset = train_subset

        else:
            # 加载独立验证集
            train_dataset = full_train_dataset
            val_dataset = DatasetClass(
                istrain=False,
                root=args.val_root,
                data_size=args.data_size,
                return_index=True
            )
            print(f"[Loader] 加载独立验证集，样本数：{len(val_dataset)}")

        # 3. 构建DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=min(args.num_workers, 12),  # 避免线程过多
            pin_memory=True,
            collate_fn=collate_fn,  # 过滤无效样本
            persistent_workers=True  # 提升加载效率
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size * 2,  # 验证集批次加倍
            shuffle=False,  # 验证集不打乱
            num_workers=min(args.num_workers, 12),
            pin_memory=True,
            collate_fn=collate_fn
        )

    return train_loader, val_loader


def get_dataset(args):
    """获取训练集（用于其他场景，如数据统计）"""
    if not args.train_root or not os.path.exists(args.train_root):
        print(f"[Warning] 训练集路径无效：{args.train_root}")
        return None
    DatasetClass = ImageDataset_FG if args.finegrain else ImageDataset
    return DatasetClass(
        istrain=True,
        root=args.train_root,
        data_size=args.data_size,
        return_index=True
    )


def collate_fn(batch):
    """处理批次中的无效样本（过滤None）"""
    batch = [x for x in batch if x is not None]
    if not batch:
        raise ValueError("当前批次无有效样本，请检查数据集或增大batch_size")
    return torch.utils.data.dataloader.default_collate(batch)


class BaseDataset(Dataset):
    """基础数据集类，抽取公共逻辑"""
    def __init__(self, istrain, root, data_size, return_index, data_infos=None):
        self.istrain = istrain
        self.root = root
        self.data_size = data_size
        self.return_index = return_index
        self.data_infos = data_infos if data_infos is not None else self._get_data_infos()
        self._init_transforms()  # 初始化数据增强

    def _get_data_infos(self):
        """子类实现：获取数据路径和标签"""
        raise NotImplementedError

    def _init_transforms(self):
        """初始化数据增强（强化版，抗过拟合）"""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if self.istrain:
            # 训练集：强增强策略（小数据集专用）
            self.transforms = transforms.Compose([
                transforms.Resize((510, 510), Image.BILINEAR),
                # 随机裁剪+缩放（增强尺度鲁棒性）
                transforms.RandomResizedCrop(
                    (self.data_size, self.data_size),
                    scale=(0.5, 1.0),  # 裁剪区域50%-100%
                    ratio=(0.8, 1.2)   # 宽高比波动
                ),
                # 随机翻转
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.2),
                # # 随机旋转
                # transforms.RandomRotation(degrees=30),
                # # 颜色抖动
                # transforms.ColorJitter(
                #     brightness=0.3,
                #     contrast=0.3,
                #     saturation=0.3,
                #     hue=0.1
                # ),
                # # 随机灰度化
                # transforms.RandomGrayscale(p=0.1),
                # 高斯模糊
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
                ], p=0.2),
                # 自动增强（RandAugment）
                RandAugment(n=2, m=10),  # 小数据集增强强度加大
                # 转为张量并归一化
                transforms.ToTensor(),
                normalize
                # 随机擦除（遮挡部分区域）
                # transforms.RandomErasing(p=0.2, scale=(0.02, 0.33))
            ])
        else:
            # 验证集：仅确定性变换
            self.transforms = transforms.Compose([
                transforms.Resize((510, 510), Image.BILINEAR),
                transforms.CenterCrop((self.data_size, self.data_size)),
                transforms.ToTensor(),
                normalize
            ])

    def _load_image(self, image_path):
        """加载图片，处理异常情况"""
        try:
            # 尝试读取图片
            img = cv2.imread(image_path)
            if img is None:
                raise IOError("图片无法读取（可能损坏）")
            img = img[:, :, ::-1]  # BGR转RGB
            return Image.fromarray(img)
        except Exception as e:
            print(f"[Warning] 图片 {image_path} 无效：{str(e)}")
            # 训练集返回增强占位图，验证集返回None（由collate_fn过滤）
            if self.istrain:
                # 随机噪声占位图（增加多样性）
                img = Image.fromarray(
                    np.random.randint(0, 256, (self.data_size, self.data_size, 3), dtype=np.uint8)
                )
                # 对占位图额外增强，避免模型记住
                return transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.5, contrast=0.5)
                ], p=0.8)(img)
            else:
                return None

    def __getitem__(self, index):
        try:
            info = self.data_infos[index]
            img = self._load_image(info["path"])
            if img is None:
                return None  # 验证集无效样本直接过滤
            img = self.transforms(img)
            if self.return_index:
                return index, img, info["label"]
            return img, info["label"]
        except Exception as e:
            print(f"[Error] 处理样本 {index} 失败：{str(e)}")
            return None

    def __len__(self):
        return len(self.data_infos)


class ImageDataset(BaseDataset):
    """普通图片数据集（单级目录结构：root/类别/图片）"""
    def _get_data_infos(self):
        """获取所有有效图片的路径和标签"""
        data_infos = []
        classes = sorted(os.listdir(self.root))  # 按字母序排序类别
        print(f"[Dataset] 普通分类 - 类别数：{len(classes)}")

        for class_id, class_name in enumerate(classes):
            class_path = os.path.join(self.root, class_name)
            if not os.path.isdir(class_path):
                print(f"[Warning] 跳过非目录：{class_path}")
                continue
            # 遍历类别下的图片
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    img_path = os.path.join(class_path, img_name)
                    data_infos.append({"path": img_path, "label": class_id})

        print(f"[Dataset] 普通分类 - 总样本数：{len(data_infos)}")
        return data_infos


class ImageDataset_FG(BaseDataset):
    """细粒度图片数据集（两级目录结构：root/粗类/细类/图片）"""
    def _get_data_infos(self):
        """获取细粒度分类的路径和标签（细类ID全局唯一）"""
        data_infos = []
        coarse_classes = sorted(os.listdir(self.root))
        print(f"[Dataset] 细粒度分类 - 粗类别数：{len(coarse_classes)}")

        fine_class_id = 0  # 细类别全局计数器
        for coarse_name in coarse_classes:
            coarse_path = os.path.join(self.root, coarse_name)
            if not os.path.isdir(coarse_path):
                print(f"[Warning] 跳过非粗类目录：{coarse_path}")
                continue
            # 遍历粗类下的细类
            fine_classes = sorted(os.listdir(coarse_path))
            for fine_name in fine_classes:
                fine_path = os.path.join(coarse_path, fine_name)
                if not os.path.isdir(fine_path):
                    print(f"[Warning] 跳过非细类目录：{fine_path}")
                    continue
                # 遍历细类下的图片
                for img_name in os.listdir(fine_path):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        img_path = os.path.join(fine_path, img_name)
                        data_infos.append({"path": img_path, "label": fine_class_id})
                fine_class_id += 1  # 细类别ID递增

        print(f"[Dataset] 细粒度分类 - 总细类别数：{fine_class_id}，总样本数：{len(data_infos)}")
        return data_infos