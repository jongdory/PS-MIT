import numpy as np
from torch.utils.data import Dataset

import os
import pandas
from monai import transforms
from monai.data import Dataset as MonaiDataset
import torch

from scipy.ndimage import center_of_mass
from monai.transforms import MapTransform, SpatialCrop
from monai.utils import ensure_tuple

class CropAroundCenterOfMassd(MapTransform):
    def __init__(self, keys, label_key, spatial_size, channel_index=0):
        super().__init__(keys)
        self.label_key = label_key
        self.spatial_size = ensure_tuple(spatial_size)
        self.channel_index = channel_index

    def __call__(self, data):
        d = dict(data)
        label = d[self.label_key]

        if isinstance(label, torch.Tensor):
            label_np = label.detach().cpu().numpy()
        else:
            label_np = label
            
        if label_np.ndim == 4:
            target_mask = label_np[self.channel_index].copy()
        else:
            target_mask = label_np.copy()
        
        image_shape = target_mask.shape
        center_coords = [dim // 2 for dim in image_shape]
        target_mask[target_mask == 2] = 0
        target_mask[target_mask != 0] = 1
        if np.sum(target_mask) != 0:
            center_coords[-1] = center_of_mass(target_mask)[-1]

        cropper = SpatialCrop(roi_center=center_coords, roi_size=self.spatial_size)
        for key in self.key_iterator(d):
            d[key] = cropper(d[key])
            
        return d

def get_transforms(dataset="brats", phase="train"):

    if dataset == "brats":
        modalities = ["t1", "t1ce", "t2", "flair", "seg"]
    elif dataset == "ixi":
        modalities = ["t1", "t2", "pd"]

    if phase == "train":
        crop_transforms_list = []
        crop_transforms_list.append(transforms.RandSpatialCropd(keys=modalities, roi_size=(224, 224, 1), random_size=False, allow_missing_keys=True))
        crop_transforms = transforms.Compose(crop_transforms_list + [

            ])
    elif phase == "val":
        crop_transforms = transforms.Compose(
            [
                transforms.CenterSpatialCropd(keys=modalities, roi_size=(224, 224, 1), allow_missing_keys=True),
            ]
        )
    else:
        crop_transforms = transforms.Compose(
            [
                CropAroundCenterOfMassd(keys=modalities, 
                label_key="seg", 
                spatial_size=(224, 224, 1), 
                channel_index=0
                ) if "seg" in modalities else transforms.CenterSpatialCropd(
                    keys=modalities, 
                    roi_size=(224, 224, 32), 
                    allow_missing_keys=True
                ),
            ]
        )
    
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=modalities, allow_missing_keys=True),
            transforms.AddChanneld(keys=modalities, allow_missing_keys=True),
            transforms.Orientationd(keys=modalities, axcodes="RAS", allow_missing_keys=True),
            transforms.EnsureTyped(keys=modalities, allow_missing_keys=True),
            transforms.CropForegroundd(keys=modalities, source_key="t1", margin=0, allow_missing_keys=True),
            transforms.SpatialPadd(keys=modalities, spatial_size=(224, 224, 144), allow_missing_keys=True),
            transforms.ScaleIntensityRangePercentilesd(keys=modalities, lower=0, upper=99.5, b_min=0, b_max=1, clip=True, allow_missing_keys=True),
            crop_transforms,
        ]
    )

def load_data(
    *,
    data_dir,
    phase="train",
    dataset = "brats",
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    if "brats" in dataset:
        if phase == "train":
            dataset = BraTS2021Train(
                data_dir,
                phase="train",
                dataset=dataset,
            )
        elif phase == "test":
            dataset = BraTS2021Test(
                data_dir,
                phase="test",
                dataset=dataset,
            )
    elif "ixi" in dataset:
        if phase == "train":
            dataset = IXITrain(
                data_dir,
                phase="train",
                dataset=dataset,
            )
        elif phase == "test":
            dataset = IXITest(
                data_dir,
                phase="test",
                dataset=dataset,
            )
    
    return dataset
    


def get_brats_dataset(data_path, phase="train", dataset="brats"):
    transform = get_transforms(dataset=dataset, phase=phase)
    
    datalist = []
    csv_path = "dataset/brats_split.csv"
    df = pandas.read_csv(csv_path)
    for sub_id in df["BraTS2021"].tolist():
        split_list = df[df["BraTS2021"] == sub_id]["split"].tolist()
        if split_list and split_list[0] == phase:
            datalist.append(sub_id)

    data = []

    for subject in datalist:
        sub_path = os.path.join(data_path, subject)
        
        if os.path.exists(sub_path) == False: 
            continue

        t1 = os.path.join(sub_path, f"{subject}_t1.nii.gz")
        t1ce = os.path.join(sub_path, f"{subject}_t1ce.nii.gz")
        t2 = os.path.join(sub_path, f"{subject}_t2.nii.gz")
        flair = os.path.join(sub_path, f"{subject}_flair.nii.gz")
        seg = os.path.join(sub_path, f"{subject}_seg.nii.gz")

        data.append({"t1":t1, "t1ce":t1ce, "t2":t2, "flair":flair, "seg":seg, "subject_id": subject, "path": t1})
                    
    print(phase, " num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)


def get_ixi_dataset(data_path, phase="train", dataset="ixi"):
    transform = get_transforms(dataset=dataset, phase=phase)
    datalist = []

    csv_path = "dataset/ixi_split.csv"
    df = pandas.read_csv(csv_path)
    for sub_id in df["id"].tolist():
        split_list = df[df["id"] == sub_id]["split"].tolist()
        if split_list and split_list[0] == phase:
            datalist.append(sub_id)

    data = []
    for subject in datalist:
        sub_id = subject
        subject = f"IXI{sub_id:03d}"
        sub_path = os.path.join(data_path, subject)

        if os.path.exists(sub_path) == False: 
            continue

        t1 = os.path.join(sub_path, "T1.nii.gz")
        t2 = os.path.join(sub_path, "T2.nii.gz")
        pd = os.path.join(sub_path, "PD.nii.gz")

        if os.path.exists(t1) == False or os.path.exists(t2) == False or os.path.exists(pd) == False:   
            continue

        data.append({"t1":t1, "t2":t2, "pd":pd, "subject_id": subject, "path": t1})
                    
    print(phase, " num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)

class ImageBase(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]


class BrainTrain(ImageBase):    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = dict(self.data[i])
        if self.dataset == "brats":
            image = torch.cat([item["t1"], item["t2"], item["t1ce"], item["flair"]], dim=0)
        elif self.dataset == "ixi":
            image = torch.cat([item["t1"], item["t2"], item["pd"]], dim=0)
        
        return image

class BrainTest(ImageBase):
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = dict(self.data[i])
        if self.dataset == "brats":
            image = torch.cat([item["t1"], item["t2"], item["t1ce"], item["flair"]], dim=0)
        elif self.dataset == "ixi":
            image = torch.cat([item["t1"], item["t2"], item["pd"]], dim=0)
        subject = item["subject_id"]

        return image, subject


class BraTS2021Train(BrainTrain):
    def __init__(self, data_path, phase="train", dataset="brats"):
        super().__init__(dataset=dataset)
        self.data = get_brats_dataset(data_path, phase, dataset)
        self.dataset = dataset

class BraTS2021Test(BrainTest):
    def __init__(self, data_path, phase="test", dataset="brats"):
        super().__init__(dataset=dataset)
        self.data = get_brats_dataset(data_path, phase, dataset)
        self.dataset = dataset

class IXITrain(BrainTrain):
    def __init__(self, data_path, phase="train", dataset="ixi"):
        super().__init__(dataset=dataset)
        self.data = get_ixi_dataset(data_path, phase, dataset)
        self.dataset = dataset

class IXITest(BrainTest):
    def __init__(self, data_path, phase="test", dataset="ixi"):
        super().__init__(dataset=dataset)
        self.data = get_ixi_dataset(data_path, phase, dataset)
        self.dataset = dataset