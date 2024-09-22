import os
import re
import logging
import coloredlogs
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

"""
Current structure: Learned from GSVCities dataset, each image can be labeled as area_id(like lingang, zhangjiang, caohejin) + place_id(maybe extended with timestamp as well),
Use img_group.py to process data of SF_XL.
"""

# transformation will be done twice, first in TALADataset to do basic transforms, second in TALADataloader to set transforms for val&train data respectively.
default_transform = T.Compose([
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalizing once is quite abundant.
])

# BASE_PATH = '/home/SENSETIME/jiangyintian/TALA/Collection/TALADataset/sample/reg/' # hard code path to dataset, e.g. BASE_PATH/train_dir/*area/ BASE_PATH/val_dir/query(database)/*area/
# BASE_PATH = '/home/sensetime/data/TALA/TALAData/group/'
# BASE_PATH = 'E:/TALA/Sensetime/benchmark_data/pano'
BASE_PATH = 'E:/TALA/benchmark_data' # 

# Set up logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s - %(levelname)s - %(message)s')

class TALADataset(Dataset):
    def __init__(self, root_dir, transform=default_transform):
        super(TALADataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.unique_places = set()
       
        logger.info(f"Initializing dataset with root directory: {root_dir}")  # path to your dataset
        self._load_dataset()

    def _load_dataset(self):
        logger.info("Loading dataset...")
        for area_index, area in enumerate(sorted(os.listdir(self.root_dir))):
            area_path = os.path.join(self.root_dir, area)
            if os.path.isdir(area_path):
                logger.debug(f"Processing area: {area} (index {area_index})")
                label_prefix = f"{str(area_index+1).zfill(3)}"
                for img_name in os.listdir(area_path):
                    if img_name.endswith('.jpg'): # for kitti data, change to png
                        place_index = img_name.split('_')[1]
                        label = f"{label_prefix}{place_index}"
                        self.unique_places.add(label)
                        img_path = os.path.join(area_path, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(label)
                        # logger.debug(f"Added image: {img_path} with label {label}, currently there are {len(self.unique_places)} places and {len(self.labels)} images.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = int(self.labels[idx])
       
        if self.transform:
            image = self.transform(image)
       
        # logger.debug(f"Loaded image: {img_path} with label {label}")
        return image, label

class TALADataloader():
    def __init__(self, world_size=2, rank=0, image_size=(256, 1024), batch_size=32, num_workers=4, shuffle=False, train_dir="sf_train", val_dir="sf_val",
                 ddp=False):
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.world_size = world_size
        self.rank = rank
        self.ddp = ddp

        self.train_dir = BASE_PATH + train_dir
        # self.val_dir = BASE_PATH + val_dir
        self.val_dir = BASE_PATH
        self.query_dir = self.val_dir + "/query"
        self.db_dir = self.val_dir + "/database"
        # self.db_dir = self.val_dir + "/query" # for data cleaning
        
        self.query_image_path = []
        self.db_image_path = []

        self.train_transform = T.Compose([
            # T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.val_transform = T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': self.shuffle}

        self.valid_loader_config = {
            'batch_size': self.batch_size//2,  # consider /2 if out of memory
            'num_workers': self.num_workers//2,
            'drop_last': False,
            'pin_memory': False,
            'shuffle': False}

    def train_dataset(self):
        logger.info(f"Creating train_dataset on device[{self.rank}]...")
        train_dataset = TALADataset(root_dir=self.train_dir, transform=self.train_transform)
        return train_dataset

    def val_dataset(self):
        logger.info(f"Creating val_dataset on device[{self.rank}]...")
        query_dataset = TALADataset(root_dir=self.query_dir, transform=self.val_transform)
        db_dataset = TALADataset(root_dir=self.db_dir, transform=self.val_transform)
        return query_dataset, db_dataset

    def train_dataloader(self):
        train_dataset = self.train_dataset()
        if self.ddp:
            train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank)
            return DataLoader(train_dataset, **self.train_loader_config, sampler=train_sampler)
        return DataLoader(dataset=train_dataset, **self.train_loader_config)

    def val_dataloader(self):
        query_dataset, db_dataset = self.val_dataset()
        self.query_image_path = query_dataset.image_paths
        self.db_image_path = db_dataset.image_paths
        
        if self.ddp:
            query_sampler = DistributedSampler(query_dataset, num_replicas=self.world_size, rank=self.rank)
            db_sampler = DistributedSampler(db_dataset, num_replicas=self.world_size, rank=self.rank)
            query_loader = DataLoader(dataset=query_dataset, **self.valid_loader_config, sampler=query_sampler)
            db_loader = DataLoader(dataset=db_dataset, **self.valid_loader_config, sampler=db_sampler)
            return query_loader, db_loader
        query_loader = DataLoader(dataset=query_dataset, **self.valid_loader_config)
        db_loader = DataLoader(dataset=db_dataset, **self.valid_loader_config)
        return query_loader, db_loader


if __name__ == '__main__':
    # Check whether db and query are loaded as expected.
    loader = TALADataloader()
