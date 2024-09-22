import torch
import numpy as np
import logging
import colorlog
from dataloaders.TALAData import TALADataloader
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image, ImageDraw, ImageFont
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import os
import faiss
import cv2

from typing import Tuple
import vpr_models
from skimage.transform import rescale


# Configure the logger
logger = logging.getLogger('VPREvaluator')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)s: %(message)s',
    log_colors={
        'DEBUG': 'white',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)

handler.setFormatter(formatter)
logger.addHandler(handler)

# global varibales
# Height and width of a single image
H = 512
W = 512
TEXT_H = 175
FONTSIZE = 50
SPACE = 50  # Space between two images

class VPREvaluator:
    def __init__(self, model_name, db_loader, query_loader, query_path, db_path, recall_k=5, batch_size=32, device='cuda'):
        self.db_loader = db_loader
        self.query_loader = query_loader
        self.recall_k = recall_k
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.faiss_gpu = False
        

        self.query_image_path = query_path
        self.db_image_path = db_path
        logger.info(f"Loading {model_name} model.")
        self.model_name = model_name
        self.model = self.load_model(model_name).to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self, model_name):
        model = vpr_models.get_model(method=model_name, descriptors_dimension=512)
        return model
    
    def record_matches(self,
                    top_k_matches,
                   out_file: str = 'record.txt') -> None:
        with open(f'{out_file}', 'w') as f:   # "a" means append, which will add content to existing file
            for query_index, db_indices in enumerate(tqdm(top_k_matches, ncols=100, desc='Recording matches')):
                pred_query_path = self.query_image_path[query_index]
                pred_db_paths = []
                for i in db_indices.tolist():
                    pred_db_paths.append(self.db_image_path[i])
                    # logger.debug(f"prediction[{query_index}]: db_image_paths[{i}] is {self.db_image_path[i]}")
                f.write(f'prediction[{query_index}]: query_path: {pred_query_path} & db_paths: {pred_db_paths}\n')
    
    def write_labels_to_image(self, labels=["text1", "text2"]):
        """Creates an image with text"""
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONTSIZE)
        img = Image.new('RGB', ((W * len(labels)) + 50 * (len(labels)-1), TEXT_H), (1, 1, 1))
        d = ImageDraw.Draw(img)
        for i, text in enumerate(labels):
            _, _, w, h = d.textbbox((0,0), text, font=font)
            d.text(((W+SPACE)*i + W//2 - w//2, 1), text, fill=(0, 0, 0), font=font)
        return np.array(img)[:100]  # Remove some empty space

    def draw(self, img, c=(0, 255, 0), thickness=20):
        """Draw a colored (usually red or green) box around an image."""
        p = np.array([[0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0]])
        for i in range(3):
            cv2.line(img, (p[i, 0], p[i, 1]), (p[i+1, 0], p[i+1, 1]), c, thickness=thickness*2)
        return cv2.line(img, (p[3, 0], p[3, 1]), (p[0, 0], p[0, 1]), c, thickness=thickness*2)


    def build_prediction_image(self, images_paths, preds_correct):
        """Build a row of images, where the first is the query and the rest are predictions.
        For each image, if is_correct then draw a green/red box.
        """

        assert len(images_paths) == len(preds_correct)
        labels = ["Query"]
        for i, is_correct in enumerate(preds_correct[1:]):
            if is_correct is None:
                labels.append(f"Pred{i}")
            else:
                labels.append(f"Pred{i} - {is_correct}")
        
        num_images = len(images_paths)
        images = [np.array(Image.open(path).convert("RGB")) for path in images_paths]
        for img, correct in zip(images, preds_correct):
            if correct is None:
                continue
            color = (0, 255, 0) if correct else (255, 0, 0)
            self.draw(img, color)
        concat_image = np.ones([H, (num_images*W)+((num_images-1)*SPACE), 3])
        rescaleds = [rescale(i, [min(H/i.shape[0], W/i.shape[1]), min(H/i.shape[0], W/i.shape[1]), 1]) for i in images]
        for i, image in enumerate(rescaleds):
            pad_width = (W - image.shape[1] + 1) // 2
            pad_height = (H - image.shape[0] + 1) // 2
            image = np.pad(image, [[pad_height, pad_height], [pad_width, pad_width], [0, 0]], constant_values=1)[:H, :W]
            concat_image[: , i*(W+SPACE) : i*(W+SPACE)+W] = image
        try:
            labels_image = self.write_labels_to_image(labels)
            final_image = np.concatenate([labels_image, concat_image])
        except OSError:  # Handle error in case of missing PIL ImageFont
            final_image = concat_image
        final_image = Image.fromarray((final_image*255).astype(np.uint8))
        return final_image
    

    def visualize(self, indices,
            query_labels, database_labels,
            visual_dir: str = './LOGS/visualize',
            img_resize_size: Tuple = (1024, 256)) -> None:
        
        if not os.path.exists(visual_dir):
            os.makedirs(visual_dir)

        for query_index, preds in enumerate(tqdm(indices, desc=f"Saving results in {visual_dir}")):
            query_path = self.query_image_path[query_index]
            list_of_images_paths = [query_path]
            # List of None (query), True (correct preds) or False (wrong preds)
            preds_correct = [None] # None holds place for query
            for pred_index, pred in enumerate(preds):  # pred: index of retrieval in database
                pred_path = self.db_image_path[pred]
                list_of_images_paths.append(pred_path)
                if query_labels[query_index] in database_labels[indices[query_index, :5]]: # indices are composed of 1 query and 5 retrivals
                    is_correct = True
                else:
                    is_correct = None
                preds_correct.append(is_correct)
            
            save_only_wrong_preds = False
            if save_only_wrong_preds and preds_correct[1]:
                continue

            prediction_image = self.build_prediction_image(list_of_images_paths, preds_correct)
            pred_image_path = visual_dir + f"/{query_index:03d}.jpg"
            prediction_image.save(pred_image_path)
        


    def calculate_recall_at_k(self, query_features, query_labels, database_features, database_labels, k=[1, 5, 10]): # k here need to be specific for visualization
        if self.faiss_gpu:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatL2(res, database_features.shape[1], flat_config)
        else:
            index = faiss.IndexFlatL2(database_features.shape[1])
        logger.debug(f"shape of database_features:{database_features.shape}, num of database_labels:{len(database_labels)}")
        logger.debug(f"shape of query_features:{query_features.shape}, num of query_labels:{len(query_labels)}")
        
        index.add(database_features)
        distances, indices = index.search(query_features, k=5) # indices: indices of nn(to query_i, top_k) in database

        recalls = []
        # logger.debug(f"length of indices:{len(indices)}, length of query_labels:{len(query_labels)}.")
        for threshold in [1, 5, 10]:
            correct = 0
            for i, query_label in enumerate(query_labels):
                if query_label in database_labels[indices[i, :threshold]]:  # indices[i, :threshold] means the top k(threshold) nearest neighbors of query i
                    correct += 1  # as long as query is retrieved in top k results, regard it as a successful retrieval(refer to definition of Recall@K)
            recall = correct / len(query_labels) # eq. len(query_features[0]) & len(indices)
            recalls.append(recall)
            logger.debug(f'Recall@{threshold}: {recall:.4f}')

        self.record_matches(indices, out_file='./LOGS/record.txt')
        self.visualize(indices, query_labels, database_labels, visual_dir='./LOGS/visualize')
        return recalls
    
    def extract_features(self, dataloader):
        features = []
        labels = []
        for images, lbls in dataloader: # lbls here refer to a batch of labels, and will not overwritten list labels(list to reserve all labels) above
            images = images.to(self.device)
            descriptors = self.model(images) # model here seems it's still on device cuda
            features.append(descriptors.cpu().numpy())
            labels.append(lbls.numpy())
        features = np.vstack(features)
        labels = np.hstack(labels) # each label can be reserved in a list and constitute a group of labels
        # logger.debug(f"labels:{labels}")
        logger.debug(f"shape of labels:{labels.shape}")
        return features, labels

    def run(self):
        logger.info(f"Evaluating VPR model: {self.model_name}")

        k = [1, 5, 10]

        self.model.eval() # still need to remove some processes only for training
        with torch.inference_mode():
            logger.debug(f"Calculating descriptors of query...")
            query_loader = tqdm(self.query_loader, desc="Query", unit="batch")
            query_features, query_labels = self.extract_features(query_loader)

            logger.debug(f"Calculating descriptors of database...")
            db_loader = tqdm(self.db_loader, desc="Database", unit="batch")
            db_features, db_labels = self.extract_features(db_loader)

            recalls = self.calculate_recall_at_k(query_features, query_labels, db_features, db_labels, k)
            # query_loader.set_postfix(recall=recalls[1])
        return recalls


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--resume', action='store_true', help='resume training from checkpoint')
    parser.add_argument('--ddp', action='store_true', help='whether to train the model on multiple GPUs')
    parser.add_argument('--method', required=True, help='choose the method you\'d like to evaluate' )
    args = parser.parse_args()

    # Load dataset
    logger.info("Creating dataloader...")
    # root_dir = 'path/to/output/folder'  # Update this to your output folder in TALAData.py
    loader = TALADataloader(ddp=args.ddp) # there are transformations for val & train data in TALADataloader respectively

    query_loader, db_loader = loader.val_dataloader()
    query_path = loader.query_image_path
    db_path = loader.db_image_path
    
    # evaluator = VPREvaluator(model_name='mixvpr', 
    #                     db_loader=db_loader, 
    #                    query_loader=query_loader,
    #                    query_path = query_path,
    #                    db_path = db_path,
    #                     recall_k=5, 
    #                     batch_size=32, 
    #                     device='cuda')

    method = args.method
    evaluator = VPREvaluator(model_name=method, 
                    db_loader=db_loader, 
                    query_loader=query_loader,
                    query_path = query_path,
                    db_path = db_path,
                    recall_k=5, 
                    batch_size=32, 
                    device='cuda')
    
    recall_at_k = evaluator.run()
