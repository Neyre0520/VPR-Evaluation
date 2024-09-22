import os
import shutil
from collections import defaultdict
import logging
import coloredlogs

# Set up logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s - %(levelname)s - %(message)s')

class ImageDivider:
    def __init__(self, main_directory, query_directory, database_directory):
        self.main_directory = main_directory
        self.query_directory = query_directory
        self.database_directory = database_directory
        self.setup_directories()

    def setup_directories(self):
        os.makedirs(self.query_directory, exist_ok=True)
        os.makedirs(self.database_directory, exist_ok=True)

    def divide_images(self):
        logger.info("Starting image division process.")
        
        for area_folder in os.listdir(self.main_directory):
            area_path = os.path.join(self.main_directory, area_folder)
            
            if os.path.isdir(area_path):
                logger.info(f"Processing area: {area_folder}")
                os.makedirs(os.path.join(self.query_directory, area_folder), exist_ok=True)
                os.makedirs(os.path.join(self.database_directory, area_folder), exist_ok=True)

                images_by_place_id = defaultdict(list)
                
                for image in os.listdir(area_path):
                    if image.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
                        place_id = image.split('_')[1]
                        images_by_place_id[place_id].append(image)
                
                for place_id, images in images_by_place_id.items():
                    if len(images) >= 3:
                        logger.info(f"Place ID {place_id} has {len(images)} images. Moving one to the query.")
                        shutil.copy(os.path.join(area_path, images[0]), os.path.join(self.query_directory, area_folder, images[0]))
                        shutil.copy(os.path.join(area_path, images[1]), os.path.join(self.query_directory, area_folder, images[1]))
                        # shutil.copy(os.path.join(area_path, images[1]), os.path.join(self.database_directory, area_folder, images[1])) # for 1:1 query&db
                        for image in images[2:]:  # for your "script"
                            logger.info(f"Moving image {image} to database.")
                            shutil.copy(os.path.join(area_path, image), os.path.join(self.database_directory, area_folder, image))
                    else:
                        logger.info(f"Place ID {place_id} has {len(images)} image. Moving it to database.")
                        for image in images:
                            shutil.copy(os.path.join(area_path, image), os.path.join(self.database_directory, area_folder, image))

        logger.info("Image division process completed.")

# Usage example
if __name__ == '__main__':
    # main_directory = '/home/SENSETIME/jiangyintian/TALA/Collection/TALADataset/sample/reg/sf_val'
    # query_directory = '/home/SENSETIME/jiangyintian/TALA/Collection/TALADataset/sample/reg/query'
    # database_directory = '/home/SENSETIME/jiangyintian/TALA/Collection/TALADataset/sample/reg/database'
    main_directory = 'E:/TALA/Sensetime/benchmark_data/pano/'
    query_directory = 'E:/TALA/Sensetime/benchmark_data/query/'
    database_directory = 'E:/TALA/Sensetime/benchmark_data/database' # copy query and database to sf_val manually

    image_divider = ImageDivider(main_directory, query_directory, database_directory)
    image_divider.divide_images()
