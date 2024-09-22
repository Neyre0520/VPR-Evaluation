import os
import logging
from termcolor import colored

class ImageRenamer:
    def __init__(self, base_directory):
        self.base_directory = base_directory
        self.logger = self.setup_logger()

    def setup_logger(self):
        # Setting up the logger with colored output
        logger = logging.getLogger("ImageRenamer")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def log_colored(self, message, level='info', color='white'):
        colored_message = colored(message, color)
        if level == 'info':
            self.logger.info(colored_message)
        elif level == 'warning':
            self.logger.warning(colored_message)
        elif level == 'error':
            self.logger.error(colored_message)

    def rename_images_in_folder(self, folder):
        area_name = os.path.basename(folder)
        images = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        images.sort()

        self.log_colored(f"Renaming images in folder: {area_name}", color='blue')

        for i, img in enumerate(images):
            place_index = (i // 7) + 1
            timestamp, ext = os.path.splitext(img)
            # timestamp = timestamp.split("_")[-1]
            new_name = f"{area_name}_{place_index}_{timestamp}{ext}"
            old_path = os.path.join(folder, img)
            new_path = os.path.join(folder, new_name)

            try:
                os.rename(old_path, new_path)
                self.log_colored(f"Renamed: {old_path} -> {new_path}", color='green')
            except Exception as e:
                self.log_colored(f"Failed to rename {old_path}: {e}", level='error', color='red')

    def process_folders(self):
        for folder in os.listdir(self.base_directory):
            folder_path = os.path.join(self.base_directory, folder)
            if os.path.isdir(folder_path):
                self.rename_images_in_folder(folder_path)
            else:
                self.log_colored(f"{folder} is not a directory", level='warning', color='yellow')

if __name__ == "__main__":
    base_directory = "E:/TALA/Sensetime/benchmark_data/pano"  # Replace with your base directory path
    renamer = ImageRenamer(base_directory)
    renamer.process_folders()
