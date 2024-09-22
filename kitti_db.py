import os
import shutil

def get_place_idx(image_name):
    """Extract place index from the image name in format 'placeidx_idx'."""
    return int(image_name.split('_')[0])

def rename_and_copy_images(src_folder, dst_folder, place_offset=0):
    """Rename images in the source folder and copy them to the destination folder."""
    # Ensure destination folder exists
    os.makedirs(dst_folder, exist_ok=True)
    
    for image_name in os.listdir(src_folder):
        if image_name.endswith('.png'):
            # Get place index and image index
            place_idx, img_idx = image_name.split('_')
            
            # Add place_offset to place index for the second folder
            new_place_idx = int(place_idx) + place_offset
            
            # Create new image name in 'kitti_placeidx_idx' format
            new_image_name = f'kitti_{new_place_idx:06d}_right_{img_idx}' # add "left" or "right"
            
            # Define source and destination paths
            src_path = os.path.join(src_folder, image_name)
            dst_path = os.path.join(dst_folder, new_image_name)
            
            # Copy the image to the new folder with the new name
            shutil.copy2(src_path, dst_path)
            print(f'Copied {image_name} to {new_image_name}')

def main(folder1, folder2, dest_folder):
    # Get the highest place index from folder1
    folder1_images = [img for img in os.listdir(folder1) if img.endswith('.png')]
    if folder1_images:
        last_place_idx = max(get_place_idx(img) for img in folder1_images)
    else:
        last_place_idx = 0
    
    # Rename and copy images from the first folder to the destination
    print(f'Copying and renaming images from {folder1} to {dest_folder}...')
    rename_and_copy_images(folder1, dest_folder)
    
    # Rename and copy images from the second folder to the destination with place index offset
    print(f'Copying and renaming images from {folder2} to {dest_folder}...')
    rename_and_copy_images(folder2, dest_folder, place_offset=last_place_idx + 1)

if __name__ == "__main__":
    folder1 = "E:/Download/data_object_prev_3/training"
    folder2 = "E:/Download/data_object_prev_3/testing"
    dest_folder = "E:/Download/data_object_prev_3/database"
    
    main(folder1, folder2, dest_folder)
