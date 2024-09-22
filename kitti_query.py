import os
import shutil

# Function to rename images, keeping the secondary index (e.g., _01, _02) if present
def rename_images(source_folder, destination_folder, area_name, start_index=0):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get all files in the source folder
    image_files = sorted([f for f in os.listdir(source_folder) if f.endswith('.png')])

    for index, filename in enumerate(image_files):
        # Extract the suffix (_01, _02) if it exists
        base_name, ext = os.path.splitext(filename)
        if "_" in base_name:
            place_index, sub_index = base_name.split("_")
            new_filename = f"{area_name}_{start_index + index:06}_{sub_index}_right{ext}" # add "left" or "right"
        else:
            new_filename = f"{area_name}_{start_index + index:06}_right{ext}"
        
        # Full paths for source and destination
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, new_filename)

        # Copy and rename the file to the new destination folder
        shutil.copy2(source_path, destination_path)

        print(f"Renamed {filename} to {new_filename} and saved in {destination_folder}")

# Get the total number of images in the first folder
def count_images_in_folder(folder):
    return len([f for f in os.listdir(folder) if f.endswith('.png')])

# Set your folder paths and area name
first_folder = 'E:/Download/data_object_image_3/training'  # First folder path
second_folder = 'E:/Download/data_object_image_3/testing'  # Second folder path
other_folder_1 = 'E:/Download/data_object_prev_2/training'  # First additional folder path
other_folder_2 = 'E:/Download/data_object_prev_2/testing'  # Second additional folder path
destination_folder_1 = 'E:/Download/data_object_image_3/query'  # Destination for first folder
destination_folder_2 = 'E:/Download/data_object_prev_2/database'  # Destination for second folder
area_name = 'kitti'  # Area name to prepend

# Get the total count of images from the first folder (this will be the start index for the second folder)
start_index_for_second = count_images_in_folder(first_folder)

# Rename images in the first folder
rename_images(first_folder, destination_folder_1, area_name)

# Rename images in the second folder, starting from the next index
rename_images(second_folder, destination_folder_1, area_name, start_index=start_index_for_second)

