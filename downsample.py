import os
import shutil
import cv2

def downsample_images(input_folders, output_folder, ratio, crop_settings, req_crop):
    """
    Downsamples the number of images in the input_folders by the given ratio and copies them to the output_folder.
    Keeps images at regular intervals rather than randomly.

    :param input_folders: List of paths to input folders containing images
    :param output_folder: Path to the output folder to save downsampled images
    :param ratio: Downsampling ratio (e.g., 0.5 to keep every second image)
    :param crop_settings: Dictionary with folder names as keys and crop settings as values
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for input_folder in input_folders:
        # Get the name of the current input folder
        folder_name = os.path.basename(os.path.normpath(input_folder))
        # Create the corresponding output subfolder
        output_subfolder = os.path.join(output_folder, folder_name)

        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        # Collect all image paths in the current input folder
        image_files = []
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                    image_files.append(os.path.join(root, file))

        # Sort the image files to ensure consistent ordering
        image_files.sort()

        # Determine the step size
        step_size = int(1 / ratio)

        # Select images at regular intervals
        selected_images = image_files[::step_size]

        for image_path in selected_images:
            # Get the filename of the image
            file_name = os.path.basename(image_path)
            # Construct the output path by joining the output subfolder and the filename
            output_path = os.path.join(output_subfolder, file_name)

            # resize and save images
            img = cv2.imread(image_path)

            # Crop the image according to the settings for the folder, 
            if req_crop:
                if crop_settings and input_folder in crop_settings:
                    crop_size, crop_position = crop_settings[input_folder]
                    img = crop_image(img, crop_size, crop_position)
                else:
                    print("*------------Crop failed------------*")

            # img = imutils.resize(img, width=400)
            # img = imutils.resize(img, height=400)  # cancel resize above won't change the size of output but improve resolution
            img = cv2.resize(img, (1024, 256), interpolation=cv2.INTER_CUBIC) # (width,height) # in this case, cv2.resize(img, (711,400)) works almost the same as imutils.resize(img, height=400)

            cv2.imwrite(output_path, img)

            # Copy the selected image to the output folder
            # shutil.copy(image_path, output_path)
            print(f"Copied and resized {image_path} to {output_path}")


def crop_image(image, crop_size, position):
    """
    Crops the image according to the specified size and position.

    :param image: Input image to be cropped
    :param crop_size: Tuple indicating the size to crop images to (width, height)
    :param position: String indicating the position to crop from ('center', 'bottom', 'left', 'right')
    :return: Cropped image
    """
    img_height, img_width = image.shape[:2]
    crop_width, crop_height = crop_size

    if position == 'center':
        start_x = (img_width - crop_width) // 2
        start_y = (img_height - crop_height) // 2
    elif position == 'bottom':
        start_x = 0
        end_x =  img_width
        start_y = 0 + 10 # to remove green mask in the image
        end_y = img_height - crop_height
    elif position == 'left':
        start_x = crop_width
        end_x = img_width
        start_y = 0 + 10
        end_y = img_height - crop_height
    elif position == 'right':
        start_x = 0
        end_x = img_width - crop_width
        start_y = 0 + 10
        # end_y = img_height - 10 # to remove green mask from the bottom
        end_y = img_height - crop_height # to make height of 3 images identical
    else:
        raise ValueError(f"Invalid crop position: {position}")

    cropped_image = image[start_y:end_y, start_x:end_x] # identical with pixel ordination
    return cropped_image


if __name__ == "__main__":
    # directory = "/home/SENSETIME/jiangyintian/TALA/Collection/L2/2024_06_19_18_08_52_AutoCollect/parsed_data/"
    # directory = "/home/SENSETIME/jiangyintian/TALA/Collection/UNI/uni_1/parsed_data/"
    directory = "E:/TALA/Sensetime/TALA/lingang/data2/parsed_data/downsample/"
    input_folders = [
        # directory + "center_camera_fov120",
        # directory + "left_front_camera",
        # directory + "right_front_camera"
        # directory + "front_camera_fov195",
        # directory + "right_camera_fov195",
        # directory + "left_camera_fov195",
        directory + "panorama"
        # Add more input folders as needed
    ]
    output_folder = directory + "benchmark"
    ratio = 1  # Set your desired downsampling ratio here (e.g., 0.5 to keep every second image)

    # Crop settings for each folder
    req_crop = False
    # this one suits vehicle A02-932, crop three images to the same size to see whether better results can be accquired.
    crop_settings = {
        input_folders[0]: ((0, 80), 'bottom'),  # Crop size (width, height) and position(size to remove rather than keeping) # 80 for CN cars and 150 for A02 cars
        # input_folders[1]: ((63, 150), 'right'),  # orginally crop size of 'left' & 'right' was (63,0), now set to (63,150)
        # input_folders[2]: ((85, 150), 'left'), # to make height of 3 images identical.
    }

    downsample_images(input_folders, output_folder, ratio, crop_settings, req_crop)
