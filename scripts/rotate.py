import os
from PIL import Image

def rotate_folder_images(folder_path):
    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    
    # Create an output folder to avoid overwriting originals
    output_folder = os.path.join(folder_path, 'rotated_90_ccw')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(folder_path, filename)
            try:
                with Image.open(img_path) as img:
                    # rotate() takes degrees CCW; expand=True ensures the canvas 
                    # resizes to fit the new orientation
                    rotated_img = img.rotate(90, expand=True)
                    
                    rotated_img.save(os.path.join(output_folder, filename))
                    print(f"Successfully rotated: {filename}")
            except Exception as e:
                print(f"Could not process {filename}: {e}")

# Replace with the path to your folder
folder_to_process = '../data/extracted_frames/bell-bottom'
rotate_folder_images(folder_to_process)
