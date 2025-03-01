import os

# Define the path to your data directory
data_dir = 'data/'  # Path to the directory containing your images

# Get a list of all jpg files in the directory
image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]

# Sort files to ensure they are renamed sequentially
image_files.sort()

# Loop through the image files and rename them
for i, img_file in enumerate(image_files, start=1):
    # Get the full path of the image
    old_path = os.path.join(data_dir, img_file)
    
    # Define the new filename
    new_filename = f"{i}.jpg"
    new_path = os.path.join(data_dir, new_filename)
    
    # Rename the image file
    os.rename(old_path, new_path)
    print(f"Renamed {img_file} to {new_filename}")
