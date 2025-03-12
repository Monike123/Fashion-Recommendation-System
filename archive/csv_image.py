import os
import pandas as pd

def clean_images(image_folder, csv_file, id_col):
    # Load CSV file
    df = pd.read_csv(csv_file)
    
    # Ensure the column exists in the DataFrame
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found in CSV file.")
    
    # Convert column to string to match filenames
    valid_ids = set(df[id_col].astype(str))
    
    # Get the list of images in the folder
    for filename in os.listdir(image_folder):
        file_path = os.path.join(image_folder, filename)
        
        # Extract ID (assuming images are named as <ID>.jpg, <ID>.png, etc.)
        file_id, _ = os.path.splitext(filename)
        
        # Delete images not in the CSV
        if file_id not in valid_ids:
            os.remove(file_path)
            print(f"Deleted: {filename}")
    
    print("Image cleanup complete. Kept only images listed in the CSV.")

# Example usage
image_folder = "archive/men/men_images"  # Change this to your image folder path
csv_file = "archive/men/men.csv"  # Change this to your CSV file path
id_col = "id"  # Change this to the column name containing the image IDs

clean_images(image_folder, csv_file, id_col)
