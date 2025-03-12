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
    valid_image_ids = set()  # To store the image IDs that exist in the folder
    for filename in os.listdir(image_folder):
        file_path = os.path.join(image_folder, filename)
        
        # Extract ID (assuming images are named as <ID>.jpg, <ID>.png, etc.)
        file_id, _ = os.path.splitext(filename)
        
        if file_id in valid_ids:
            valid_image_ids.add(file_id)
        else:
            os.remove(file_path)
            print(f"Deleted: {filename}")
    
    # Remove rows from the DataFrame if the image is missing
    df_cleaned = df[df[id_col].astype(str).isin(valid_image_ids)]
    
    # Save the cleaned DataFrame back to CSV
    df_cleaned.to_csv(csv_file, index=False)
    
    print("Image cleanup and CSV update complete. Deleted missing images and their rows in the CSV.")

# Example usage
image_folder = "archive/men/men_images"  # Change this to your image folder path
csv_file = "archive/men/men.csv"  # Change this to your CSV file path
id_col = "id"  # Change this to the column name containing the image IDs

clean_images(image_folder, csv_file, id_col)
