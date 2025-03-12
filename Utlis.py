import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.api.applications import EfficientNetB0 
from keras.api.applications.efficientnet import preprocess_input
from keras.api.preprocessing import image
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import KDTree
from joblib import Parallel, delayed
from sklearn.neighbors import KDTree
import streamlit as st
import colorsys
import random
import pickle
import joblib
import os
import colorsys
import cv2
from skimage import color

#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################

"""...  FEATURE EXTRACTION/TRAIN/TEST FUNCTIONS  ..."""


# Load EfficientNetB0 model for feature extraction
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model = tf.keras.Model(inputs=base_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output))

def load_and_preprocess_image(img_path):
    """Load an image and preprocess it for EfficientNetB0."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def extract_image_features(img_input):
    """Extract EfficientNetB0 features from an image file path or array."""
    if isinstance(img_input, str):  # If input is a file path
        img_array = load_and_preprocess_image(img_input)
    else:  # If input is an image array (OpenCV format)
        img = cv2.resize(img_input, (224, 224))
        img = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img)

    features = model.predict(img_array)
    return features.flatten()

import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_dominant_color(img_input, k=3):
    """Fast & optimized dominant color extraction focusing on the clothing item."""
    
    # Load image
    if isinstance(img_input, str):  # If it's a file path
        img = cv2.imread(img_input)
    else:  # If it's an image array
        img = img_input.copy()

    if img is None:
        raise ValueError("Invalid image input. Check file path or image array.")

    # Resize to speed up processing (optional, adjust based on your system)
    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------------------- 1️⃣ FAST CLOTHING DETECTION ----------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding for better edge detection
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)

    # Morphology to reduce noise (only 1 iteration for speed)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find largest contour (clothing area)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return (0, 0, 0)  # Return black if no contour found

    # Get the largest contour & crop around it
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Ensure valid size
    if w < 40 or h < 40:  
        return (0, 0, 0)
    
    # Crop to the detected clothing region
    cropped_img = img_rgb[y:y+h, x:x+w]

    # ---------------------- 2️⃣ FAST COLOR EXTRACTION ----------------------
    pixels = cropped_img.reshape(-1, 3)

    # Remove near-black background pixels (faster than DBSCAN)
    pixels = pixels[np.any(pixels > [20, 20, 20], axis=1)]  

    if len(pixels) < 50:  
        return (0, 0, 0)

    # K-Means for dominant color detection
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=5)
    kmeans.fit(pixels)

    # Get the most frequent cluster
    cluster_labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_cluster = cluster_labels[np.argmax(counts)]
    dominant_color = kmeans.cluster_centers_[dominant_cluster]

    return tuple(map(int, dominant_color))  # Convert to integer RGB values

def load_metadata(csv_path):
    """Load metadata from CSV file and encode categorical variables."""
    df = pd.read_csv(csv_path)
    categorical_columns = ['gender', 'subCategory', 'articleType', 'baseColour', 'usage']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(df[categorical_columns])
    return df['id'].values, encoded_features

def prepare_training_data(image_folder, csv_path):
    """Prepare training data by combining image features and metadata."""
    ids, metadata_features = load_metadata(csv_path)
    image_features = []
    
    for img_id in ids:
        img_path = os.path.join(image_folder, str(img_id) + ".jpg")
        if os.path.exists(img_path):
            img_encodings = extract_image_features(img_path)
            image_features.append(img_encodings)
        else:
            image_features.append(np.zeros(1280))  # Placeholder if image is missing
    
    image_features = np.array(image_features)
    combined_features = np.hstack((image_features, metadata_features))
    return combined_features



def prepare_single_image(image_array):
    """Prepares a single image (OpenCV format) for model prediction."""
    # Resize the image to match EfficientNetB0 input size
    image = cv2.resize(image_array, (224, 224))
    
    # Convert to float32 and normalize
    image = image.astype(np.float32) / 255.0
    
    # Expand dimensions to match batch format (1, 224, 224, 3)
    image = np.expand_dims(image, axis=0)
    
    # Preprocess for EfficientNetB0
    image = preprocess_input(image)

    # Extract deep features
    features = extract_image_features(image)  # This should return a (1, N) feature vector

    return features

#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################

"""...  SIMILAR ITEM RECOMMENDATION FUNCTIONS  ..."""


  # Ensure this function is optimized


def precompute_dominant_colors(df, image_folder):
    """Precompute dominant colors and subcategories for all images and store them in a pickle file."""
    dominant_colors = []
    valid_image_paths = []
    subcategories = []
    artcleTypes = []
    usage = []


    for _, row in df.iterrows():
        image_path = os.path.join(image_folder, f"{row['id']}.jpg")  # Construct full image path
        subcategory = row.get("subCategory", None)
        articleType = row.get("articleType", None)
        usage_ = row.get("usage", None)

        # Skip missing paths or invalid files
        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue
        
        try:
            # Extract dominant color
            color = extract_dominant_color(image)
            dominant_colors.append(color)
            valid_image_paths.append(image_path)
            subcategories.append(subcategory)
            artcleTypes.append(articleType)
            usage.append(usage_)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return np.array(dominant_colors), valid_image_paths, subcategories, artcleTypes, usage

def extract_color_features(image_path):
    """Extracts a color feature vector from an image after resizing."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (50, 50))  # Reduce image size for faster processing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV for better color representation
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()  # Flatten to 1D feature vector

# Cache the dataset loading to avoid reloading on every interaction
 # Ensure correct import

CACHE_FILE = "cache_data/cache_men_features.pkl"  # File to store precomputed dataset

@st.cache_data(ttl=0)  # Keeps data cached during the session
def load_dataset(csv_path, image_folder):
    """Loads dataset and precomputes color features efficiently."""

    # ✅ 1️⃣ Check if precomputed dataset exists (avoid recomputation)
    if os.path.exists(CACHE_FILE):
        return joblib.load(CACHE_FILE)

    df = pd.read_csv(csv_path)
    df['image_path'] = df['id'].apply(lambda x: os.path.join(image_folder, f"{x}.jpg"))
    df = df[df['image_path'].apply(os.path.exists)]

    # ✅ 2️⃣ Extract color features in parallel (Fixes caching issue)
    df['color_features'] = Parallel(n_jobs=-1)(
        delayed(extract_dominant_color)(path) for path in df['image_path']
    )

    # ✅ 3️⃣ Save precomputed dataset for fast loading
    joblib.dump(df, CACHE_FILE)

    return df


def load_precomputed_colors(pkl_path):
    """Load precomputed dominant colors and categories from the pickle file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    return np.array(data["dominant_colors"]), data["image_paths"], data["subcategory_labels"] , data["articleType_labels"] , data["usage_labels"]  


#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################

"""...  SIMILAR FUNCTIONS  ..."""


def find_similar_colors(image, pkl_path, selected_category, selected_usages, min_similar=5, threshold=100):
    """Finds the top 5 most similar items based on color, considering all images in the selected category."""
    dominant_colors, image_paths, categories, article_types, usages = load_precomputed_colors(pkl_path)

    # Extract dominant color from the uploaded image
    target_color = np.array(extract_dominant_color(image))

    # Filter images by selected category
    filtered_indices = [
        i for i, cat in enumerate(categories) if cat == selected_category
    ]

    if len(filtered_indices) == 0:
        return []  # No images in the selected category

    # Further filter by selected usage types
    if selected_usages:  # Only apply if the user selected any usage types
        filtered_indices = [
            i for i in filtered_indices if int(usages[i]) in selected_usages
        ]

    if len(filtered_indices) == 0:
        return []  # No images match the selected usage types

    # Get the colors, paths, article types, and usage of filtered images
    filtered_colors = np.array([dominant_colors[i] for i in filtered_indices])
    filtered_paths = [image_paths[i] for i in filtered_indices]
    filtered_article_types = [article_types[i] for i in filtered_indices]
    filtered_usages = [usages[i] for i in filtered_indices]

    # Compute color distances (Euclidean distance in RGB space)
    distances = np.linalg.norm(filtered_colors - target_color, axis=1)

    # Sort by closest color match
    sorted_indices = np.argsort(distances)
    
    similar_images = [filtered_paths[i] for i in sorted_indices]
    similar_article_types = [filtered_article_types[i] for i in sorted_indices]
    similar_usages = [filtered_usages[i] for i in sorted_indices]
    similar_colors = [filtered_colors[i] for i in sorted_indices]

    # Filter images within the threshold
    valid_indices = [i for i in range(len(similar_images)) if distances[sorted_indices[i]] < threshold]

    if len(valid_indices) < min_similar:
        valid_indices = range(min_similar)  # Ensure at least min_similar results

    # Use session state to cycle through suggestions
    if "suggestion_index" not in st.session_state:
        st.session_state.suggestion_index = 0  # Initialize index

    if st.button("Refresh"):
        i = st.session_state.suggestion_index
        st.session_state.suggestion_index = i + 1

    # Get the next 5 suggestions
    start_idx = st.session_state.suggestion_index
    end_idx = start_idx + 5
    displayed_indices = valid_indices[start_idx:end_idx]

    # Update index for next refresh
    st.session_state.suggestion_index = (start_idx + 5) % len(valid_indices)  # Loop back when reaching the end

    # Return images with metadata
    return [
        {
            "path": similar_images[i],
            "article_type": similar_article_types[i],
            "usage": similar_usages[i],
            "color": tuple(map(int, similar_colors[i]))  # Convert numpy array to tuple
        }
        for i in displayed_indices
    ]

#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################
 

def recommend_outfit(image, pkl_path, selected_category, selected_usage, color_theory="Complementary", min_similar=5, threshold=150):
    """
    Recommends a matching Topwear or Bottomwear based on the uploaded image using color theory.
    """
    try:
        # Load precomputed color data
        dominant_colors, image_paths, categories, article_types, usages = load_precomputed_colors(pkl_path)

        # Extract dominant color from the uploaded image
        target_color = np.array(extract_dominant_color(image))
        target_lab = color.rgb2lab(target_color.reshape(1, 1, 3) / 255)[0, 0]  # Convert to LAB
        st.write(f"🎨 Extracted Color (RGB): {target_color}")
        st.write(f"🎨 Extracted Color (LAB): {target_lab}")

        # Filter items belonging to the selected category and usage
        filtered_indices = [i for i, cat in enumerate(categories) if cat == int(selected_category) and usages[i] == int(selected_usage)]

        if not filtered_indices:
            st.warning("⚠️ No matching items found for the selected category and usage.")
            return []

        # Extract relevant data for the filtered items
        filtered_colors = np.array([dominant_colors[i] for i in filtered_indices])
        filtered_paths = [image_paths[i] for i in filtered_indices]
        filtered_article_types = [article_types[i] for i in filtered_indices]

        # Get the matching color based on selected color theory approach
        match_color = np.array(get_color_scheme(target_color, color_theory))
        match_lab = color.rgb2lab(match_color.reshape(1, 1, 3) / 255)[0, 0]  # Convert to LAB
        st.write(f"🎨 Matching Color using {color_theory} (RGB): {match_color}")
        st.write(f"🎨 Matching Color (LAB): {match_lab}")

        # Convert filtered colors to LAB
        filtered_lab_colors = np.array([color.rgb2lab(c.reshape(1, 1, 3) / 255)[0, 0] for c in filtered_colors])

        # Compute Delta E distances
        delta_e_distances = np.array([np.linalg.norm(match_lab - lab) for lab in filtered_lab_colors])

        # Sort items by closest Delta E match
        sorted_indices = np.argsort(delta_e_distances)[:10]  # Get top 10 closest matches
        similar_images = [filtered_paths[i] for i in sorted_indices]
        similar_article_types = [filtered_article_types[i] for i in sorted_indices]
        similar_colors = [filtered_colors[i] for i in sorted_indices]

        # Return matched outfit items
        return [
            {
                "path": similar_images[i],
                "article_type": similar_article_types[i],
                "color": tuple(map(int, similar_colors[i]))
            }
            for i in range(len(sorted_indices))
        ]

    except Exception as e:
        st.error(f"🚨 Error in recommend_outfit: {e}")
        return []

def get_color_scheme(color, approach):
    """Generate a matching color based on the selected approach."""
    r, g, b = color
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)

    if approach == "Complementary":
        h = (h + 0.5) % 1  # Opposite hue
    elif approach == "Analogous":
        h = (h + 0.083) % 1  # Slightly shifted hue
    elif approach == "Triadic":
        h = (h + 1/3) % 1  # 120-degree shift
    elif approach == "Monochromatic":
        s = max(0.3, min(1, s * 1.2))  # Adjust saturation

    new_color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))
    return new_color


