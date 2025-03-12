import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from Utlis import extract_hybrid_features, hybrid_knn_search
from keras._tf_keras.keras.applications.efficientnet import EfficientNetB0
from keras.api.layers import GlobalMaxPooling2D
from keras.api.models import Sequential


# ✅ Define the model directory
load_dir = "model_details"

# ✅ Load trained data
features_list = pickle.load(open(os.path.join(load_dir, "image_features_embedding.pkl"), "rb"))
img_files_list = pickle.load(open(os.path.join(load_dir, "img_files.pkl"), "rb"))
kd_tree = pickle.load(open(os.path.join(load_dir, "kd_tree.pkl"), "rb"))
ball_tree = pickle.load(open(os.path.join(load_dir, "ball_tree.pkl"), "rb"))

print("✅ Model loaded successfully from 'model_details' folder.")

# ✅ Load EfficientNetB0 model
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = Sequential([base_model, GlobalMaxPooling2D()])

# ✅ Input image path
img_path = 'archive/images/1529.jpg'

# ✅ Show the query image before extracting features
query_img = cv2.imread(img_path)
if query_img is None:
    raise ValueError(f"Error: Could not load image at {img_path}")
query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(5, 5))
plt.imshow(query_img)
plt.title("Query Image")
plt.axis("off")
plt.show()

# ✅ Extract features and get recommendations
features = extract_hybrid_features(img_path, model)
indices = hybrid_knn_search(features, features_list, k=6)  # ✅ Fetching top 6 recommendations

# ✅ Function to display images
def show_images(query_img_path, recommended_indices, img_files_list):
    plt.figure(figsize=(12, 6))
    
    # ✅ Display query image
    query_img = cv2.imread(query_img_path)
    if query_img is None:
        raise ValueError(f"Error: Could not load image at {query_img_path}")
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 4, 1)
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis("off")

    # ✅ Display recommended images
    for i, idx in enumerate(recommended_indices):
        recommended_img = cv2.imread(img_files_list[idx])
        if recommended_img is None:
            continue
        recommended_img = cv2.cvtColor(recommended_img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 4, i + 2)
        plt.imshow(recommended_img)
        plt.title(f"Recommendation {i+1}")
        plt.axis("off")

    plt.show()

# ✅ Show recommendations
show_images(img_path, indices, img_files_list)
