import os
import numpy as np
import pickle
import pandas as pd
import joblib
import tensorflow as tf
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, Input
from Utlis import extract_image_features,precompute_dominant_colors  # Ensure this extracts 1280-dim features
from sklearn.preprocessing import LabelEncoder

# Define paths
dataset_csv = "archive/women/women.csv"
image_folder = "archive/women/women_images"
unlabeled_images_folder = "archive/women/women_unlabel_images"
graphs_folder = "graphs"
model_folder = "model_details"
os.makedirs(graphs_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)

#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################

"""...... Load labeled data and extract only EfficientNetB0 features ......"""




print("Extracting labeled image features...")
df = pd.read_csv(dataset_csv)

image_paths = [os.path.join(image_folder, f"{row['id']}.jpg") for _, row in df.iterrows()]
X = np.array([extract_image_features(img_path) for img_path in image_paths])  # 1280-dim

# Encode categorical labels
usage_encoder = LabelEncoder()
subcategory_encoder = LabelEncoder()

df['usage'] = usage_encoder.fit_transform(df['usage'])  
df['subCategory'] = subcategory_encoder.fit_transform(df['subCategory'])  

# Extract labels
y_usage = df['usage'].values
y_subcategory = df['subCategory'].values

# Save encoders for later inference
joblib.dump(usage_encoder, os.path.join(model_folder, "usage_encoder.pkl"))
joblib.dump(subcategory_encoder, os.path.join(model_folder, "subcategory_encoder.pkl"))

#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################

"""...... Precompute Dominant Colors/Usage/Subcategory ......"""

dominant_colors, valid_image_paths,subCategory, articleType, usage = precompute_dominant_colors(df,image_folder)

# Save to pickle file
dominant_colors_data = {
    "image_paths": valid_image_paths,
    "dominant_colors": dominant_colors,
    "subcategory_labels": subCategory,
    "usage_labels": usage,
    "articleType_labels": articleType
}

pkl_path = os.path.join(model_folder, "dominant_colors.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(dominant_colors_data, f)

print(f"Saved dominant colors for {len(valid_image_paths)} images to dominant_colors.pkl")

#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################



"""...... Split data and train models ......"""

# Split into train and test sets
X_train, X_test, y_train_usage, y_test_usage = train_test_split(X, y_usage, test_size=0.2, random_state=42)
X_train, X_test, y_train_sub, y_test_sub = train_test_split(X, y_subcategory, test_size=0.2, random_state=42)

# Train Usage Prediction Model (MLP)
print("Training Usage Prediction Model...")

usage_model = Sequential([
    Input(shape=(1280,)),  # Fixed feature size from EfficientNetB0
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(set(y_usage)), activation='softmax')  # Multi-class classification
])

usage_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

history = usage_model.fit(X_train, y_train_usage, epochs=20, batch_size=32, validation_data=(X_test, y_test_usage))

# Save the trained model
usage_model.save(os.path.join(model_folder, "usage_model.keras"))

print("Usage Prediction Model trained and saved!")

#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################

"""...... Train Subcategory Prediction Model (MLP) ......"""

# Train Subcategory Prediction Model (MLP)
print("Training Subcategory Prediction Model...")
subcategory_model = Sequential([
    Input(shape=(1280,)),  # Fixed feature size
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(set(y_subcategory)), activation='softmax')
])

subcategory_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = subcategory_model.fit(X_train, y_train_sub, epochs=20, batch_size=32, validation_data=(X_test, y_test_sub))
subcategory_model.save(os.path.join(model_folder, "subcategory_model.keras"))

#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################
"""...... Self-Training on Unlabeled Images ......"""

# Self-Training on Unlabeled Images
print("Performing self-training with unlabeled images...")
unlabeled_features = []
image_paths = []

for img_file in os.listdir(unlabeled_images_folder):
    img_path = os.path.join(unlabeled_images_folder, img_file)
    image_paths.append(img_path)
    unlabeled_features.append(extract_image_features(img_path))  # Ensure 1280-dim

unlabeled_features = np.array(unlabeled_features)

# Predict pseudo-labels
pseudo_usage_labels = np.argmax(usage_model.predict(unlabeled_features), axis=1)
pseudo_sub_labels = np.argmax(subcategory_model.predict(unlabeled_features), axis=1)

# Combine pseudo-labeled data with real training data
X_train_combined = np.vstack((X_train, unlabeled_features))
y_train_usage_combined = np.concatenate((y_train_usage, pseudo_usage_labels))
y_train_sub_combined = np.concatenate((y_train_sub, pseudo_sub_labels))

# Retrain models with pseudo-labeled data
print("Retraining Usage Model with Pseudo-Labels...")
usage_model.fit(X_train_combined, y_train_usage_combined)
joblib.dump(usage_model, os.path.join(model_folder, "usage_model.pkl"))

print("Retraining Subcategory Model with Pseudo-Labels...")
history = subcategory_model.fit(X_train_combined, y_train_sub_combined, epochs=10, batch_size=32, validation_data=(X_test, y_test_sub))
subcategory_model.save(os.path.join(model_folder, "subcategory_model.keras"))


#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################

"""...... Evaluation ......"""

# Evaluation
print("Evaluating Model...")

# Predict
y_pred_numeric = np.argmax(subcategory_model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test_sub, y_pred_numeric)
conf_matrix = confusion_matrix(y_test_sub, y_pred_numeric)
report = classification_report(y_test_sub, y_pred_numeric)

# Save Evaluation Results
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.savefig(os.path.join(graphs_folder, 'accuracy_plot.png'))
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.savefig(os.path.join(graphs_folder, 'loss_plot.png'))
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(graphs_folder, 'confusion_matrix.png'))
plt.close()

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)

print("Training complete! Models and evaluation results saved.")
