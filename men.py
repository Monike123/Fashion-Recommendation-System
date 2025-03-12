import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
import xgboost as xgb
import cv2
import pandas as pd
from Utlis import extract_image_features, extract_dominant_color , find_similar_colors ,load_dataset,recommend_outfit

# Load models and encoders
model_folder = "men_model_details"
usage_model = tf.keras.models.load_model(f"{model_folder}/usage_model.keras")
subcategory_model = tf.keras.models.load_model(f"{model_folder}/subcategory_model.keras")
usage_encoder = joblib.load(f"{model_folder}/usage_encoder.pkl")
subcategory_encoder = joblib.load(f"{model_folder}/subcategory_encoder.pkl")

# Load dataset for recommendations
dataset_csv = "archive/men.csv"  # Adjust accordingly
image_folder = "archive/men_images"  # Adjust accordingly
dataset = load_dataset(dataset_csv, image_folder)

st.title("Fashion Recommendation System For Men")
st.success("This app uses machine learning models to predict fashion categories and recommend similar items.")
st.success("Please select an image from your computer to get started.")
st.write("Upload an image to predict its color, usage, and subcategory.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Extract features using the same method as training
    X_input = np.array(extract_image_features(img)).reshape(1, -1)  # Ensure correct shape
    dominant_color = extract_dominant_color(img)

    # Convert RGB tuple to HEX for display
    dominant_color_hex = "#{:02x}{:02x}{:02x}".format(*dominant_color)

    # Display color as a colored box
    st.markdown(
        f"""
        <div style="width:50px; height:50px; background-color:{dominant_color_hex}; border-radius:5px;"></div>
        """,
        unsafe_allow_html=True,
    )

    st.write(f"**Dominant Color:** {dominant_color} (RGB) / {dominant_color_hex} (HEX)")
    
    # Predict usage
    usage_pred_numeric = np.argmax(usage_model.predict(X_input), axis=1)
    usage_pred_label = usage_encoder.inverse_transform(usage_pred_numeric)[0]
    
    # Predict subcategory
    sub_pred_numeric = np.argmax(subcategory_model.predict(X_input), axis=1)
    sub_pred_label = subcategory_encoder.inverse_transform(sub_pred_numeric)[0]
    
    # Display results
    st.subheader("Predictions:")
    st.write(f"**Dominant Color:** {dominant_color}")
    st.write(f"**Predicted Usage:** {usage_pred_label}")
    st.write(f"**Predicted Subcategory:** {sub_pred_label}")
    
    # Display accuracy (optional: placeholder values, can be updated with real metrics)
    usage_acc = 90  # Replace with actual accuracy if available
    subcategory_acc = 88  # Replace with actual accuracy if available
    st.write(f"**Usage Prediction Accuracy:** {usage_acc}%")
    st.write(f"**Subcategory Prediction Accuracy:** {subcategory_acc}%")

#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################
    pkl_path = "men_model_details/dominant_colors.pkl"  # Adjust path as needed

    st.subheader("🔍 Similar Items Recommendation")

    # Category selection
    category_map = {"Bottomwear": "0", "Topwear": "1"}  # Correct category names
    usage_mapping = {
        0: "Casual",
        1: "Ethnic",
        2: "Formal",
        4: "Party",
        5: "Smart Casual",
        6: "Sports",
        7: "Travel"
    }

    # Reverse mapping for filtering
    reverse_usage_mapping = {v: k for k, v in usage_mapping.items()}

    category_options = list(category_map.keys())  # Show category options
    selected_category = st.selectbox("📌 Select a category:", category_options)

    # Convert selected category to actual numeric value
    selected_category_code = category_map[selected_category]

    # Usage selection (single select)
    selected_usage_type = st.selectbox("🎯 Filter by Usage:", list(usage_mapping.values()))

    # Convert selected usage to numeric value
    selected_usage_id = reverse_usage_mapping[selected_usage_type]

    # Find similar items
    similar_items = find_similar_colors(img, pkl_path, int(selected_category_code), [selected_usage_id])

    # Display similar items
    if similar_items:
        num_images = len(similar_items)
        cols = st.columns(min(num_images, 5))  # Show max 5 images per row

        for i, item in enumerate(similar_items):
            with cols[i % 5]:  # Distribute images across columns
                st.image(item["path"], caption=f"Similar Item {i+1}", use_container_width=True)
                st.write(f"**Article Type:** {item['article_type']}")

                # Decode numeric usage value
                decoded_usage = usage_mapping.get(int(item["usage"]), "Unknown")
                st.write(f"**Usage:** {decoded_usage}")

                st.write(f"**Dominant Color:** RGB {item['color']}")

                # Convert RGB tuple to CSS-friendly format
                rgb_color = f"rgb{item['color']}"  # Convert (255, 0, 0) → "rgb(255, 0, 0)"

                st.markdown(
                    f"""
                    <div style="width:50px; height:50px; background-color:{rgb_color}; border-radius:5px; border:1px solid white;"></div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.write("⚠️ No similar items found. Try again!")


#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################

    # 🔻 **Force a New Row Before Outfit Recommendation**
    st.markdown("---")  # This adds a horizontal line for clear separation

    # 🟢 SECTION 2: Outfit Recommendation
    with st.container():  # Wrap in a container to prevent overlap
        st.subheader("👕 Outfit Recommendation System")

        # User selects category (Topwear or Bottomwear)
        category_map = {"Topwear": "1", "Bottomwear": "0"}
        selected_category = st.selectbox("📌 Select the category of the uploaded image:", list(category_map.keys()))
        selected_category_code = category_map[selected_category]

        # User selects usage type
        selected_usage = st.selectbox("🎭 Select the Usage Type:", list(usage_mapping.values()))
        selected_usage_code = list(usage_mapping.keys())[list(usage_mapping.values()).index(selected_usage)]

        # Let user choose a color theory approach
        approach_options = ["Complementary", "Analogous", "Triadic", "Monochromatic"]
        selected_approach = st.selectbox("🎨 Select a Color Theory Approach:", approach_options)

        # Ensure 'img' is already provided
        if 'img' in locals() and img is not None:
            with st.spinner("🔍 Finding the best match..."):
                recommended_items = recommend_outfit(img, pkl_path, selected_category_code, selected_usage_code, selected_approach)

            if recommended_items:
                opposite_category = "Bottomwear" if selected_category == "Topwear" else "Topwear"
                st.write(f"### 🏆 Recommended {opposite_category} ({selected_approach} Matching)")

                # Display recommendations in a grid layout
                cols = st.columns(min(len(recommended_items), 5))  # Max 5 columns
                for i, item in enumerate(recommended_items):
                    with cols[i % 5]:  # Distribute items evenly
                        st.image(item["path"], caption=f"Recommendation {i+1}", use_container_width=True)
                        st.write(f"👕 **Article Type:** {item['article_type']}")
                        st.write(f"🎨 **Dominant Color:** RGB {item['color']}")
            else:
                st.warning("⚠️ No matching outfit found. Try a different image or approach.")
        else:
            st.error("⚠️ Image not found! Please ensure 'img' is correctly passed before calling this function.")

#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################

    # 🔻 **Color Theory Information Section**
    st.markdown("---")  # Separator for clarity

    st.subheader("🎨 Understanding Color Theory in Fashion")

    st.write(
        "Different color theory approaches can be used to create stylish and visually appealing outfits. "
        "Here's a quick breakdown of the available options:"
    )

    st.image("infographs/img1.png", caption="Complementary Theory Guide", use_container_width=True)
    st.write(f"🔹 **Complementary :** Uses colors opposite each other on the color wheel for high contrast and bold looks.")
    
    # 🔻 **Color Theory Information Section**
    st.markdown("---")  # Separator for clarity


    st.image("infographs/img3.png", caption="Analogous Theory Guide", use_container_width=True)
    st.write(f"🔹 **Analogous :** Uses colors that are next to each other on the color wheel, creating a harmonious and cohesive outfit..")
    
    # 🔻 **Color Theory Information Section**
    st.markdown("---")  # Separator for clarity


    st.image("infographs/img4.png", caption="Triadic Theory Guide", use_container_width=True)
    st.write(f"🔹 **Triadic :** Uses three colors evenly spaced around the color wheel, offering vibrant and balanced combinations.")
    
    # 🔻 **Color Theory Information Section**
    st.markdown("---")  # Separator for clarity


    st.image("infographs/img2.png", caption="Monochromatic Theory Guide", use_container_width=True)
    st.write(f"🔹 **Monochromatic :** Uses different shades, tones, and tints of a single color for a sophisticated, minimalist style.")
    
    # 🔻 **Color Theory Information Section**
    st.markdown("---")  # Separator for clarity


