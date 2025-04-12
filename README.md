# 🔥 Smart Outfit Recommender 🔥
![DALL·E 2025-03-13 03 07 06 - A futuristic and stylish profile image for a fashion recommendation project  The image features a sleek AI-powered mannequin with a glowing neural net](https://github.com/user-attachments/assets/d3d0b12c-9c3d-4bfb-970c-2e7c7d66aa21)


# 📌 Overview
The Smart Outfit Recommender is an AI-powered system that predicts clothing categories and suggests outfits based on color coordination. Using deep learning and machine learning, it helps users discover stylish and well-matched outfit recommendations effortlessly.

# 📌 Description
The Smart Outfit Recommender leverages advanced artificial intelligence to analyze clothing attributes such as color, style, and category. By utilizing deep learning models, it identifies patterns in fashion choices and suggests complementary outfit combinations. The system enhances the user experience by ensuring that recommended outfits align with personal preferences and modern fashion trends. It can be integrated into e-commerce platforms, personal wardrobe assistants, or virtual styling applications to provide data-driven fashion guidance.

# 📌 Usage Example


https://github.com/user-attachments/assets/20805310-63d6-4d98-98ee-8951790213e8


## ✨ Features
- **Usage Prediction Model**: Classifies clothing items based on their intended use (e.g., casual, formal, sportswear).
- **Subcategory Classification**: Predicts specific clothing subcategories (e.g., t-shirts, jackets, jeans).
- **Color-Based Recommendation**: Uses Delta E color distance to suggest visually harmonious outfits.
- **Self-Training Capability**: Improves over time with unlabeled data.
- **Performance Visualization**: Provides accuracy and loss graphs, along with a confusion matrix for model evaluation.

## Accuracy of Models
- **Men Model**
- **99.07%**

- ![Screenshot 2025-02-26 022613](https://github.com/user-attachments/assets/607b36a7-2ed1-4b8e-8a88-cb1089a7cb44)

- **Women Model**:
  
- **97.90%**
 
-![Screenshot 2025-03-04 131115](https://github.com/user-attachments/assets/31062919-c3ab-4d5f-aa84-627dca458c48)


## 🚀 Installation

### Prerequisites
Ensure you have the following installed:
- Python (>=3.8)
- Git
- TensorFlow/Keras
- OpenCV & Scikit-Image
- Pandas & NumPy
- Streamlit (for UI deployment)

### Steps
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/outfit-recommendation.git
   cd outfit-recommendation
   ```
2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

### Running the Model
To train and evaluate the model:
```bash
python train.py
```

### Running the Recommendation System (Web UI)
To launch the web-based recommendation system:
```bash
streamlit run app.py
```

## 🛠 Technologies Used
- **Deep Learning**: TensorFlow, Keras
- **Image Processing**: OpenCV, Scikit-Image
- **Data Handling**: Pandas, NumPy
- **Web Interface**: Streamlit

## 📈 Model Performance
- **Accuracy & Loss Graphs**: Demonstrate the learning progress over epochs.
- **Accuracy Graphs**:
  
- Men Model Accuracy Graph :
- ![men_accuracy_plot](https://github.com/user-attachments/assets/4da1f245-0b95-43d0-ba1f-23db4c47bc12)
- Women Model Accuracy Graph :
- ![women_accuracy_plot](https://github.com/user-attachments/assets/429afdf4-1eee-4884-8d00-c411af095263)

- **Loss Graphs**:
- Men Model Loss Graph :
- ![men_loss_plot](https://github.com/user-attachments/assets/2237971a-2bb0-405d-8df4-09179bd58c3e)
- Women Model Loss Graph :
- ![women_loss_plot](https://github.com/user-attachments/assets/f62e82d2-7eda-4312-b483-18049dfda05e)

- **Confusion Matrix**: Provides insights into misclassifications.

- Men Model Confusion Matrix Graph
-![men_confusion_matrix](https://github.com/user-attachments/assets/dd3b2205-4455-44fc-8a33-193e513440be)
- Women Model Confusion Matrix Graph
-![women_confusion_matrix](https://github.com/user-attachments/assets/e01ec653-c08e-4126-83d7-dbcf10d02ece)

- **Color Matching**: Delta E ensures perceptually accurate outfit recommendations.

## 🤝 Contribution
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## 📜 License


## 📬 Contact
For inquiries, please reach out via manassawant5913@gmail.com or open an issue in the repository.

---
🎨 **Fashion meets AI – Transforming outfit recommendations with machine learning!**
