# 🌟 Skin Cancer Prediction Using Convolutional Neural Networks (CNN)

A deep learning project that classifies malignant vs. benign skin cancer images using a custom-built Convolutional Neural Network (CNN).
The model is trained on a high-quality Kaggle dataset and achieves strong performance in predicting skin cancer from dermoscopic images.

## 📌 Project Overview

Skin cancer is one of the most common cancers worldwide, and early detection plays a crucial role in improving patient outcomes.
This project applies **deep learning (CNN)** to automatically classify skin lesions into:

- **Benign**

- **Malignant**

The model is trained and evaluated using image data and achieves robust performance.


📁 Dataset

The dataset used in this project is publicly available on Kaggle:

🔗 [Skin Cancer (Malignant vs Benign)](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)

The dataset includes dermoscopic images categorized into two classes.
All images were preprocessed, augmented, and fed into the CNN training pipeline.

## 🧠 Model Architecture

![Model Architecture](https://github.com/user-attachments/assets/4f028e20-7434-4190-a98f-8f6da9e74341)


## 📊 Model Performance

After training the model, the following metrics were achieved:

Metric|	Score
------| ------
Loss	|3.659
Accuracy|	0.810
Precision	|0.826
Recall	|0.772

These results indicate that the model performs well in distinguishing between benign and malignant lesions.


## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- openCV
- NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Kaggle API

## 📈 Example Predictions

The model outputs:

- Predicted class (Benign / Malignant)

- Probability scores

Visualization includes sample predictions and confusion matrix.

## 🧪 Future Improvements

- Integrating Transfer Learning (EfficientNet / ResNet / Inception)

- Deploying the model as a Web App using Streamlit

- Improving augmentation and class balancing

- Adding Grad-CAM visualization for explainability

## 🙌 Acknowledgments

Special thanks to **Kaggle** and the dataset creator for providing the data used in this project.