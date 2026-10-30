# Image Classification with 10 Categories

## Project Overview

This project develops a deep learning image classification system that recognizes images belonging to 10 different object categories.

It uses Convolutional Neural Networks (CNN) and transfer learning with MobileNetV2 to classify uploaded images. A Streamlit web application provides an interactive interface for making predictions.

## Project Objectives

- Explore and preprocess a multi-class image dataset.
- Perform exploratory data analysis (EDA).
- Build and train a custom CNN model.
- Apply data augmentation.
- Implement transfer learning using MobileNetV2.
- Evaluate and compare model performance.
- Save and reload the selected trained model.
- Deploy an interactive image classification application using Streamlit.

## Categories

1. Tench
2. Springer
3. Cassette Player
4. Chainsaw
5. Church
6. French Horn
7. Garbage Truck
8. Gas Pump
9. Golf Ball
10. Parachute

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow
- Streamlit
- Jupyter Notebook

## Model Performance Comparison

Two deep learning approaches were developed and evaluated:

| Metric | Custom CNN | MobileNetV2 |
|---|---:|---:|
| Accuracy | 56.84% | 98.75% |
| Precision | 60.03% | 98.79% |
| Recall | 56.73% | 98.73% |
| F1-Score | 56.08% | 98.75% |

MobileNetV2 achieved the best validation performance and was selected for the Streamlit application.

The selected trained model is saved as `best_image_classification_model.keras`.
## Streamlit Application

The application allows users to:

- Upload JPG, JPEG, PNG, and WEBP images.
- Classify uploaded images into one of 10 categories.
- View prediction confidence.
- View the top 3 predicted categories.

## Application Testing

Ten exported validation images were tested through the prediction pipeline.

- Images tested: 10
- Correct predictions: 10
- Sample accuracy: 100%

**Note:** This is a small validation-image application check, not an independent test-set accuracy measurement. Model performance should be assessed using the complete held-out evaluation dataset.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app.py
```

## Project Structure

```text
PRAICP-1010-Image-Classification/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── best_image_classification_model.keras
│   ├── class_mapping.json
│   └── app_config.json
│
├── outputs/
│   └── application_test_results.csv
│
└── sample_images/
```

## Disclaimer

This project is developed for educational and learning purposes as part of the DataMites project PRAICP-1010.
