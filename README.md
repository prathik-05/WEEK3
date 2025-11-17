♻️ Smart Waste Classification – Week 3
Deep Learning Model + Web App (Streamlit)

A Sustainability-Themed AI Project

📌 Overview
This project is part of the Skill4Future AI/ML Internship (Sustainability Theme).
In Week 3, the goal is to:

->Train a CNN model to classify Organic vs Recyclable waste
->Preprocess and load dataset from Kaggle
->Generate training accuracy graph
->Save the trained model
->Build a simple Streamlit Web App for prediction
->Upload everything to GitHub

📁 Dataset
Dataset (Kaggle):
🔗 https://www.kaggle.com/datasets/techsash/waste-classification-data

The dataset contains 2 classes:
    O → Organic Waste
    R → Recyclable Waste
Folder structure:
DATASET/
│── TRAIN/
│     ├── O/
│     └── R/
│
└── TEST/
      ├── O/
      └── R/
🧠 Model (CNN)
A simple CNN model was trained:
    Input size: 128×128×3
    Optimizer: Adam
    Loss: Categorical Crossentropy
    Epochs: 10
    Classes: 2
Files saved as:
  waste_cnn_model.h5
  accuracy_plot.png

📙 Notebook
Training code is inside:
  📄 Week3-train.ipynb 

This notebook includes:
  -Dataset loading
  -Preprocessing
  -Model training
  -Accuracy plot
  -Model saving
  
🌐 Streamlit Web App
File: app2.py
This app allows users to upload an image and predicts:
    -Organic Waste
    -Recyclable Waste

pip install -r requirements.txt
streamlit run app2.py

📂 Repository Structure
WEEK3/
│── README.md
│── Week3-train.ipynb
│── app2.py
│── accuracy_plot.png
│── waste_cnn_model.h5   (LFS)
│── requirements.txt

✍️ Author

S. Prathik
GitHub: https://github.com/prathik-05
