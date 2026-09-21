# 🏠 House Price Prediction

A Machine Learning project that predicts the selling price of a house based on its characteristics, using supervised regression models and an interactive Streamlit application.

## 📌 Project Overview

This project uses the **House Prices - Advanced Regression Techniques** dataset to build a complete Machine Learning solution for house price prediction.

The project covers the complete workflow:

* Data exploration and preparation
* Exploratory Data Analysis
* Feature Engineering
* Training and comparison of regression models
* Cross-validation
* Hyperparameter optimization with GridSearchCV
* Model evaluation
* Feature importance analysis
* Interactive predictions with Streamlit
* Docker containerization

The target variable is **`SalePrice`**.

## 🎯 Main Objectives

* Analyze and clean the housing dataset.
* Handle missing values, duplicates, and outliers.
* Prepare numerical and categorical features.
* Create relevant features to improve predictions.
* Train and compare at least three regression models.
* Evaluate models using **MAE, RMSE, and R²**.
* Use cross-validation to validate model performance.
* Optimize a model using **GridSearchCV**.
* Identify the most important features influencing house prices.
* Save the trained model for later predictions.
* Create a Streamlit application for estimating house prices.
* Containerize the application using Docker.

## 🎓 Context

This project was completed individually at **YouCode** from **September 21 to September 25, 2026**, as part of a **5-day Machine Learning brief**.

The project was based on a real-world business scenario where a real estate agency wants to automatically estimate the selling price of a property based on its characteristics.

The objective was to practice the complete Machine Learning workflow, from **data preparation and exploratory analysis to model training, evaluation, optimization, and deployment**.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* Joblib / Pickle
* Docker
* Git & GitHub

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/your-username/house-price-prediction.git
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser and allow you to enter the characteristics of a house and receive an estimated selling price.

### 🐳 Run with Docker

Build the Docker image:

```bash
docker build -t house-price-prediction .
```

Run the container:

```bash
docker run -p 8501:8501 house-price-prediction
```

Then open the Streamlit application in your browser.
