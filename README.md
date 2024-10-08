# Fuel Consumption Predictor

A Machine Learning web application to predict fuel consumption based on vehicle characteristics such as engine size, number of cylinders, transmission type, and vehicle class. The app is built using Python and deployed on Streamlit Cloud.

- **Live Demo**: [Fuel Consumption Predictor](https://fuel-consumption-predictor.streamlit.app/)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [License](#license)
- [Contact](#contact)

## Introduction
This project is a part of my Data Science capstone. The application predicts fuel consumption (in liters per 100 kilometers) based on various vehicle features. It uses machine learning models, with the XGBoost model giving the best prediction accuracy, and is deployed for real-time prediction.

## Features
- Predict fuel consumption based on vehicle characteristics:
  - Engine Size (L)
  - Number of Cylinders
  - Transmission Type (Manual/Automatic)
  - Vehicle Class (Compact, SUV, etc.)
- Visualize engine size vs. fuel consumption with a scatter plot.
- Interactive user interface built with Streamlit.

## Dataset
The dataset used for this project is available on Kaggle:  
[Fuel Consumption Dataset](https://www.kaggle.com/datasets/ahmettyilmazz/fuel-consumption)

- **Columns in Dataset:**
  - `YEAR`: Year of the car model
  - `MAKE`: Car manufacturer
  - `MODEL`: Car model name
  - `VEHICLE CLASS`: Type of vehicle (e.g., compact, mid-size)
  - `ENGINE SIZE`: Size of the engine in liters
  - `CYLINDERS`: Number of cylinders
  - `TRANSMISSION`: Transmission type (e.g., automatic, manual)
  - `FUEL CONSUMPTION`: Fuel consumption in the city (L/100 km)
  - `HWY (L/100 km)`: Fuel consumption on highways (L/100 km)
  - `COMB (L/100 km)`: Combined fuel consumption (L/100 km)
  - `COMB (mpg)`: Combined fuel consumption (miles per gallon)
  - `EMISSIONS`: CO2 emissions (g/km)

## Technologies Used
- **Programming Language**: Python
- **Framework**: Streamlit
- **Machine Learning Models**: XGBoost, Random Forest, Linear Regression
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Streamlit Cloud
- **Libraries**: 
  - Pandas
  - NumPy
  - scikit-learn
  - XGBoost
  - Seaborn
  - Matplotlib

## Installation
To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hamzajamil23/Fuel-Consumption-Predictor.git
2. **Navigate to the project directory**:
    ```bash
    cd Fuel-Consumption-Predictor
3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
4. **Run the Streamlit app**:
    ```bash
    streamlit run fuel_con_app.py

### Usage
- Open the application on your local machine by navigating to the provided URL after running the above commands.
- Input vehicle characteristics such as engine size, cylinders, transmission, and vehicle class.
- Get the predicted fuel consumption and visualize the engine size vs. fuel consumption on the scatter plot.

### Model
The model used for fuel consumption prediction is **XGBoost** (eXtreme Gradient Boosting). Other models, including Random Forest and Linear Regression, were tested, but **XGBoost** provided the highest accuracy.

#### Model Training
- **Training Data**: A portion of the Kaggle dataset was used for model training.
- **Features**: Engine size, cylinders, fuel type, transmission, and vehicle class were used to train the model.

### Results
The **XGBoost** model demonstrated the best performance with the lowest error rates in predicting fuel consumption. The deployed model is available via the Streamlit web app.

### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Contact
For any questions or feedback, feel free to reach out to me:

- **Name**: Hamza Jamil
- **GitHub**: [hamzajamil23](https://github.com/hamzajamil23)
- **Email**: [hamzajamilbaig@gmail.com](mailto:hamzajamilbaig@gmail.com)

