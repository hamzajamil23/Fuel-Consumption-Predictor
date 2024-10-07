Fuel Consumption Predictor
![image](https://github.com/user-attachments/assets/3777a991-3532-41eb-83af-2b2cb0f0f10c)


Table of Contents
Overview
Features
Technologies Used
Installation
Usage
Dataset
Model
License
Contact
Overview
The Fuel Consumption Predictor is a web application designed to estimate fuel consumption based on various vehicle specifications, including engine size, number of cylinders, fuel type, transmission type, and vehicle class. The application uses machine learning techniques to provide accurate predictions, assisting users in understanding fuel efficiency better.

Features
User-friendly interface to input vehicle details.
Real-time fuel consumption predictions.
Visualizations that illustrate the relationship between engine size and fuel consumption.
Detailed information about the dataset and model used.
Technologies Used
Programming Languages: Python
Web Framework: Streamlit
Machine Learning: XGBoost, scikit-learn
Data Analysis: Pandas, NumPy
Data Visualization: Matplotlib, Seaborn
Deployment: Streamlit Cloud
Installation
To run this application locally, follow these steps:

Clone the repository:

bash
Copy code
git clone https://github.com/hamzajamil23/Fuel-Consumption-Predictor.git
Navigate to the project directory:

bash
Copy code
cd Fuel-Consumption-Predictor
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit application:

bash
Copy code
streamlit run fuel_con_test.py
Usage
After launching the application, enter the vehicle details in the provided fields, and click on the "Predict" button to obtain the estimated fuel consumption. The application will display the results alongside a visual representation of engine size versus fuel consumption.

Dataset
The dataset used for training the models is sourced from Kaggle. You can access the dataset here: Fuel Consumption Dataset.

Model
The application employs the XGBoost model, which has demonstrated superior performance in terms of accuracy compared to other models. The model is deployed on Streamlit Cloud, allowing users to access it online at the following link: Fuel Consumption Predictor Web App.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contact
For any inquiries or feedback, please feel free to contact me at:

GitHub: hamzajamil23
Email: hamzajamilbaig@gmail.com
