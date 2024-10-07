import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model_path = 'fuel_consumption_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define the unique values for one-hot encoding
fuel_categories = ['D (Diesel)', 'E (E85 Ethanol)', 'N (Natural Gas)', 'X (Regular Petrol)', 'Z (Premium Petrol)']
transmission_categories = {
    'A10': 'Automatic (10 gears)',
    'A3': 'Automatic (3 gears)',
    'A4': 'Automatic (4 gears)',
    'A5': 'Automatic (5 gears)',
    'A6': 'Automatic (6 gears)',
    'A7': 'Automatic (7 gears)',
    'A8': 'Automatic (8 gears)',
    'A9': 'Automatic (9 gears)',
    'AM5': 'Auto-manual (5 gears)',
    'AM6': 'Auto-manual (6 gears)',
    'AM7': 'Auto-manual (7 gears)',
    'AM8': 'Auto-manual (8 gears)',
    'AM9': 'Auto-manual (9 gears)',
    'AS10': 'Automatic with Sport Mode (10 gears)',
    'AS4': 'Automatic with Sport Mode (4 gears)',
    'AS5': 'Automatic with Sport Mode (5 gears)',
    'AS6': 'Automatic with Sport Mode (6 gears)',
    'AS7': 'Automatic with Sport Mode (7 gears)',
    'AS8': 'Automatic with Sport Mode (8 gears)',
    'AS9': 'Automatic with Sport Mode (9 gears)',
    'AV1': 'Continuously Variable (1 gear)',
    'AV10': 'Continuously Variable (10 gears)',
    'AV6': 'Continuously Variable (6 gears)',
    'AV7': 'Continuously Variable (7 gears)',
    'AV8': 'Continuously Variable (8 gears)',
    'M4': 'Manual (4 gears)',
    'M5': 'Manual (5 gears)',
    'M6': 'Manual (6 gears)',
    'M7': 'Manual (7 gears)'
}


vehicle_class_categories = [
    'COMPACT', 'FULL-SIZE', 'MID-SIZE', 'MINICOMPACT', 'MINIVAN', 
    'PICKUPTRUCK-SMALL', 'PICKUPTRUCK-STANDARD', 'SPECIALPURPOSEVEHICLE',
    'STATIONWAGON-MID-SIZE', 'STATIONWAGON-SMALL', 'SUBCOMPACT', 'SUV', 
    'SUV-SMALL', 'SUV-STANDARD', 'TWO-SEATER', 'VAN-CARGO', 'VAN-PASSENGER'
]

# Function to create a one-hot encoded input array for the model
def create_input_array(engine_size, cylinders, fuel, transmission, vehicle_class):
    input_array = np.zeros(53)
    input_array[0] = engine_size
    input_array[1] = cylinders
    
    if fuel in fuel_categories:
        fuel_index = fuel_categories.index(fuel) + 2
        input_array[fuel_index] = 1

    if transmission in transmission_categories:
        trans_index = transmission_categories.index(transmission) + 7
        input_array[trans_index] = 1
    
    if vehicle_class in vehicle_class_categories:
        class_index = vehicle_class_categories.index(vehicle_class) + 36
        input_array[class_index] = 1
    
    return input_array

# Define the main function for the Streamlit app
def main():
    st.set_page_config(layout="wide", page_title="Fuel Consumption Predictor")
    
    st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.5rem;  /* Set top padding to zero */
        }
    </style>
    """,
    unsafe_allow_html=True
)

    st.title("Fuel Consumption Predictor")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("Enter Vehicle Details")
        engine_size = st.slider("Engine Size (L)", min_value=1.0, max_value=10.0, step=0.1, value=1.3)
        cylinders = st.slider("Number of Cylinders", min_value=2, max_value=16, step=1, value=4)
        fuel = st.selectbox("Fuel Type", options=fuel_categories, index=fuel_categories.index('X (Regular Petrol)'))
        transmission_list = list(transmission_categories.values())  # Convert dictionary values to a list
        transmission = st.selectbox("Transmission", options=transmission_list, index=transmission_list.index('Automatic (5 gears)'))
        vehicle_class = st.selectbox("Vehicle Class", options=vehicle_class_categories, index=vehicle_class_categories.index('SUV'))

        
        prediction = None  # Initialize prediction variable

        if st.button("Predict"):
            input_array = create_input_array(engine_size, cylinders, fuel, transmission, vehicle_class)
            prediction = model.predict([input_array])[0]
            st.success(f"The predicted fuel consumption is **{prediction:.2f} L/100 km** or **{100/prediction:.2f} km/L**")
        
        

    # Load CSV data
    df = pd.read_csv('Fuel_Consumption_2000-2022.csv')

    with col2:
            
        st.markdown("<h3 class='subtitle'>Engine Size vs Fuel Consumption</h3>", unsafe_allow_html=True)

        # Create a professional-looking scatter plot using Seaborn
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='ENGINE SIZE', y='COMB (L/100 km)', ax=ax, color='#3498db', s=60)
        ax.set_title("Engine Size vs Fuel Consumption", fontsize=16)
        ax.set_xlabel('Engine Size (L)', fontsize=12)
        ax.set_ylabel('Fuel Consumption (L/100km)', fontsize=12)
       
        # Plot only if prediction exists
        if prediction is not None:
            ax.plot(engine_size, prediction, 'ro', label='Your Car', markersize=7)
        
        st.pyplot(fig)

    st.write("---")
    st.markdown("## Dataset Overview")
    st.markdown("""
    The dataset utilized for training the fuel consumption prediction model contains features that help us understand vehicle performance and emissions. Below is a detailed description of each attribute as outlined below:

    | **Attribute**                        | **Description**                                                            |
    |--------------------------------------|---------------------------------------------------------------------------|
    | **YEAR**                             | The year the car model was manufactured.                                  |
    | **MAKE**                             | The manufacturer of the vehicle.                                         |
    | **MODEL**                            | The specific name or designation of the vehicle model.                   |
    | **VEHICLE CLASS**                    | The classification of the vehicle type (e.g., Compact, SUV, Sedan, etc.).|
    | **ENGINE SIZE**                      | The size of the engine, measured in liters.                             |
    | **CYLINDERS**                        | The total number of cylinders in the engine.                             |
    | **TRANSMISSION**                     | The type of transmission (e.g., Automatic, Manual).                     |
    | **FUEL CONSUMPTION (L/100 km)**      | The amount of fuel consumed in urban driving conditions, measured in liters per 100 kilometers. |
    | **HWY FUEL CONSUMPTION (L/100 km)**  | The amount of fuel consumed on highways, measured in liters per 100 kilometers. |
    | **COMBINED FUEL CONSUMPTION (L/100 km)** | The overall fuel consumption calculated from both urban and highway driving, measured in liters per 100 kilometers. |
    | **COMBINED FUEL CONSUMPTION (mpg)** | The overall fuel efficiency, measured in miles per gallon.               |
    | **EMISSIONS (g/km)**                | The amount of carbon dioxide (CO2) emissions produced, measured in grams per kilometer. |
    """)

    st.markdown("## Model Details")
    st.markdown("""
    The **Fuel Consumption Prediction Model** employs several features to predict fuel efficiency. Key highlights include:

    - **Engine Size**: Larger engine sizes generally correlate with higher fuel consumption.
    - **Cylinders**: The number of cylinders can influence both performance and efficiency.
    - **Fuel Type and Transmission**: Different fuel types and transmission modes significantly affect fuel efficiency.
    
    This model utilizes advanced algorithms to ensure accurate predictions, with the **XGBoost** algorithm demonstrating optimal performance in terms of accuracy and reliability.
    """)


# Run the app
if __name__ == "__main__":
    main()
