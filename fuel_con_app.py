import streamlit as st
import pickle
import numpy as np  

# Load the trained model
model_path = 'fuel_consumption_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define the unique values for one-hot encoding
fuel_categories = ['D (Diesel)', 'E (E85 Ethanol)', 'N (Natural Gas)', 'X (Regular Petrol)', 'Z (Premium Petrol)']

transmission_categories = [
    'A10', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9',
    'AM5', 'AM6', 'AM7', 'AM8', 'AM9', 'AS10', 'AS4',
    'AS5', 'AS6', 'AS7', 'AS8', 'AS9', 'AV1', 'AV10',
    'AV6', 'AV7', 'AV8', 'M4', 'M5', 'M6', 'M7'
]

vehicle_class_categories = [
    'COMPACT', 'FULL-SIZE', 'MID-SIZE', 'MINICOMPACT', 'MINIVAN', 
    'PICKUPTRUCK-SMALL', 'PICKUPTRUCK-STANDARD', 'SPECIALPURPOSEVEHICLE',
    'STATIONWAGON-MID-SIZE', 'STATIONWAGON-SMALL', 'SUBCOMPACT', 'SUV', 
    'SUV-SMALL', 'SUV-STANDARD', 'TWO-SEATER', 'VAN-CARGO', 'VAN-PASSENGER'
]

# Function to create a one-hot encoded input array for the model
def create_input_array(engine_size, cylinders, fuel, transmission, vehicle_class):
    # Initialize a zero array with length 53 (as per your model's input size)
    input_array = np.zeros(53)

    # Set the engine size and cylinders in their respective positions
    input_array[0] = engine_size
    input_array[1] = cylinders
    
    # One-hot encode the fuel type
    if fuel in fuel_categories:
        fuel_index = fuel_categories.index(fuel) + 2
        input_array[fuel_index] = 1

    # One-hot encode the transmission type
    if transmission in transmission_categories:
        trans_index = transmission_categories.index(transmission) + 7  # Adjust index based on fuel categories
        input_array[trans_index] = 1
    
    # One-hot encode the vehicle class
    if vehicle_class in vehicle_class_categories:
        class_index = vehicle_class_categories.index(vehicle_class) + 36  # Adjust index based on fuel and transmission categories
        input_array[class_index] = 1
    
    return input_array

# Define the main function for the Streamlit app
def main():
    st.title("Fuel Consumption Predictor")

    # Add description
    st.markdown("""
    ### Description:
    This project analyzes the fuel consumption/efficiency of different cars using a dataset containing various car attributes. 
    We employ machine learning models to predict and understand factors influencing fuel consumption.
    """)

    st.markdown("### Input the car details below to predict fuel consumption:")
    
    # Add a divider
    st.write("---")
    
    # Input fields for user data
    engine_size = st.slider("Engine Size (L)", min_value=1.0, max_value=10.0, step=0.1)
    cylinders = st.slider("Number of Cylinders", min_value=2, max_value=16, step=1)
    fuel = st.selectbox("Fuel Type", options=fuel_categories)
    transmission = st.selectbox("Transmission (Number=Gears)", options=transmission_categories)
    vehicle_class = st.selectbox("Vehicle Class", options=vehicle_class_categories)

    # When the Predict button is clicked
    if st.button("Predict Fuel Consumption"):
        # Create input array
        input_array = create_input_array(engine_size, cylinders, fuel, transmission, vehicle_class)
        
        # Make a prediction
        prediction = model.predict([input_array])[0]
        
        # Display the prediction
        st.success(f"The predicted fuel consumption is {prediction:.2f} L/100 km  or {100/prediction:.2f} km/L")
    
    # Add another divider
    st.write("---")
    
    # Additional information
    st.markdown("""
    ### Dataset Information:
    The dataset used in this project includes attributes such as:
    
    - **YEAR**: The year of the car model.
    - **MAKE**: The manufacturer of the car.
    - **MODEL**: The model name of the car.
    - **VEHICLE CLASS**: The class of the vehicle (e.g., compact, mid-size, subcompact).
    - **ENGINE SIZE**: The size of the engine in liters.
    - **CYLINDERS**: The number of cylinders in the engine.
    - **TRANSMISSION**: The type of transmission (e.g., A4 for automatic with 4 gears, M5 for manual with 5 gears).
    - **FUEL**: The type of fuel used (e.g., X for regular gasoline, Z for premium gasoline).
    - **FUEL CONSUMPTION**: The fuel consumption in the city in liters per 100 km.
    - **HWY (L/100 km)**: The fuel consumption on the highway in liters per 100 km.
    - **COMB (L/100 km)**: The combined fuel consumption in liters per 100 km.
    - **COMB (mpg)**: The combined fuel consumption in miles per gallon.
    - **EMISSIONS**: The CO2 emissions in grams per km.
    """)

    st.markdown("### Model Details:")
    st.markdown("""
    We trained various models to predict fuel consumption, and the best-performing model is deployed in this app. 
    It takes into account engine size, number of cylinders, fuel type, transmission type, and vehicle class to provide 
    an accurate prediction of fuel consumption.
    """)

# Run the app
if __name__ == "__main__":
    main()
