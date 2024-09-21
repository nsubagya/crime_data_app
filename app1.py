import streamlit as st
from pycaret.classification import load_model, predict_model
import pandas as pd
import folium

# Load the model
model = load_model("https://github.com/nsubagya/crime_data_app/blob/main/final_model")

# Load your data
data_2023 = pd.read_csv("https://github.com/nsubagya/crime_data_app/blob/main/data_after_2023.csv")

# Filter relevant data and compute mean coordinates
area_coordinates = data_2023[['AREA', 'LAT', 'LON']].dropna(subset=['LAT', 'LON'])
mean_coordinates = area_coordinates.groupby('AREA').agg({'LAT': 'mean', 'LON': 'mean'}).reset_index()
area_coordinates_dict = mean_coordinates.set_index('AREA').T.to_dict('list')
area_coordinates_dict = {k: (v[0], v[1]) for k, v in area_coordinates_dict.items()}

def get_coordinates(area):
    # Return coordinates for the given AREA number
    return area_coordinates_dict.get(area, (None, None))

# Streamlit App
st.title('Crime Data Prediction and Mapping')

# Initialize session state if it doesn't exist
if 'results_list' not in st.session_state:
    st.session_state.results_list = []

# Create a date input for user to select the date of occurrence
selected_date = st.date_input('Date of Occurrence', value=pd.Timestamp.now())

# Extract year, month, and day from the selected date
year = selected_date.year
month = selected_date.month
day = selected_date.day

# Create a slider for selecting the hour of occurrence
hour = st.slider('Hour of Occurrence', min_value=0, max_value=23, value=0)

# Add input for 'Crm Cd' (Crime Code)
crm_cd = st.number_input('Crime Code (Crm Cd)', min_value=0, max_value=1000, value=0)

# Add input for 'Vict Age' (Victim Age)
vict_age = st.slider('Victim Age (Vict Age)', min_value=0, max_value=120, value=30)

# Add input for 'Vict Sex' (Victim Sex)
vict_sex = st.selectbox('Victim Sex (Vict Sex)', options=[1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female')

# Add input for 'Premis Cd' (Premises Code)
premis_cd = st.number_input('Premises Code (Premis Cd)', min_value=0, max_value=1000, value=0)

# Add input for 'Weapon Used Cd' (Weapon Used Code)
weapon_used_cd = st.number_input('Weapon Used Code (Weapon Used Cd)', min_value=0, max_value=1000, value=0)

# Add input for 'Status'
status = st.selectbox('Status', options=['A', 'I'], format_func=lambda x: 'Active' if x == 'A' else 'Inactive')

# Prepare input data for prediction
input_data = pd.DataFrame({
    'YEAR_occ': [year],
    'MONTH_occ': [month],
    'DAY_occ': [day],
    'HOUR_OF_DAY_occ': [hour],
    'Crm Cd': [crm_cd],  
    'Vict Age': [vict_age],  
    'Vict Sex': [vict_sex], 
    'Premis Cd': [premis_cd],  
    'Weapon Used Cd': [weapon_used_cd], 
    'Status': [status], 
    'YEAR_rptd': [year],
    'MONTH_rptd': [month],
    'DAY_rptd': [day]
})

# Make prediction
if st.button('Predict'):
    predicted_prob = predict_model(model, data=input_data)
    predicted_area = predicted_prob['prediction_label'].values[0]  

    # Get coordinates
    lat, lon = get_coordinates(predicted_area)

    # Store results
    if lat is not None and lon is not None:
        st.session_state.results_list.append((predicted_area, lat, lon))

    # Display results
    st.write(f'Predicted AREA: {predicted_area}')
    st.write(f'Coordinates: Latitude {lat}, Longitude {lon}')

# Create a map and plot all markers
if st.session_state.results_list:
    # Initialize the map at the first result location
    first_lat, first_lon = st.session_state.results_list[0][1:3]
    m = folium.Map(location=[first_lat, first_lon], zoom_start=12)

    # Add markers to the map for all predictions
    for predicted_area, lat, lon in st.session_state.results_list:
        folium.Marker(
            location=[lat, lon],
            popup=f'Predicted Area: {predicted_area}',
            tooltip=f'Predicted Area: {predicted_area}'  # Tooltip for hover
        ).add_to(m)

    st.write('Map of the Predicted Locations:')
    st.components.v1.html(m._repr_html_(), height=500)
else:
    st.write('No predictions made yet.')
