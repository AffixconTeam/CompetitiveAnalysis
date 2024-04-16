import pandas as pd
import streamlit as st
import geohash2
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import folium
from streamlit_folium import folium_static
import os
from collections import Counter
from itertools import combinations
import plotly.graph_objects as go
# Set up Streamlit app

st.set_page_config(page_title='Geo Segmentation',page_icon=':earth_asia:',layout='wide')
custom_css = """
<style>
body {
    background-color: #0E1117; 
    secondary-background {
    background-color: #262730; 
    padding: 10px; 
}
</style>
"""
st.write(custom_css, unsafe_allow_html=True)
st.markdown(custom_css, unsafe_allow_html=True)

st.title("Competitive Analysis")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Calculate differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Distance in kilometers
    distance = R * c
    return distance

# Function to decode geohash into latitude and longitude
def decode_geohash(geohash):
    if pd.notna(geohash):
        try:
            latitude, longitude = geohash2.decode(geohash)
            return pd.Series({'Home_latitude': latitude, 'Home_longitude': longitude})
        except ValueError:
            # Handle invalid geohashes, you can modify this part based on your requirements
            return pd.Series({'Home_latitude': None, 'Home_longitude': None})
        except TypeError:
            # Handle the case where geohash2.decode returns a single tuple
            return pd.Series({'Home_latitude': geohash[0], 'Home_longitude': geohash[1]})
    else:
        # Handle null values
        return pd.Series({'Home_latitude': None, 'Home_longitude': None})
# selected_distance = st.radio("Select Radius Type", ["Fixed Radius", "Varying Radius"])

# Generate random datetime values
start_date = pd.Timestamp('2023-12-01')
end_date = pd.Timestamp('2023-12-31')

st.sidebar.markdown(':red[**Select Date Range**]')

selected_start_date = st.sidebar.date_input("Select Start Date", start_date)
selected_end_date = st.sidebar.date_input("Select End Date", end_date)


selected_start_date = pd.to_datetime(selected_start_date)
selected_end_date = pd.to_datetime(selected_end_date)

df = pd.read_csv('10000_Movements.csv', sep=",").dropna(subset=['latitude', 'longitude'])
df['datetimestamp'] = pd.to_datetime(df['datetimestamp'])

df = df[(df['datetimestamp'] >= selected_start_date) & (df['datetimestamp'] <= selected_end_date)]
st.sidebar.markdown(f"<font color='orange'><b>Number of records within Date Range: {len(df)}</b></font>", unsafe_allow_html=True)


# df = pd.read_csv('10000_Movements.csv', sep=",").dropna(subset=['latitude', 'longitude'])
df[['Home_latitude', 'Home_longitude']] = df['homegeohash9'].apply(decode_geohash)
df[['Work_latitude', 'Work_longitude']] = df['workgeohash'].apply(decode_geohash)

items = {}
no_of_locations = st.text_input("Enter No of Addresses : ",value=5)
for i in range(1, int(no_of_locations) + 1):
    location_name = f"loc{i}"
    items[location_name] = []
    items['user_loc']=[]

dist = st.radio(":orange[**Select Distance Unit**]", ["Meters","Kilometers"])

if  no_of_locations:
    no_of_locations = int(no_of_locations)

    # if selected_distance=="Fixed Radius":
    #     count_within_radius = {}
    #     filtered_coordinates = {}
    #     industry_list = ['latitude', 'longitude']
    #     user_coordinates = []

    #     for _ in range(no_of_locations):
    #         location_data = []
    #         row = st.columns(2)
            
    #         for i, industry in enumerate(industry_list):
    #             value = row[i].number_input(f"{industry} - Location {_ + 1}", format="%.6f")
    #             location_data.append(value)
            
    #         user_coordinates.append(location_data)
    #     if len(user_coordinates[0]) ==2:
    #         for user_lat, user_lon in user_coordinates:
    #             count_within_radius[(user_lat, user_lon)] = 0
    #             filtered_coordinates[(user_lat, user_lon)] = []
    #         if dist == 'Kilometers':
    #             radius_input = st.slider("Select radius (in kilometers):", min_value=1, max_value=100, value=10)

    #         elif dist == 'Meters':
    #             radius_input = st.slider("Select radius (in Meters):", min_value=1, max_value=1000, value=10)
    #             radius_input=radius_input/1000

    #         for index, row in df.iterrows():
    #             for user_lat, user_lon in user_coordinates:
    #                 distance = haversine(user_lat, user_lon, row['latitude'], row['longitude'])
    #                 if distance <= radius_input:
    #                     count_within_radius[(user_lat, user_lon)] += 1
    #                     filtered_coordinates[(user_lat, user_lon)].append((row['latitude'], row['longitude']))
    #     else:
    #         st.warning("Please Enter Only Longitude and Latitude Sequentially")

    # else:
    count_within_radius = {}
    filtered_coordinates = {}
    home_counts = {}
    home_counts = {}
    work_counts = {}
    maid_ids_within_radius={}
    home_ids_within_radius={}
    work_ids_within_radius={}


    industry_list = ['latitude', 'longitude', 'Radius','Location']
    user_coordinates = []

    default_values = [[-33.864201, 151.21644, 30.0,'Royal Botanic Garden'], [-33.8471, 151.0634, 11.0,'Accor Stadium'], [-33.866159, 151.215256, 28.0,'Sydney Opera House'], [-33.8833, 151.2050, 22.0,'Central Station'], [-33.893000, 151.215000, 21.0,'Monash University']]

    for idx in range(no_of_locations):
        location_data = []
        row = st.columns(4)

        # Get the default values for the current location
        default_value = default_values[idx] if idx < len(default_values) else [0.0, 0.0, 0.0,'']

        for i, industry in enumerate(industry_list):
            if industry == 'Location':
                # Display the location identifier separately
                location_identifier = row[i].text_input(f"{industry} - Location {idx + 1}", value=default_value[i])
                
            else:
                # Display numeric fields for latitude, longitude, and radius
                value = row[i].number_input(f"{industry} - Location {idx + 1}", format="%.6f", value=default_value[i])
                location_data.append(value)

        # Append the location identifier to the location data
        location_data.append(location_identifier)
        user_coordinates.append(location_data)
    st.write(':orange[**User Specific Location**]')
    location_data = []

    row = st.columns(4)
    default_values = [-37.82968153089708, 145.05531534492368, 20.0,"Dan Murphy's Camberwell"]

    for i, industry in enumerate(industry_list):
        if industry == 'Location':
            # Display the location identifier separately
            location_identifier = row[i].text_input(f"{industry} - User Location", value=default_values[i])
            location_data.append(location_identifier)

        else:
            # Display numeric fields for latitude, longitude, and radius
            value = row[i].number_input(f"{industry} - User Location", format="%.6f", value=default_values[i])
            location_data.append(value)
    
    all_locations=user_coordinates+[location_data]

    if len(user_coordinates[0]) ==4:

        for user_lat, user_lon, radius_input,_ in user_coordinates:
            count_within_radius[(user_lat, user_lon)] = 0
            filtered_coordinates[(user_lat, user_lon)] = []
            filtered_coordinates[(location_data[0],location_data[1])] = []


            # filtered_coordinates[(user_lat, user_lon)]+=(filtered_coordinates[(location_data[0],location_data[1])])
            home_counts[(user_lat, user_lon)] = 0
            work_counts[(user_lat, user_lon)] = 0
            maid_ids_within_radius[(user_lat, user_lon)] = set()
            home_ids_within_radius[(user_lat, user_lon)] = set()
            work_ids_within_radius[(user_lat, user_lon)] = set()

        for index, row in df.iterrows():
            for user_lat, user_lon, radius,_ in user_coordinates:
                distance = haversine(user_lat, user_lon, row['latitude'], row['longitude'])
                home_distance = haversine(user_lat, user_lon, float(row['Home_latitude']), float(row['Home_longitude']))
                work_distance = haversine(user_lat, user_lon, float(row['Work_latitude']), float(row['Work_longitude']))

                if dist == 'Meters':
                    radius= radius/1000

                if distance <= radius:
                    count_within_radius[(user_lat, user_lon)] += 1
                    filtered_coordinates[(user_lat, user_lon)].append((row['latitude'], row['longitude']))
                    maid_ids_within_radius[(user_lat, user_lon)].add(row['maid'])

                if home_distance <= radius:
                    home_counts[(user_lat, user_lon)] += 1
                    home_ids_within_radius[(user_lat, user_lon)].add((row['Home_latitude'], row['Home_longitude']))


                if work_distance <= radius:
                    work_counts[(user_lat, user_lon)] += 1
                    work_ids_within_radius[(user_lat, user_lon)].add((row['Work_latitude'], row['Work_longitude']))

            for user_lat, user_lon, radius,_ in [location_data]:
                distance = haversine(user_lat, user_lon, row['latitude'], row['longitude'])

                if dist == 'Meters':
                    radius= radius/1000

                if distance <= radius:
                    filtered_coordinates[(location_data[0],location_data[1])].append((row['latitude'], row['longitude']))
            

            
        unique_maid_counts = {user_coords: len(maid_ids) for user_coords, maid_ids in maid_ids_within_radius.items()}
        unique_home_counts = {user_coords: len(maid_ids) for user_coords, maid_ids in home_ids_within_radius.items()}
        unique_work_counts = {user_coords: len(maid_ids) for user_coords, maid_ids in work_ids_within_radius.items()}




    else:
        st.warning("Please Enter Only Longitude and Latitude and Radius Sequentially")

    
    # all_radius = [sublist[2] for sublist in user_coordinates]
    for idx, i in enumerate(user_coordinates):

    # for radius in all_radius:
        if dist == 'Meters':
            i[2]= i[2]/1000
        filtered_df = df[df.apply(lambda row: haversine(i[0], i[1], row['latitude'], row['longitude']) <= i[2], axis=1)]
        if len(filtered_df) !=0:
            filtered_df[['Home_latitude', 'Home_longitude']] = filtered_df['homegeohash9'].apply(decode_geohash)
            filtered_df[['Work_latitude', 'Work_longitude']] = filtered_df['workgeohash'].apply(decode_geohash)

        filtered_df['Distance_To_Home (Km)'] = filtered_df.apply(lambda row:
            haversine(i[0], i[1], float(row['Home_latitude']), float(row['Home_longitude']))
            if pd.notna(row['Home_latitude']) and pd.notna(row['Home_longitude']) else None, axis=1)
        filtered_df['Distance_To_WorkPlace (Km)'] = filtered_df.apply(lambda row:
            haversine(i[0], i[1], float(row['Work_latitude']), float(row['Work_longitude']))
            if pd.notna(row['Work_latitude']) and pd.notna(row['Work_longitude']) else None, axis=1)

        unique_maid = filtered_df['maid'].unique()
        items[f'loc{idx + 1}'] = unique_maid

    # st.write(filtered_coordinates.values())
        
    count_within_radius_df = pd.DataFrame(list(count_within_radius.items()), columns=['coordinates', 'Total_record_counts'])
    unique_maid_within_radius_df = pd.DataFrame(list(unique_maid_counts.items()), columns=['coordinates4', 'Total_unique_maid_counts'])
    home_counts_df = pd.DataFrame(list(unique_home_counts.items()), columns=['coordinates2', 'Unique_home_counts'])
    # work_counts_df = pd.DataFrame(list(work_counts.items()), columns=['coordinates3', 'work_counts'])
    work_counts_df = pd.DataFrame(list(unique_work_counts.items()), columns=['coordinates3', 'Unique_work_counts'])

    loc=pd.DataFrame(item for item in items.keys() if item !='user_loc')
    count_df = pd.concat([loc,count_within_radius_df, unique_maid_within_radius_df,home_counts_df, work_counts_df], axis=1).drop(columns=['coordinates2','coordinates3','coordinates4']).rename(columns={0: 'Locations'})


    # Display the count DataFrame
    # st.write("Count within radius for each location:")
    # st.write(count_df)
    # Create a counter for all items
    all_items_counter = Counter([item for sublist in items.values() for item in sublist])
    # st.write({key: value for key, value in all_items_counter.items() if value == 1})

    # st.write('items',items)

    # Find records that are unique to each location
    unique_records_per_location = {}
    for location, records in items.items():
        location_counter = Counter(records)
        unique_records_per_location[location] = [item for item in records if location_counter[item] == 1 and all_items_counter[item] == 1]
        # unique_records_per_location[location] = [item for item in records ]


    locations_list = []
    common_items_list = []
    item_count_list = []
    # Print unique records for each location
    for location, unique_records in unique_records_per_location.items():
        locations_list.append(location)
        common_items_list.append(unique_records)
        item_count_list.append(len(unique_records))
    # Create the DataFrame
    df_common_items_counts = pd.DataFrame({
        'Locations': locations_list,
        'Unique Items': common_items_list,
        'only selected location visited': item_count_list
    })

    # df_common_items_counts=df_common_items_counts[df_common_items_counts['Count']>0]
    # df_common_items_counts=df_common_items_counts
    df_common_items_counts = count_df.merge(df_common_items_counts,on='Locations',how='inner').drop(columns=['Unique Items'])

    # Display the DataFrame
    st.write(':orange[**Unique records for each competitor location**]',df_common_items_counts)

    #-----------------------------------------------------------



    count_within_radius = {}
    # filtered_coordinates = {}
    maid_ids_within_radius={}
    count_within_radius[(location_data[0], location_data[1])] = 0
    # filtered_coordinates[(location_data[0], location_data[1])] = []

    maid_ids_within_radius[(location_data[0], location_data[1])] = set()

    for index, row in df.iterrows():
        distance = haversine(location_data[0], location_data[1], row['latitude'], row['longitude'])
            # home_distance = haversine(user_lat, user_lon, float(row['Home_latitude']), float(row['Home_longitude']))
            # work_distance = haversine(user_lat, user_lon, float(row['Work_latitude']), float(row['Work_longitude']))

        if dist == 'Meters':
            radius= radius/1000

        if distance <= location_data[2]:
            count_within_radius[(location_data[0], location_data[1])] += 1
            # filtered_coordinates[(location_data[0], location_data[1])].update((row['latitude'], row['longitude']))
            maid_ids_within_radius[(location_data[0], location_data[1])].add(row['maid'])

    
    items['user_loc'] = maid_ids_within_radius[(location_data[0], location_data[1])]

    # Initialize a dictionary to store common items and their counts for each combination of locations
    common_items_counts = {}

    # Calculate common items and their counts for each combination of locations
    for combination_size in range(2, len(items) + 1):
        for combination in combinations(items.keys(), combination_size):
            common_items = set.intersection(*(set(items[loc]) for loc in combination))
            common_items_key = tuple(sorted(combination))
            if common_items_key not in common_items_counts:
                common_items_counts[common_items_key] = Counter()
            common_items_counts[common_items_key].update(common_items)
    # Initialize lists to store data
    locations_list = []
    common_items_list = []
    item_count_list = []

    # Iterate through the common_items_counts dictionary
    for combination, counts in common_items_counts.items():
        # Extract the locations, common items, and their counts
        locations = ', '.join(combination)
        common_items = ', '.join([item for item, count in counts.items() if count > 0])
        item_count = sum(counts[item] for item in counts if counts[item] > 0)
        
        # Append the data to the lists
        locations_list.append(locations)
        common_items_list.append(common_items)
        item_count_list.append(item_count)

    # Create the DataFrame
    df_common_items_counts = pd.DataFrame({
        'Locations': locations_list,
        'Common Items': common_items_list,
        'Device count Visited Both Locations': item_count_list
    })

    df_common_items_counts = df_common_items_counts[df_common_items_counts['Locations'].str.contains('user_loc')]
    df_common_items_counts = df_common_items_counts[df_common_items_counts['Locations'].apply(lambda x: len(x.split(', '))) == 2]

    # Display the DataFrame
    st.write('Common records for multiple locations',df_common_items_counts[['Locations','Device count Visited Both Locations']])

else:
    st.warning("Please Enter No of Locations")



# st.write(filtered_coordinates[(location_data[0], location_data[1])])
# Analyzing_Method=st.radio('Select one for Analyze',['Single Location','Multiple Location'])

filtered_dfs = []
# Create an empty list to store histogram traces
histogram_traces = []
histogram_traces1 = []

day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
locations=[]
for loc in all_locations:
    locations.append(loc[3])

# st.write(locations)
# first_five_records = {key: value[:5] for key, value in filtered_coordinates.items()}

# Print the new dictionary containing the first five records
# st.write(first_five_records.items())
# Iterate over each list of coordinates in filtered_coordinates
for i, coordinates_list in enumerate(filtered_coordinates.values(), start=0):
    # Extract latitude and longitude values from each tuple
    latitudes = [coord[0] for coord in coordinates_list]
    longitudes = [coord[1] for coord in coordinates_list]
    # Filter DataFrame based on latitude and longitude
    filtered_df_location = df[(df['latitude'].isin(latitudes)) & (df['longitude'].isin(longitudes))]
    
    
    # Create a histogram trace for the current location
    histogram_trace = go.Histogram(
        x=filtered_df_location['datetimestamp'].dt.hour,
        name=locations[i],  # Dynamically generate trace name
        opacity=1
    )
        # Create a histogram trace for the current location
    histogram_trace1 = go.Histogram(
        x=filtered_df_location['datetimestamp'].dt.dayofweek.map(lambda x: day_names[x]),
        name=locations[i],  # Dynamically generate trace name
        opacity=1
    )
    
    # Append the histogram trace to the list
    histogram_traces.append(histogram_trace)
    histogram_traces1.append(histogram_trace1)



# Create the clustered bar chart
fig = go.Figure(data=histogram_traces)
fig1 = go.Figure(data=histogram_traces1)


# Update layout
fig.update_layout(
    barmode='group',  # Set barmode to group for clustered bars
    xaxis={'title': 'Hour of Day'},
    yaxis={'title': 'Total Count'},
    title='Clustered Histogram of Hour Variations for each location',
    plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
    paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent paper background
    xaxis_showgrid=False,  # Hide gridlines on x-axis
    yaxis_showgrid=False,  # Hide gridlines on y-axis
    width=1250,  # Set the width of the figure
    height=700,  # Set the height of the figure
)

# Display the clustered bar chart
st.plotly_chart(fig)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Update layout
fig1.update_layout(
    barmode='group',  # Set barmode to group for clustered bars
    xaxis={'title': 'Day', 'categoryorder': 'array', 'categoryarray': day_order},
    yaxis={'title': 'Total Count'},
    title='Clustered Histogram of Day Variations for each location',
    plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
    paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent paper background
    xaxis_showgrid=False,  # Hide gridlines on x-axis
    yaxis_showgrid=False,  # Hide gridlines on y-axis
    width=1250,  # Set the width of the figure
    height=700,  # Set the height of the figure
)

# Display the clustered bar chart
st.plotly_chart(fig1)