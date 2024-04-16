
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static
import folium
import plotly.express as px
import numpy as np

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

st.title("Additional Insights")

df = pd.read_csv('10000_Movements.csv', sep=",").dropna(subset=['latitude', 'longitude'])
df['datetimestamp'] = pd.to_datetime(df['datetimestamp'])

df['year'] = df['datetimestamp'].dt.year
df['month'] = df['datetimestamp'].dt.month
df['day'] = df['datetimestamp'].dt.day
df['day_of_week'] = df['datetimestamp'].dt.dayofweek + 1  # Adjusting dayofweek to start from Monday as 1
df['hour'] = df['datetimestamp'].dt.hour

day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
def map_day_name(day_of_week):
    day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    return day_names[day_of_week - 1]

# Apply the function to create the 'day_name' column
df['day_name'] = df['day_of_week'].apply(map_day_name)

def map_day_type(day_of_week):
    if day_of_week <= 5 and day_of_week > 1:
        return 'Weekday'
    else:
        return 'Weekend'

# Apply the function to create the 'day_type' column
df['day_type'] = df['day_of_week'].apply(map_day_type)


start_date = pd.Timestamp('2023-12-01')
end_date = pd.Timestamp('2023-12-31')

selected_start_date = st.sidebar.date_input("Select Start Date", start_date)
selected_end_date = st.sidebar.date_input("Select End Date", end_date)

dist = st.radio("Select Distance Unit", ["Meters","Kilometers"])
selected_start_date = pd.to_datetime(selected_start_date)
selected_end_date = pd.to_datetime(selected_end_date)
df = df[(df['datetimestamp'] >= selected_start_date) & (df['datetimestamp'] <= selected_end_date)]

user_input_lat = st.sidebar.text_input("Enter a User latitude:", value="-37.82968153089708")
user_input_lon = st.sidebar.text_input("Enter a User longitude :", value="145.05531534492368")


center = (-37.82968153089708, 145.05531534492368)
if user_input_lat =='-37.82968153089708' and user_input_lon=='145.05531534492368':
    st.sidebar.text("Dan Murphy's Camberwell")

if dist == 'Kilometers':
    radius_input = st.slider("Select radius (in kilometers):", min_value=1, max_value=100, value=10)

elif dist == 'Meters':
    radius_input = st.slider("Select radius (in Meters):", min_value=1, max_value=1000, value=15)
    radius_input=radius_input/1000

user_lat = float(user_input_lat)
user_lon = float(user_input_lon)

EARTH_RADIUS = 6371.0

# Define a function to calculate the distance between two points using Haversine formula
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = EARTH_RADIUS * c
    return distance

# Calculate the distance between user's location and each point in the DataFrame
df['distance'] = haversine_distance(user_lat, user_lon, df['latitude'], df['longitude'])

# Filter the DataFrame based on the distance condition
count_within_radius_df = df[df['distance'] <= radius_input]

# Count the number of points within the specified radius
count_within_radius = len(count_within_radius_df)

st.text(f"count within radius {count_within_radius}")


mymap = folium.Map(location=[user_lat, user_lon], zoom_start=10)

for index, row in df.iterrows():
    folium.CircleMarker(location=[row['latitude'], row['longitude']], radius=2, color='blue', fill=True, fill_color='blue',
                        fill_opacity=1).add_to(mymap)
    
folium.CircleMarker(location=[user_lat, user_lon], radius=4, color='red', fill=True, fill_color='red',
                        fill_opacity=1).add_to(mymap)
folium.Circle(
    location=(user_lat,user_lon),
    radius=radius_input*1000,
    color='green',
    fill=True,
    fill_opacity=0.4,
    ).add_to(mymap)
col1,col2=st.columns((0.6,0.3))
with col1:
    folium_static(mymap)
with col2:
    with st.expander('About',expanded=True):
        st.write('This Map is plotting all the records for seleted date range with user radius')






st.markdown("Frequent Visition")
count_within_radius_df_maid = count_within_radius_df.groupby('maid').size().reset_index(name='count').sort_values(by='count', ascending=False)

# Calculate frequency by maid and day
freq_day = count_within_radius_df.groupby(['maid', 'day']).size().reset_index(name='count').sort_values(by='count')
with st.expander('View grouped maids'):
    col1,col2=st.columns((2))
    with col1:
        st.write('All days freuency',count_within_radius_df_maid)
    with col2:
        st.write('Days wise freuency',freq_day)

pandas_df = count_within_radius_df

pandas_unique = pandas_df.drop_duplicates(subset=['maid'])

day_pattern_fig = px.bar(pandas_df.groupby(['day']).size().reset_index(name='count'), 
                            x='day', y='count', color='count',
                            title='Daily Pattern', labels={'count': 'Count'}, text='count')
day_pattern_fig.update_layout(xaxis=dict(showgrid=False),
                                  yaxis=dict(showgrid=False))

hourly_pattern_fig = px.bar(pandas_df.groupby(['hour']).size().reset_index(name='count'), 
                            x='hour', y='count', color='count',
                            title='Hourly Pattern', labels={'count': 'Count'}, text='count')
hourly_pattern_fig.update_layout(xaxis=dict(showgrid=False),
                                  yaxis=dict(showgrid=False))
age_pattern_fig = px.bar(pandas_unique.groupby(['Age_Range']).size().reset_index(name='count'), 
                            x='Age_Range', y='count', color='count',
                            title='Age_Range Variation', labels={'count': 'Count'}, text='count')
age_pattern_fig.update_layout(xaxis=dict(showgrid=False),
                                  yaxis=dict(showgrid=False))
gender_pattern_fig = px.pie(pandas_unique, names='Gender', title='Gender Variation')

day_pattern_fig.update_layout(barmode='group')

colors = {'Monday': 'rgb(31, 119, 180)', 'Tuesday': 'rgb(31, 119, 180)', 'Wednesday': 'rgb(31, 119, 180)',
          'Thursday': 'rgb(31, 119, 180)', 'Friday': 'rgb(31, 119, 180)',
          'Saturday': 'rgb(255, 127, 14)', 'Sunday': 'rgb(255, 127, 14)'}

pandas_df['day_name'] = pd.Categorical(pandas_df['day_name'], categories=list(colors.keys()), ordered=True)
weekday_weekend_fig = px.bar(pandas_df.groupby('day_name').size().reset_index(name='count'), 
                             x='day_name', y='count', color='day_name',
                             color_discrete_map=colors, title='Weekday Vs Weekend Pattern',
                             labels={'count': 'Percentage'}, text='count')
weekday_weekend_fig.update_layout(xaxis=dict(showgrid=False),
                                  yaxis=dict(showgrid=False))

stack_age_pattern_fig = px.bar(pandas_unique.groupby(['Age_Range', 'Gender']).size().reset_index(name='Count'), x='Age_Range', y='Count', color='Gender', 
                         title='Age Range Variation by Gender', labels={'Count': 'Count'})
stack_age_pattern_fig.update_layout(xaxis=dict(showgrid=False),
                                  yaxis=dict(showgrid=False))
weekday_weekend_fig.update_layout(barmode='group')

st.plotly_chart(day_pattern_fig)
st.plotly_chart(hourly_pattern_fig)
st.plotly_chart(weekday_weekend_fig)
st.plotly_chart(age_pattern_fig)
st.plotly_chart(gender_pattern_fig)
st.plotly_chart(stack_age_pattern_fig)


heatmap_data = pandas_df.pivot_table(index='day_name', columns='hour', aggfunc='size')

# Reorder the rows of the heatmap data to match the order in the image
heatmap_data = heatmap_data.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

import plotly.graph_objects as go

# Convert the heatmap_data DataFrame to a list of lists
heatmap_values = heatmap_data.values.tolist()

# Create the heatmap using Plotly Graph Objects
fig = go.Figure(data=go.Heatmap(
                   z=heatmap_values,
                   x=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24'],
                   y=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                   colorscale='YlOrRd'))

# Update layout
fig.update_layout(title='Visitation Insights',
                  xaxis_title='Hour of the Day',
                  yaxis_title='Day of the Week')

# Show the plot
st.plotly_chart(fig)


hour_difference_df = count_within_radius_df.groupby(['maid', 'day']).agg(
    {'datetimestamp': lambda x: (x.max() - x.min()).total_seconds() / 3600}
).reset_index().rename(columns={'datetimestamp': 'hour_difference'})

# Filter out rows with hour_difference <= 0
hour_difference_df = hour_difference_df[hour_difference_df['hour_difference'] > 0]

# Sort by hour_difference
hour_difference_df = hour_difference_df.sort_values(by='hour_difference')

hour_difference_pandas = hour_difference_df

# Define bin edges
bin_edges = [0, 2, 5, 24]

# Categorize minute differences into bins
hour_difference_pandas['bin'] = pd.cut(hour_difference_pandas['hour_difference'], bins=bin_edges, labels=['0-2', '2-5', '5-24'])

# Aggregate counts by bin
agg_df = hour_difference_pandas.groupby('bin').size().reset_index(name='count')

# Create histogram using Plotly Express
histogram_fig = px.bar(agg_df, x='bin', y='count', 
                       title='Histogram of Hour Spending',
                       labels={'bin': 'Hour Spending Range', 'count': 'Frequency'})
histogram_fig.update_layout(xaxis=dict(showgrid=False),
                                  yaxis=dict(showgrid=False))

col1,col2=st.columns((0.6,0.3))
with col1:
    st.plotly_chart(histogram_fig)
with col2:
    st.write(hour_difference_df)


# Merge dataframes based on 'maid' column
count_within_radius_df_filtered = pd.merge(df, count_within_radius_df, on='maid', how='inner', indicator=True)
count_within_radius_df_filtered = count_within_radius_df_filtered[count_within_radius_df_filtered['_merge'] == 'both']

# Drop the indicator column
count_within_radius_df_filtered = count_within_radius_df_filtered.drop('_merge', axis=1)
# Select latitude and longitude columns
lon_lat = count_within_radius_df_filtered[['latitude_x', 'longitude_x','maid']].sort_values('maid')

lon_lat = lon_lat.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame for better visualization
# Create a scatter plot using Plotly Express
fig = px.scatter(lon_lat, x='longitude_x', y='latitude_x', title='Plot of Latitude and Longitude who visited in given area')
fig.update_traces(marker=dict(size=8, opacity=0.5, color='blue'))
fig.update_layout(xaxis_title='Longitude', yaxis_title='Latitude', plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig)

from math import radians, sin, cos, sqrt, atan2

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
    distance = R * c*1000
    return distance

# Collect latitude and longitude values from the DataFrame

# Convert list of Row objects to Pandas DataFrame

# Create a Folium map centered at the first coordinate
mymap_filtered = folium.Map(location=[user_lat, user_lon], zoom_start=10)

# Define he radius of the circle
# circle_radius = 5000  # in meters
# Add markers for all coordinates to the map
for _, row in lon_lat.iterrows():
    # Calculate the distance from the user's location to the current point
    distance = haversine(user_lat, user_lon, row['latitude_x'], row['longitude_x'])
    if distance <= radius_input*1000:
        # Add marker inside the circle with blue color
        folium.CircleMarker(location=[row['latitude_x'], row['longitude_x']], radius=2, color='blue', fill=True, fill_color='blue',
                            fill_opacity=1).add_to(mymap_filtered)
    else:
        # Add marker outside the circle with orange color
        folium.CircleMarker(location=[row['latitude_x'], row['longitude_x']], radius=2, color='brown', fill=True, fill_color='brown',fill_opacity=1).add_to(mymap_filtered)

# Add the user's location marker
folium.Marker(location=[user_lat, user_lon], radius=4, color='red', fill=True, fill_color='red',
                    fill_opacity=1).add_to(mymap_filtered)

# Add the circle representing the radius
folium.Circle(location=(user_lat, user_lon), radius=radius_input*1000, color='green', fill=True, fill_opacity=0.4).add_to(mymap_filtered)

col1,col2=st.columns((0.6,0.3))
with col1:
    folium_static(mymap_filtered)
with col2:
    with st.expander('About',expanded=True):
        st.write('This Map is plotting who visited the place and their other visiting points in seleted date range with user radius')