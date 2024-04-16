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
from shapely.geometry import Point, Polygon
from streamlit_folium import folium_static
from collections import Counter

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

st.title("🎯Location Insights")
st.markdown(':green[**These all points are representing based on all device Mobile Advertisement Id**]')


# Function to calculate Haversine distance between two coordinates
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
    
# Function to generate points along the circumference of a circle
def generate_circle_points(center_lat, center_lon, radius, num_points=100):
    circle_points = []
    for i in range(num_points):
        angle = 2 * radians(i * (360 / num_points))
        lat = center_lat + (radius / 111.32) * sin(angle)
        lon = center_lon + (radius / (111.32 * cos(center_lat))) * cos(angle)
        circle_points.append((lat, lon))
    return circle_points
def diagrams(filtered_df):
    day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

    fig1 = px.histogram(filtered_df, x=filtered_df['datetimestamp'].dt.hour, nbins=24, labels={'datetimestamp': 'Hour of Day', 'count': 'Count'})
    filtered_df['day_of_week'] = filtered_df['datetimestamp'].dt.dayofweek.map(lambda x: day_names[x])
    fig2 = px.histogram(filtered_df, x=filtered_df['day_of_week'], nbins=7,
                labels={'day_of_week': 'Day of the Week', 'count': 'Count'},
                category_orders={'day_of_week': day_names})
    fig1.update_traces(marker_color='yellow', opacity=0.7)

    # Set background color to be transparent
    fig1.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'xaxis': {'showgrid': False,'title': 'Hour'},
        'yaxis': {'showgrid': False,'title': 'Total Count'},
    })
    fig2.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'xaxis': {'showgrid': False,'title': 'Days'},
        'yaxis': {'showgrid': False,'title': 'Total Count'},
    })

    # with col1:
    st.write("Histogram of Hour Variations")
    st.plotly_chart(fig1)
    st.write("Histogram of Day Variations")
    st.plotly_chart(fig2)
start_date = pd.Timestamp('2023-12-01')
end_date = pd.Timestamp('2023-12-31')


def more_insights(count_within_radius_df,user_lat,user_lon):
    st.markdown("Frequent Visition")
    count_within_radius_df_maid = count_within_radius_df.groupby('maid').size().reset_index(name='count').sort_values(by='count', ascending=False)
    count_within_radius_df = count_within_radius_df.rename(columns={'day': 'date'})
    # Calculate frequency by maid and day
    freq_day = count_within_radius_df.groupby(['maid', 'date']).size().reset_index(name='count').sort_values(by='count')
    with st.expander('View grouped maids'):
        col1,col2=st.columns((2))
        with col1:
            st.write('All days freuency',count_within_radius_df_maid)
            st.write("""
                    - :orange[**This represents frequency of all the devices within date ranges**]""")
        with col2:
            st.write('Days wise freuency',freq_day)
            st.write("""
                    - :orange[**This represents frequency of all the devices within same day**]""")

    pandas_df = count_within_radius_df

    pandas_unique = pandas_df.drop_duplicates(subset=['maid'])

    day_pattern_fig = px.bar(pandas_df.groupby(['date']).size().reset_index(name='count'), 
                                x='date', y='count', color='count',
                                title='Daily Pattern', labels={'count': 'Count'}, text='count')
    day_pattern_fig.update_layout(xaxis=dict(showgrid=False),
                                    yaxis=dict(showgrid=False))

    hourly_pattern_fig = px.bar(pandas_df.groupby(['hour']).size().reset_index(name='count'), 
                                x='hour', y='count', color='count',
                                title='Hourly Pattern', labels={'count': 'Count'}, text='count')
    hourly_pattern_fig.update_layout(xaxis=dict(showgrid=False),
                                    yaxis=dict(showgrid=False))
    # age_pattern_fig = px.bar(pandas_unique.groupby(['Age_Range']).size().reset_index(name='count'), 
    #                             x='Age_Range', y='count', color='count',
    #                             title='Age_Range Variation', labels={'count': 'Count'}, text='count')
    # age_pattern_fig.update_layout(xaxis=dict(showgrid=False),
    #                                 yaxis=dict(showgrid=False))
    # gender_pattern_fig = px.pie(pandas_unique, names='Gender', title='Gender Variation')

    day_pattern_fig.update_layout(barmode='group')

    colors = {'Monday': 'rgb(31, 119, 180)', 'Tuesday': 'rgb(31, 119, 180)', 'Wednesday': 'rgb(31, 119, 180)',
            'Thursday': 'rgb(31, 119, 180)', 'Friday': 'rgb(31, 119, 180)',
            'Saturday': 'rgb(255, 127, 14)', 'Sunday': 'rgb(255, 127, 14)'}

    pandas_df['day_name'] = pd.Categorical(pandas_df['day_name'], categories=list(colors.keys()), ordered=True)
    weekday_weekend_fig = px.bar(pandas_df.groupby('day_name').size().reset_index(name='count'), 
                                x='day_name', y='count', color='day_name',
                                color_discrete_map=colors, title='Weekday Vs Weekend Pattern',
                                labels={'count': 'count'}, text='count')
    weekday_weekend_fig.update_layout(xaxis=dict(showgrid=False),
                                    yaxis=dict(showgrid=False))

    # stack_age_pattern_fig = px.bar(pandas_unique.groupby(['Age_Range', 'Gender']).size().reset_index(name='Count'), x='Age_Range', y='Count', color='Gender', 
                            # title='Age Range Variation by Gender', labels={'Count': 'Count'})
    # stack_age_pattern_fig.update_layout(xaxis=dict(showgrid=False),
                                    # yaxis=dict(showgrid=False))
    weekday_weekend_fig.update_layout(barmode='group')

    col1,col2=st.columns((0.6,0.3))

    with col1:
        st.plotly_chart(day_pattern_fig)
    # st.plotly_chart(age_pattern_fig)
    # st.plotly_chart(gender_pattern_fig)
    # st.plotly_chart(stack_age_pattern_fig)
    with col2:
        with st.expander('About', expanded=True):
            st.write("""
                :orange[**The chart provides a visual representation of the daily pattern of occurrences, 
                     highlighting any trends or fluctuations in the data over the course of the month.**]""")

    col1,col2=st.columns((0.6,0.3))

    with col1:
        st.plotly_chart(hourly_pattern_fig)
    with col2:
        with st.expander('About', expanded=True):
            st.write("""
                :orange[**The chart provides a visual representation of the hourly pattern of occurrences, 
                     highlighting any trends or fluctuations throughout the day.**]""")

    col1,col2=st.columns((0.6,0.3))

    with col1:
        st.plotly_chart(weekday_weekend_fig)
    with col2:
        with st.expander('About', expanded=True):
            st.write("""
                :orange[**The chart provides a visual representation of the day wise pattern of occurrences, 
                     highlighting any trends or fluctuations throughout the week.**]""")



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

    col1,col2=st.columns((0.6,0.3))

    with col1:
        st.plotly_chart(fig)
    with col2:
        with st.expander('About', expanded=True):
            st.write("""
                :orange[**The provided heatmap visualization
                    represents the visitation patterns throughout the week across different hours of the day.**]""")



    hour_difference_df = count_within_radius_df.groupby(['maid', 'date']).agg(
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
        # st.write(hour_difference_df)
        with st.expander('About', expanded=True):
            st.write("""
                :orange[**The histogram displays the distribution of hour spending ranges, providing insights into the variability of spending durations across same days.**]""")
            st.write("""-:orange[**The "0-2", "2-5", "5-24" hour spending range represents instances where the difference between the maximum and minimum spending time within the same day.**]""")


    # Merge dataframes based on 'maid' column
    count_within_radius_df_filtered = pd.merge(df, count_within_radius_df, on='maid', how='inner', indicator=True)
    count_within_radius_df_filtered = count_within_radius_df_filtered[count_within_radius_df_filtered['_merge'] == 'both']

    # Drop the indicator column
    count_within_radius_df_filtered = count_within_radius_df_filtered.drop('_merge', axis=1)
    # Select latitude and longitude columns
    lon_lat = count_within_radius_df_filtered[['latitude_x', 'longitude_x','maid']].sort_values('maid')
    lon_lat['Locations']='Other Visiting Points'

    lon_lat = lon_lat.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame for better visualization
    fig = px.scatter(lon_lat, x='longitude_x', y='latitude_x',color='Locations',title='Plot of Latitude and Longitude who visited in given area')
    fig.update_traces(marker=dict(size=8, opacity=0.5, color='blue'))
    fig.update_layout(xaxis_title='Longitude', yaxis_title='Latitude', plot_bgcolor='rgba(0,0,0,0)', showlegend=True)

    # Create another scatter plot for count_within_radius_df
    fig.add_scatter(x=count_within_radius_df['longitude'], 
                            y=count_within_radius_df['latitude'], 
                            mode='markers', 
                            marker=dict(size=8, opacity=0.5, color='red'),
                            name='Points within Area')
    

    # Update layout
    fig.update_layout(xaxis_title='Longitude', yaxis_title='Latitude', plot_bgcolor='rgba(0,0,0,0)', showlegend=True)

    col1,col2=st.columns((0.6,0.3))

    with col1:
        st.plotly_chart(fig)

    with col2:
        # st.write(hour_difference_df)
        with st.expander('About', expanded=True):
            # st.write("""
                # :orange[**The plot represents the latitude and longitude coordinates of individuals who visited a specific area, along with other places of interest they visited. Each point on the plot corresponds to a location where a customer visited, and the clustering of points indicates areas with higher customer visitation.**]""")
            st.write("""-:orange[**This visualization represents the latitude and longitude coordinates of individuals who visited a specific area, along with other places of interest they visited.Provides insights into customer behavior and preferences regarding visitation patterns. It helps identify popular places of interest and areas with high customer traffic, which can be valuable for businesses in understanding customer preferences, optimizing marketing strategies, and making informed decisions related to location-based services or targeted advertising.**]""")

    return lon_lat


st.sidebar.markdown('<p style="color: red;">Select Date Range</p>', unsafe_allow_html=True)

# # Allow user to pick start and end dates
selected_start_date = st.sidebar.date_input("Select Start Date", start_date)
selected_end_date = st.sidebar.date_input("Select End Date", end_date)


# Define the HTML content with icons and labels
st.markdown("""
    <style>
    .stRadio [role=radiogroup]{
        align-items: center;
        justify-content: center;
    }
    </style>
""",unsafe_allow_html=True)

options = ["Search by Radius", "Search by Polygon"]

options=st.radio("Select Search Option", options=options, horizontal=True)

if options == 'Search by Radius':
    dist = st.radio("Select Distance Unit", ["Meters","Kilometers"],horizontal=True)

    # # Convert date inputs to datetime objects
    selected_start_date = pd.to_datetime(selected_start_date)
    selected_end_date = pd.to_datetime(selected_end_date)

    @st.cache_data
    def load(selected_start_date,selected_end_date):
        df = pd.read_csv('10000_Movements.csv', sep=",").dropna(subset=['latitude', 'longitude'])
        df['datetimestamp'] = pd.to_datetime(df['datetimestamp'])

        df = df[(df['datetimestamp'] >= selected_start_date) & (df['datetimestamp'] <= selected_end_date)]
        df['year'] = df['datetimestamp'].dt.year
        df['month'] = df['datetimestamp'].dt.month
        df['day'] = df['datetimestamp'].dt.day
        df['day_of_week'] = df['datetimestamp'].dt.dayofweek + 1  # Adjusting dayofweek to start from Monday as 1
        df['hour'] = df['datetimestamp'].dt.hour

        return df

    df=load(selected_start_date,selected_end_date)
    # st.button("Rerun")

    day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    def map_day_name(day_of_week):
        day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        return day_names[day_of_week - 1]

    # Apply the function to create the 'day_name' column
    df['day_name'] = df['day_of_week'].apply(map_day_name)

    def map_day_type(day_of_week):
        if day_of_week <= 5 and day_of_week >= 1:
            return 'Weekday'
        else:
            return 'Weekend'

    # Apply the function to create the 'day_type' column
    df['day_type'] = df['day_of_week'].apply(map_day_type)


    st.sidebar.markdown(f"<font color='orange'><b>Number of records within Date Range: {len(df)}</b></font>", unsafe_allow_html=True)


    st.sidebar.write("----------------")

    st.sidebar.markdown('<p style="color: red;">Select a Location</p>', unsafe_allow_html=True)
    # st.sidebar.text("Dan Murphy's Camberwell")

    placeholder = st.sidebar.empty()
    user_input_lat = st.sidebar.text_input("Enter a User latitude:", value="-37.82968153089708")
    user_input_lon = st.sidebar.text_input("Enter a User longitude :", value="145.05531534492368")


    center = (float(user_input_lat), float(user_input_lon))
    if user_input_lat =='-37.82968153089708' and user_input_lon=='145.05531534492368':
        placeholder.markdown("Location: Dan Murphy's Camberwell")

    if dist == 'Kilometers':
        radius_input = st.slider("Select radius (in kilometers):", min_value=1, max_value=100, value=10)
        unit = 'km'

    elif dist == 'Meters':
        radius_input = st.slider("Select radius (in Meters):", min_value=1, max_value=1000, value=15)
        radius_m=radius_input
        radius_input=radius_input/1000
        unit = 'm'


    # Process user input
    if user_input_lat and user_input_lon :
        user_lat = float(user_input_lat)
        user_lon = float(user_input_lon)

        # Create a folium map centered on the user-specified location
        m = folium.Map(location=[user_lat, user_lon], zoom_start=17)

        # Plot sample data as blue points
        for lat, lon in zip(df['latitude'], df['longitude']):
            color = 'blue'
            folium.CircleMarker(location=[lat, lon], radius=2, color=color, fill=True, fill_color=color,
                                fill_opacity=1).add_to(m)

        # Highlight the user-specified location as a red point
        folium.CircleMarker(location=[user_lat, user_lon], radius=4, color='red', fill=True, fill_color='red',
                            fill_opacity=1).add_to(m)


        # Perform radius search and count points within the specified radius
        count_within_radius = 0
        for index, row in df.iterrows():
            distance = haversine(user_lat, user_lon, row['latitude'], row['longitude'])
            if distance <= radius_input:
                count_within_radius += 1

        if dist == 'Meters':
            st.text(f"Total number of  devices within {radius_m} {unit} radius: {count_within_radius}")
        else:
            st.text(f"Total number of  devices within {radius_input} {unit} radius: {count_within_radius}")


        # Draw a circle around the user-specified location
        # circle_points = generate_circle_points(user_lat, user_lon, radius_input)
        # folium.PolyLine(circle_points, color='green', weight=2.5, opacity=1).add_to(m)
        folium.Circle(
        location=center,
        radius=radius_input*1000,
        color='green',
        fill=True,
        fill_opacity=0.4,
        ).add_to(m)
        filtered_df = df[df.apply(lambda row: haversine(user_lat, user_lon, row['latitude'], row['longitude']) <= radius_input, axis=1)]
        filtered_df_maid_unique_count = filtered_df['maid'].nunique()

        if filtered_df_maid_unique_count ==0:
            filtered_df_maid_unique_count=0
            filtered_df_homegeo_unique_count=0
            filtered_df_workgeo_unique_count=0
            filtered_df_workgeo_and_home_unique_count=0

        if filtered_df_maid_unique_count !=0:
            filtered_df[['Home_latitude', 'Home_longitude']] = filtered_df['homegeohash9'].apply(decode_geohash)
            filtered_df[['Work_latitude', 'Work_longitude']] = filtered_df['workgeohash'].apply(decode_geohash)


            filtered_df['Distance_To_Home (Km)'] = filtered_df.apply(lambda row:
                haversine(user_lat, user_lon, float(row['Home_latitude']), float(row['Home_longitude']))
                if pd.notna(row['Home_latitude']) and pd.notna(row['Home_longitude']) else None, axis=1)
            filtered_df['Distance_To_WorkPlace (Km)'] = filtered_df.apply(lambda row:
                haversine(user_lat, user_lon, float(row['Work_latitude']), float(row['Work_longitude']))
                if pd.notna(row['Work_latitude']) and pd.notna(row['Work_longitude']) else None, axis=1)


            filtered_df_homegeo_unique_count = filtered_df[filtered_df['Distance_To_Home (Km)']<=radius_input]['homegeohash9'].nunique()
            feature_group_home = folium.FeatureGroup(name='Home Locations')
            feature_group_work = folium.FeatureGroup(name='Work Locations')
            for lat, lon in zip(filtered_df[filtered_df['Distance_To_Home (Km)']<=radius_input]['latitude'], filtered_df[filtered_df['Distance_To_Home (Km)']<=radius_input]['longitude']):
                color = 'Orange'
                folium.CircleMarker(location=[lat, lon], radius=2, color=color, fill=True, fill_color=color,
                                    fill_opacity=1).add_to(feature_group_home)


            filtered_df_workgeo_unique_count = filtered_df[filtered_df['Distance_To_WorkPlace (Km)']<=radius_input]['workgeohash'].nunique()
            for lat, lon in zip(filtered_df[filtered_df['Distance_To_WorkPlace (Km)']<=radius_input]['latitude'], filtered_df[filtered_df['Distance_To_WorkPlace (Km)']<=radius_input]['longitude']):
                color = 'green'
                folium.CircleMarker(location=[lat, lon], radius=2, color=color, fill=True, fill_color=color,
                                    fill_opacity=1).add_to(feature_group_work)
            filtered_df_workgeo_and_home_unique_count = filtered_df[(filtered_df['Distance_To_WorkPlace (Km)']<=radius_input) & (filtered_df['Distance_To_Home (Km)']<=radius_input)]['workgeohash'].nunique()

            # Add feature groups to the map
            feature_group_home.add_to(m)
            feature_group_work.add_to(m)

            # Add legend to the map
            folium.LayerControl().add_to(m)

            same_values_count = filtered_df[(filtered_df['homegeohash9'] == filtered_df['workgeohash'])].shape[0]
            if same_values_count>0:
                filtered_df_workgeo_unique_count=filtered_df_workgeo_unique_count-same_values_count

        col1,col2=st.columns((0.6,0.3))

        with col1:
            if dist == 'Kilometers':
                st.text(f"Total number of Unique devices within {radius_input}{unit} radius: {filtered_df_maid_unique_count}")
                st.text(f"Total No of devices from Home within {radius_input}{unit} radius: {filtered_df_homegeo_unique_count}")
                st.text(f"Total No of devices from WorkPlace {radius_input}{unit} radius: {filtered_df_workgeo_unique_count}")
                st.text(f"Total No of devices both Home and Workplace within {radius_input}{unit} radius: {filtered_df_workgeo_and_home_unique_count}")
            else:
                st.text(f"Total number of Unique devices within {radius_m}{unit} radius: {filtered_df_maid_unique_count}")
                st.text(f"Total No of devices from Home within {radius_m}{unit} radius: {filtered_df_homegeo_unique_count}")
                st.text(f"Total No of devices from Workplace {radius_m}{unit} radius: {filtered_df_workgeo_unique_count}")
                st.text(f"Total No of devices both Home and Workplace within {radius_m}{unit} radius: {filtered_df_workgeo_and_home_unique_count}")
        # with col2:
        #     with st.expander('About', expanded=True):
        #             if dist == 'Meters':
        #                 radius_input=int(radius_input*1000)

        #             st.write("""
        #             - :Brown[**All Devices that captured in the ploygon**]
        #             - :orange[**Home Locations**]
        #             - :green[**Work Locations**]
        #             - This all points are representing {}{} radius from {} to {}
        #             """)


        col1,col2=st.columns((0.6,0.3))

        with col1:
            folium_static(m)
        with col2:
            with st.expander('About', expanded=True):
                if dist == 'Meters':
                    radius_input=int(radius_input*1000)

                st.write("""
                - :blue[**All Devices**]
                - :orange[**Home Locations**]
                - :green[**Work Locations**]
                - This all points are representing {}{} radius from {} to {}
                """.format(radius_input,unit,selected_start_date.strftime('%Y-%m-%d'),selected_end_date.strftime('%Y-%m-%d')))
        
        # filtered_df_hr=filtered_df[['datetimestamp']].copy()
        # filtered_df_hr['hour'] = filtered_df_hr['datetimestamp'].dt.hour
        #         # Group by 'hour' and get count for each hour
        # hourly_counts = filtered_df_hr.groupby('hour').size().reset_index(name='count')

        # # Compute cumulative sum and cumulative sum percentage
        # hourly_counts['cumulative_sum'] = hourly_counts['count'].cumsum()
        # hourly_counts['cumulative_percentage'] = (hourly_counts['cumulative_sum'] / hourly_counts['count'].sum()) * 100

        # # st.write(hourly_counts)
        # total_count_70_percent = hourly_counts[hourly_counts['cumulative_percentage'] <= 70]['count'].sum()

        # st.write("Total count that contributes to 70% of cumulative percentage:", total_count_70_percent)

        # diagrams(filtered_df)

        if len(filtered_df) ==0:
            pass
        else:
            lon_lat=more_insights(filtered_df,user_lat,user_lon)
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
            mymap_filtered = folium.Map(location=[user_lat, user_lon], zoom_start=17)

            # Define he radius of the circle
            # circle_radius = 5000  # in meters
            # Add markers for all coordinates to the map
            if dist == 'Kilometers':
                radius_input = radius_input*1000 
            for _, row in lon_lat.iterrows():
                # Calculate the distance from the user's location to the current point
                distance = haversine(user_lat, user_lon, row['latitude_x'], row['longitude_x'])
                
                if distance <= radius_input:
                    # Add marker inside the circle with blue color
                    folium.CircleMarker(location=[row['latitude_x'], row['longitude_x']], radius=2, color='red', fill=True, fill_color='red',
                                        fill_opacity=1).add_to(mymap_filtered)
                else:
                    # Add marker outside the circle with orange color
                    folium.CircleMarker(location=[row['latitude_x'], row['longitude_x']], radius=2, color='blue', fill=True, fill_color='blue',fill_opacity=1).add_to(mymap_filtered)

            # Add the user's location marker
            folium.Marker(location=[user_lat, user_lon], radius=4, color='red', fill=True, fill_color='red',
                                fill_opacity=1).add_to(mymap_filtered)
            
            folium.Circle(location=(user_lat, user_lon), radius=radius_input, color='green', fill=True, fill_opacity=0.4).add_to(mymap_filtered)

            col1,col2=st.columns((0.6,0.3))
            with col1:
                folium_static(mymap_filtered)
            with col2:
                with st.expander('About',expanded=True):
                    st.write('This Map is plotting who visited the place and their other visiting points in seleted date range with user radius')


            # else:
            #     st.warning("No Records Founds")

    else:
        st.warning("Please enter both latitude and longitude values.")

else:
    selected_start_date = pd.to_datetime(selected_start_date)
    selected_end_date = pd.to_datetime(selected_end_date)

    df = pd.read_csv('10000_Movements.csv', sep=",").dropna(subset=['latitude', 'longitude'])
    df['datetimestamp'] = pd.to_datetime(df['datetimestamp'])

    df = df[(df['datetimestamp'] >= selected_start_date) & (df['datetimestamp'] <= selected_end_date)]
    st.sidebar.markdown(f"<font color='orange'><b>Number of records within Date Range: {len(df)}</b></font>", unsafe_allow_html=True)

    df['year'] = df['datetimestamp'].dt.year
    df['month'] = df['datetimestamp'].dt.month
    df['day'] = df['datetimestamp'].dt.day
    df['day_of_week'] = df['datetimestamp'].dt.dayofweek + 1  # Adjusting dayofweek to start from Monday as 1
    df['hour'] = df['datetimestamp'].dt.hour

    # st.write(df)

    day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    def map_day_name(day_of_week):
        day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        return day_names[day_of_week - 1]

    # Apply the function to create the 'day_name' column
    df['day_name'] = df['day_of_week'].apply(map_day_name)

    def map_day_type(day_of_week):
        if day_of_week <= 5 and day_of_week >= 1:
            return 'Weekday'
        else:
            return 'Weekend'

    # Apply the function to create the 'day_type' column
    df['day_type'] = df['day_of_week'].apply(map_day_type)

    no_of_polygon_points = st.text_input("Enter No of Polygon Points : ",value=4)
    if  no_of_polygon_points:
        no_of_polygon_points = int(no_of_polygon_points)
        entries = ['latitude', 'longitude']
        polygon_coordinates = []

        default_values = [[-37.82985886920226, 145.05526523266056], [-37.82968592529212, 145.05507037701463], [-37.82949052083052, 145.055352996116], [-37.82965509221098, 145.0555732778134]]
        for idx in range(no_of_polygon_points):
            location_data = []
            row = st.columns(2)
            default_value = default_values[idx] if idx < len(default_values) else [0.0, 0.0]

            for i, industry in enumerate(entries):
                value = row[i].number_input(f"{industry} - Point {idx + 1}", format="%.15f", value=default_value[i])
                location_data.append(value)


            polygon_coordinates.append(location_data)

        
        placeholder = st.sidebar.empty()

        if polygon_coordinates[0][0] ==-37.829858869202262 and polygon_coordinates[0][1]==145.055265232660560:
            placeholder.markdown("Location: Dan Murphy's Camberwell")


        total_lat = 0
        total_lon = 0
        num_vertices = len(polygon_coordinates)

        for vertex in polygon_coordinates:
            total_lat += vertex[0]
            total_lon += vertex[1]

        center_lat = total_lat / num_vertices
        center_lon = total_lon / num_vertices
        m = folium.Map(location=[center_lat, center_lon], zoom_start=17)
        polygon_shapely = Polygon(polygon_coordinates)
        for lat, lon in zip(df['latitude'], df['longitude']):
            color = 'blue'
            folium.CircleMarker(location=[lat, lon], radius=2, color=color, fill=True, fill_color=color,
                                fill_opacity=1).add_to(m)
        
        folium.Polygon(locations=polygon_coordinates, color='green', fill=True, fill_color='green', fill_opacity=0.4).add_to(m)

        # Initialize Counter for maid counts
        maid_counts = Counter()
        matching_records=[]
        # Iterate through the rows of the DataFrame
        for index, row in df.iterrows():
            # Extract latitude and longitude from the current row
            lat = row['latitude']
            lon = row['longitude']
            
            point = Point(lat, lon)
            # Check if the point is within the Shapely polygon
            if point.within(polygon_shapely):
                maid = row['maid']
                # Increment the count for the maid
                maid_counts[maid] += 1
                matching_records.append([maid,lat,lon])

        def distance_to_polygon(lat, lon):
            point = Point(lat, lon)  # Shapely Point requires (lon, lat) format
            if point.within(polygon_shapely):
                return 0
            else:
                return 1

        filtered_df=pd.DataFrame(matching_records,columns=['maid','latitude','longitude'])
        filtered_df = pd.merge(filtered_df, df[['maid','latitude','longitude','datetimestamp','workgeohash','homegeohash9','Age_Range','year','month','day','day_of_week','hour','Gender','day_name']], on=['maid','latitude','longitude'], how='inner')
        filtered_df_maid_unique_count = filtered_df['maid'].nunique()

        counts_inside_polygon=sum(list(maid_counts.values()))
        unique_count_inside_polygon=len(list(maid_counts.keys()))
        st.text(f"Number of all devices within Polygon: {counts_inside_polygon}")
        st.text(f"Number of all unique devices within Polygon {unique_count_inside_polygon}")

        if counts_inside_polygon ==0:
            filtered_df_homegeo_unique_count=0
            filtered_df_workgeo_unique_count=0
            filtered_df_workgeo_and_home_unique_count=0

        if filtered_df_maid_unique_count !=0:
            filtered_df[['Home_latitude', 'Home_longitude']] = filtered_df['homegeohash9'].apply(decode_geohash)
            filtered_df[['Work_latitude', 'Work_longitude']] = filtered_df['workgeohash'].apply(decode_geohash)



            filtered_df['Distance_To_Polygon(Home)'] = filtered_df.apply(lambda row:
                distance_to_polygon(row['Home_latitude'], row['Home_longitude']), axis=1)
            filtered_df['Distance_To_Polygon(Work)'] = filtered_df.apply(lambda row:
                distance_to_polygon(row['Work_latitude'], row['Work_longitude']), axis=1)

            filtered_df_homegeo_unique_count = filtered_df[filtered_df['Distance_To_Polygon(Home)']==0]['homegeohash9'].nunique()
            filtered_df_homegeo_unique = filtered_df[filtered_df['Distance_To_Polygon(Home)']==0]

            feature_group_home = folium.FeatureGroup(name='Home Locations')
            feature_group_work = folium.FeatureGroup(name='Work Locations')
            for lat, lon in zip(filtered_df_homegeo_unique['latitude'], filtered_df_homegeo_unique['longitude']):
                color = 'Orange'
                folium.CircleMarker(location=[lat, lon], radius=2, color=color, fill=True, fill_color=color,
                                    fill_opacity=1).add_to(feature_group_home)

            filtered_df_workgeo_unique_count = filtered_df[filtered_df['Distance_To_Polygon(Work)']==0]['workgeohash'].nunique()
            filtered_df_workgeo_unique = filtered_df[filtered_df['Distance_To_Polygon(Work)']==0]
            for lat, lon in zip(filtered_df_workgeo_unique['latitude'], filtered_df_workgeo_unique['longitude']):
                color = 'green'
                folium.CircleMarker(location=[lat, lon], radius=2, color=color, fill=True, fill_color=color,
                                    fill_opacity=1).add_to(feature_group_work)
                
            filtered_df_workgeo_and_home_unique_count=filtered_df[(filtered_df['Distance_To_Polygon(Home)']==0) & (filtered_df['Distance_To_Polygon(Work)']==0)]['workgeohash'].nunique()
            feature_group_home.add_to(m)
            feature_group_work.add_to(m)
            folium.LayerControl().add_to(m)

        else:
            pass

        same_values_count = filtered_df[(filtered_df['homegeohash9'] == filtered_df['workgeohash'])].shape[0]
        if same_values_count>0:
            filtered_df_workgeo_unique_count=filtered_df_workgeo_unique_count-same_values_count

        st.text(f"Total No of devices from Home within Polygon: {filtered_df_homegeo_unique_count}")
        st.text(f"Total No of devices from WorkPlace within Polygon: {filtered_df_workgeo_unique_count}")
        st.text(f"No of devices both Home and WorkPlace within Polygon: {filtered_df_workgeo_and_home_unique_count}")
        # sum_of_true = same_values_count.get(True, 0)
        # if sum_of_true == 0:
        #     st.write(same_values_count)
        # else:
        #     st.write("All are not unique")
        
        col1,col2=st.columns((0.6,0.3))
        with col1:
            folium_static(m)
        with col2:
            with st.expander('About', expanded=True):
                st.write("""
                - :blue[**All Devices**]
                - :orange[**Home Locations**]
                - :green[**Work Locations**]
                - This all points are representing from {} to {}
                """.format(selected_start_date.strftime('%Y-%m-%d'),selected_end_date.strftime('%Y-%m-%d')))

        # diagrams(filtered_df)
        if len(filtered_df) ==0:
            pass
        else:
            lon_lat=more_insights(filtered_df,center_lat,center_lon)
            # st.write(lon_lat)

            center_lat = total_lat / num_vertices
            center_lon = total_lon / num_vertices
            m = folium.Map(location=[center_lat, center_lon], zoom_start=17)
            polygon_shapely = Polygon(polygon_coordinates)
            for lat, lon in zip(lon_lat['latitude_x'], lon_lat['longitude_x']):
                color = 'brown'
                folium.CircleMarker(location=[lat, lon], radius=2, color=color, fill=True, fill_color=color,
                                    fill_opacity=1).add_to(m)
            folium.Polygon(locations=polygon_coordinates, color='green', fill=True, fill_color='green', fill_opacity=0.4).add_to(m)
            col1,col2=st.columns((0.6,0.3))
            with col1:
                folium_static(m)
            with col2:
                with st.expander('About',expanded=True):
                    st.write(':orange[**This Map is plotting who visited the place and their other visiting points in seleted date range with polygon**]')




    else:
        st.warning("Please Enter polygon points")


