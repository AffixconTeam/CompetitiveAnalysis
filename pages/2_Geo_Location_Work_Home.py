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

st.set_page_config(page_title='Geo Location Work Home',page_icon=':earth_asia:',layout='wide')
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
st.title("Home Work Location Insights ")
st.markdown(":green[**This page identifies individuals whose residences and workplaces fall within a specified radius. It's useful for targeting email and SMS communications to people located near a particular area.**]")
st.markdown(':green[**These all points are representing based on all device Mobile Advertisement Ids in people table**]')


def decode_geohash(geohash):
    if pd.notna(geohash):
        try:
            latitude, longitude = geohash2.decode(geohash)
            return pd.Series({'Home_latitude': latitude, 'Home_longitude': longitude})
        except ValueError:
            return pd.Series({'Home_latitude': None, 'Home_longitude': None})
        except TypeError:
            return pd.Series({'Home_latitude': geohash[0], 'Home_longitude': geohash[1]})
    else:
        return pd.Series({'Home_latitude': None, 'Home_longitude': None})
    
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

st.sidebar.markdown('<p style="color: red;">Select a Location</p>', unsafe_allow_html=True)
placeholder = st.sidebar.empty()
user_input_lat = st.sidebar.text_input("Enter a User latitude:", value="-37.82968153089708")
user_input_lon = st.sidebar.text_input("Enter a User longitude :", value="145.05531534492368")
center = (float(user_input_lat), float(user_input_lon))
if user_input_lat =='-37.82968153089708' and user_input_lon=='145.05531534492368':
    placeholder.markdown("Location: Dan Murphy's Camberwell")

radius_input = st.slider("Select radius (in kilometers):", min_value=1, max_value=100, value=3)
radius_input=int(radius_input)
user_lat=float(user_input_lat)
user_lon=float(user_input_lon)


# @st.cache_data
def main():
    df=pd.read_csv('10000_Movements.csv',usecols=['maid','homegeohash9','workgeohash'])
    df[['Home_latitude', 'Home_longitude']] = df['homegeohash9'].apply(decode_geohash)
    df[['Work_latitude', 'Work_longitude']] = df['workgeohash'].apply(decode_geohash)

    df['Distance_To_Home (Km)'] = df.apply(lambda row:
    haversine(user_lat, user_lon, float(row['Home_latitude']), float(row['Home_longitude']))
    if pd.notna(row['Home_latitude']) and pd.notna(row['Home_longitude']) else None, axis=1)
    df['Distance_To_WorkPlace (Km)'] = df.apply(lambda row:
        haversine(user_lat, user_lon, float(row['Work_latitude']), float(row['Work_longitude']))
        if pd.notna(row['Work_latitude']) and pd.notna(row['Work_longitude']) else None, axis=1)

    m = folium.Map(location=[user_lat, user_lon], zoom_start=10)
    folium.Marker(location=[user_lat, user_lon], radius=4, color='red', fill=True, fill_color='red',
                        fill_opacity=1).add_to(m)
    folium.Circle(
        location=center,
        radius=radius_input*1000,
        color='green',
        fill=True,
        fill_opacity=0.4,
        ).add_to(m)
    
    filtered_df_homegeo_unique_count = df[df['Distance_To_Home (Km)']<=radius_input]['homegeohash9'].nunique()
    feature_group_home = folium.FeatureGroup(name='Home Locations')
    feature_group_work = folium.FeatureGroup(name='Work Locations')

    home_maid=df[df['Distance_To_Home (Km)']<=radius_input]['maid'].unique().tolist()
    work_maid=df[df['Distance_To_WorkPlace (Km)']<=radius_input]['maid'].unique().tolist()
    common_items = len(list(filter(lambda x: x in home_maid, work_maid)))
    # df_work_home=df[df['maid'].isin(list(filter(lambda x: x in home_maid, work_maid)))]

    # for lat, lon in zip(df_work_home[df_work_home['Distance_To_Home (Km)']<=radius_input]['Home_latitude'], df_work_home[df_work_home['Distance_To_Home (Km)']<=radius_input]['Home_longitude']):
    #     color = 'black'
    #     folium.CircleMarker(location=[lat, lon], radius=4, color=color, fill=True, fill_color=color,
    #                         fill_opacity=1).add_to(feature_group_home)

    # for lat, lon in zip(df['latitude'], df['longitude']):
    #     color = 'blue'
    #     folium.CircleMarker(location=[lat, lon], radius=2, color=color, fill=True, fill_color=color,
    #                         fill_opacity=1).add_to(m)
        
    for lat, lon in zip(df[df['Distance_To_Home (Km)']<=radius_input]['Home_latitude'], df[df['Distance_To_Home (Km)']<=radius_input]['Home_longitude']):
        color = 'Orange'
        folium.CircleMarker(location=[lat, lon], radius=2, color=color, fill=True, fill_color=color,
                            fill_opacity=1).add_to(feature_group_home)


    filtered_df_home_geo_unique_count = df[df['Distance_To_Home (Km)']<=radius_input]['homegeohash9'].nunique()
    filtered_df_workgeo_unique_count = df[df['Distance_To_WorkPlace (Km)']<=radius_input]['workgeohash'].nunique()

        
    # for lat, lon in zip(df_work_home[df_work_home['Distance_To_WorkPlace (Km)']<=radius_input]['Work_latitude'], df_work_home[df_work_home['Distance_To_WorkPlace (Km)']<=radius_input]['Work_longitude']):
    #     color = 'black'
    #     folium.CircleMarker(location=[lat, lon], radius=4, color=color, fill=True, fill_color=color,
    #                         fill_opacity=1).add_to(feature_group_work)
    for lat, lon in zip(df[df['Distance_To_WorkPlace (Km)']<=radius_input]['Work_latitude'], df[df['Distance_To_WorkPlace (Km)']<=radius_input]['Work_longitude']):
        color = 'green'
        folium.CircleMarker(location=[lat, lon], radius=2, color=color, fill=True, fill_color=color,
                            fill_opacity=1).add_to(feature_group_work)


    # filtered_df_workgeo_and_home_unique_count = df[(df['Distance_To_WorkPlace (Km)']<=radius_input) & (df['Distance_To_Home (Km)']<=radius_input)]['workgeohash'].nunique()

    # # Add feature groups to the map





    feature_group_home.add_to(m)
    feature_group_work.add_to(m)

    folium.LayerControl().add_to(m)

    col1,col2=st.columns((0.6,0.3))
    with col1:
        folium_static(m)
    with col2:
        with st.expander('About',expanded=True):
                st.write("""
                - :blue[**All Devices**]
                - :orange[**Home Locations - {}**]
                - :green[**Work Locations - {}**]
                - No of devices that both home and work within radius - {}
                - This all points are representing within {}{} 
                - Benifits:- This can help to send Email, SMS selected area
                """.format(filtered_df_home_geo_unique_count,filtered_df_workgeo_unique_count,common_items,radius_input,'km'))

        # st.write(filtered_df_workgeo_unique_count)
    stats=pd.DataFrame({"Mailing":[12],"SMS":[8],"Telemarketing(Mob / Phn)":[11],"Mobile	Emailing":[16]}).T
    # Age_Range=pd.DataFrame({"under 18":[13],"18-24":[4],"24-28":[5],"50-54":[6]}).T
    # Gender=pd.DataFrame({"Male":[8],"Female":[12]}).T

    stats.columns=['Count']
    # Age_Range.columns=['Count']
    # Gender.columns=['Count']
    st.markdown(":orange[Stats Report]")
    col1,col2,col3=st.columns((3))
    # with col1:
    with st.expander("Mobile-Email Stats"):
        st.write(stats)
    # with col2:
    #     with st.expander("Age Range Stats"):
    #         st.write(Age_Range)
    # with col3:
    #     with st.expander("Gender Stats"):
    #         st.write(Gender)
        



if __name__=='__main__':
    main()