import pandas as pd
import streamlit as st
import geohash2
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")
from shapely.geometry import Point, Polygon
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
import folium
from streamlit_folium import folium_static
from collections import Counter
import numpy as np

st.set_page_config(page_title='Lookalike',page_icon=':earth_asia:',layout='wide')
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
st.title(":orange[**Lookalike Model**]")
st.markdown(':green[**This page displays a lookalike model that identifies individuals who visit similar locations, indicating shared interests. It highlights people who frequent comparable places, suggesting common preferences and behaviors.**]')
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

user_input_lat = st.sidebar.text_input("Enter a User latitude:", value="-37.82968153089708")
user_input_lon = st.sidebar.text_input("Enter a User longitude :", value="145.05531534492368")
lookalike_radius = st.slider("Select radius (in kilometers):", min_value=1, max_value=100, value=10)

if user_input_lat and user_input_lon :
    user_lat = float(user_input_lat)
    user_lon = float(user_input_lon)
    df = pd.read_csv('looklike_sample.csv', sep=",").dropna(subset=['latitude', 'longitude'])
    df['datetimestamp'] = pd.to_datetime(df['datetimestamp'])
    df[['Home_latitude', 'Home_longitude']] = df['homegeohash9'].apply(decode_geohash)
    df[['Work_latitude', 'Work_longitude']] = df['workgeohash'].apply(decode_geohash)
    df = df[df.apply(lambda row: haversine(user_lat, user_lon, float(row['Home_latitude']), float(row['Home_longitude'])) <= lookalike_radius, axis=1)]
    home_unique=df.homegeohash9.nunique()
    # df['Home_Distance'] = df.apply(lambda row: haversine(user_lat, user_lon, float(row['Home_latitude']), float(row['Home_longitude'])), axis=1)
    # df['Work_Distance'] = df.apply(lambda row: haversine(user_lat, user_lon, float(row['Work_latitude']), float(row['Work_longitude'])), axis=1)

st.markdown(f"<font color='orange'><b>Total Count inside Population: {home_unique}</b></font>", unsafe_allow_html=True)

# st.write(df[['Unnamed: 0','Home_Distance','Work_Distance','maid']]['maid'].unique().shape)

m = folium.Map(location=[user_lat, user_lon], zoom_start=14)
folium.Marker(location=[user_lat, user_lon], radius=4, color='red', fill=True, fill_color='red',
                    fill_opacity=1).add_to(m)
folium.Circle(
    location=(user_lat, user_lon),
    radius=lookalike_radius*1000,
    color='green',
    fill=True,
    fill_opacity=0.3,
    ).add_to(m)

for _, row in df.iterrows():
    distance = haversine(user_lat, user_lon, row['latitude'], row['longitude'])
    if distance <= lookalike_radius:
        folium.CircleMarker(location=[row['latitude'], row['longitude']], radius=2, color='blue', fill=True, fill_color='blue',
                            fill_opacity=1).add_to(m)



items = {}
no_of_locations = st.text_input("Enter No of Comparible Places : ",value=5)
for i in range(1, int(no_of_locations) + 1):
    location_name = f"loc{i}"
    items[location_name] = []
    items['user_loc']=[]

if  no_of_locations:
    no_of_locations = int(no_of_locations)
    industry_list = ['lon-lat','Location','Distance']
    user_coordinates = []

    default_values = [["""-37.83048693207621, 145.05592287738588\n
-37.83030085905829, 145.05619157963474\n
-37.83026077076795, 145.05613813428943\n
-37.83044526994285, 145.0558759836133""",'Deco Pizza Bar',130.0], 
["""-37.83070476643869, 145.05617241480425\n
-37.83080419639975, 145.0562874136186\n
-37.83066066917268, 145.05650802500193\n
-37.83055951208518, 145.0563812903305""",'East End Wine Bar',170.0], 
["""-37.82671827878225, 145.05708479417163\n
-37.826823149066556, 145.05706266613717\n
-37.82683850865025, 145.05719208280453\n
-37.82673257904397, 145.05720884625617""",'East of Everything',500.0], 
["""-37.82645174444236, 145.057577711162\n
-37.82629586014539, 145.0576019921984\n
-37.8262629038709, 145.0571824766655\n
-37.82641433358738, 145.0571531562729""",'Palace Hotel',500.0], 
["""-37.82437736290543, 145.0493417080464\n
-37.82456910141952, 145.0492759944859\n
-37.82457969449895, 145.0493846240304\n
-37.82438689667038, 145.04941949198823""",'The Tower',900.0]]

    location_data1 = []
    location_data = []

    for idx in range(no_of_locations):
        lon_lat_location_data=[]
        row = st.columns(3)

        # Get the default values for the current location
        default_value = default_values[idx] if idx < len(default_values) else ['0.0,0.0','',0.0]

        for i, industry in enumerate(industry_list):
            if industry == 'Location':
                # Display the location identifier separately
                location_identifier = row[i].text_input(f"{industry} - Location {idx + 1}", value=default_value[i])
                location_data1.append(location_identifier)


            elif industry == 'lon-lat':
                lon_lat_location_identifier = row[i].text_area(f"{industry} - Location {idx + 1}", value=default_value[i])


            else:
                # Display numeric fields for latitude, longitude, and radius
                value = row[i].number_input(f"{industry} - Location {idx + 1}", format="%.0f", value=default_value[i])
                location_data.append(value)

        # Append the location identifier to the location data
        # location_data.append(location_identifier)
        lon_lat_location_data.append(lon_lat_location_identifier)
        user_coordinates.append(lon_lat_location_data)

dfs=[]
for idx, coordinates_list in enumerate(user_coordinates):
    # Parse coordinates from the list
    coordinates = [tuple(map(float, coord.split(','))) for coord in coordinates_list[0].split('\n') if coord.strip()]

    # Add the polygon to the map
    folium.Polygon(locations=coordinates, color='red', fill=True, fill_color='red', fill_opacity=0.4).add_to(m)
    polygon_shapely = Polygon(coordinates)

    maid_counts = Counter()
    matching_records=[]
    for index, row in df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        datetimestamp=row['datetimestamp']
        
        point = Point(lat, lon)
        if point.within(polygon_shapely):
            maid = row['maid']
            maid_counts[maid] += 1
            matching_records.append([maid,lat,lon,datetimestamp])

    filtered_df=pd.DataFrame(matching_records,columns=["maid","latitude","longitude","datetimestamp"])
    count_within_radius_df_maid = filtered_df.groupby('maid').size().reset_index(name=location_data1[idx]).sort_values(by=location_data1[idx], ascending=False)

    # count_within_radius_df_maid = filtered_df.groupby('maid').size().reset_index(name=location_data1[idx]).sort_values(by=location_data1[idx], ascending=False)
    # # df1=count_within_radius_df_maid.join(df,on='maid',how='inner')
    merged_df = count_within_radius_df_maid.merge(df, on='maid', how='left')
    merged_df = merged_df.iloc[:, :2].join(merged_df[['Home_Distance', 'Work_Distance']])
    merged_df = merged_df.drop_duplicates()
    dfs.append(merged_df)
    # st.write(merged_df)

result_df = dfs[0]

for idx, df in enumerate(dfs[1:], start=2):
    result_df = pd.merge(result_df, df, on='maid', how='outer', suffixes=('', f'_{idx}'))

# Fill NaN values with 0
result_df.fillna(0, inplace=True)

# result_df.rename(columns={'Home_Distance_2': 'Home_Distance', 'Work_Distance_2': 'Work_Distance'}, inplace=True)


# Create a DataFrame with distances from loc1 to other locations
data = {location: [value] for location, value in zip(location_data1, location_data)}
df = pd.DataFrame(data)
# Calculate the distances from loc1 to other locations
distances = df.iloc[0].values
# st.write((distances))

reference_distance = distances[0]
normalized_distances = 1.0 / (distances)

# Determine the lowest distance
min_distance = min(distances)

# Calculate weights inversely proportional to distances
weights = normalized_distances / min_distance

# Normalize weights to a 0-1 scale
sum_weights = np.sum(weights)
scaled_weights = weights / sum_weights

# Create a dictionary of location weights
location_weights = {location: weight for location, weight in zip(location_data1, scaled_weights)}

st.write("location weights",location_weights)

result_df['Score'] = 0  # Initialize the Score column

# Iterate over each location and its corresponding weight
for location, weight in location_weights.items():
    # Multiply the values of the location column with its weight and add to the Score column
    result_df['Score'] += result_df[location] * weight

st.markdown(f"<font color='orange'><b>Lookalike Audience Count: {len(result_df)}</b></font>", unsafe_allow_html=True)

with st.expander('View Lookalike Audience'):
    st.write(result_df.sort_values(by=['Score', 'Home_Distance'], ascending=[False, True]))

col1,col2 = st.columns((0.6,0.3))
with col1:
    folium_static(m)
with col2:
    with st.expander("About",expanded=True):
        st.write("""
                :orange[**This lookalike module finds similar people who visits nearby similar places near the specified 
                 location(Dan Murphy). It calculates a score based on how close these places are to the specified location and 
                 how often they are visited and how many multiple locations they visits. It prioritizes places that are closer and frequently visited by people.**]

                - :green[Identify the population within a 10km radius of the specified location (Dan Murphy's) whose residences 
                 fall within this radius.]
                - :green[Determine similar nearby places (such as pubs and bars) for the given location and represent them as polygons.]
                - :green[Assign weights to prioritize each specified location based on its distance from Dan Murphy's. Locations closer to Dan Murphy's receive higher weights.]
                - :green[Calculate the visitation frequency for each specified places.]
                - :green[Combine all location visitations and calculate scores for each location based on the assigned weights.]
                - :green[Sort the maids by their scores, with the highest score indicating the most probable lookalike audience.]
                 """)