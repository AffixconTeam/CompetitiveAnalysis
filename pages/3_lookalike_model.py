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
st.markdown("""
    <style>
    .stRadio [role=radiogroup]{
        align-items: center;
        justify-content: center;
    }
    </style>
""",unsafe_allow_html=True)
st.title("Lookalike Model")

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
    

def PCA_plot(visitors_data_all,non_visitors_data):
    combined_data = pd.concat([visitors_data_all, non_visitors_data])
    combined_data=combined_data.drop('maid',axis=1)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_data)
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2']).set_index(combined_data.index)
    labels = ['Visited selected location - Dan Murphy'] * len(visitors_data_all) + ['Population 3km'] * len(non_visitors_data)
    pca_df['Label'] = labels

    # Plot the 2D scatter plot with specified colors for visitors and non-visitors
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Label', title='2D Scatter Plot with PCA', 
                    color_discrete_map={'Visited selected location - Dan Murphy': 'blue', 'Population 3km': 'red'}, opacity=1,labels={"Visitors": "Visited selected location - Dan Murphy", "Non-Visitors": "Population 3km"})

    col1,col2=st.columns((0.6,0.3))
    with col1:
        st.plotly_chart(fig)
    with col2:
        with st.expander('About', expanded=True):
            st.write("""
                :orange[**The lookalike model uses behavioral data from home and work locations to identify similar customers. This data is visualized in a 2D scatter plot, with customers represented as points. 
                     By analyzing proximity in the plot, 
                     the model predicts potential customers with similar behaviors using nearest neighbor search.**]""")
    

def find_lookalike_audience(visitors_data, non_visitors_data):
    visitors_data=visitors_data.drop('maid',axis=1)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(visitors_data)
    distances, indices = nbrs.kneighbors(non_visitors_data)
    nearest_neighbors_indices = indices.flatten()
    lookalike_audience = visitors_data.iloc[nearest_neighbors_indices]
    lookalike_distances = distances.flatten()
    lookalike_audience['Distance'] = lookalike_distances
    return lookalike_audience,nearest_neighbors_indices

start_date = pd.Timestamp('2023-12-01')
end_date = pd.Timestamp('2023-12-31')

st.sidebar.markdown('<p style="color: red;">Select Date Range</p>', unsafe_allow_html=True)

# # Allow user to pick start and end dates
selected_start_date = st.sidebar.date_input("Select Start Date", start_date)
selected_end_date = st.sidebar.date_input("Select End Date", end_date)
selected_start_date = pd.to_datetime(selected_start_date)
selected_end_date = pd.to_datetime(selected_end_date)

df = pd.read_csv('10000_Movements.csv', sep=",").dropna(subset=['latitude', 'longitude'])
df['datetimestamp'] = pd.to_datetime(df['datetimestamp'])
df = df[(df['datetimestamp'] >= selected_start_date) & (df['datetimestamp'] <= selected_end_date)]
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

st.markdown("<p style='color: red;'>Select Location:</p>", unsafe_allow_html=True)
options = ["Search by Radius", "Search by Polygon"]
options=st.radio("Select Search Option", options=options, horizontal=True)

if options == 'Search by Radius':
    dist = st.radio("Select Distance Unit", ["Meters","Kilometers"],horizontal=True)
    if dist == 'Kilometers':
        radius_input = st.slider("Select radius (in kilometers):", min_value=1, max_value=100, value=10)
        unit = 'km'
    elif dist == 'Meters':
        radius_input = st.slider("Select radius (in Meters):", min_value=1, max_value=1000, value=15)
        radius_m=radius_input
        radius_input=radius_input/1000
        unit = 'm'

    if user_input_lat and user_input_lon :
        user_lat = float(user_input_lat)
        user_lon = float(user_input_lon)
        visitors_data_inplace = df[df.apply(lambda row: haversine(user_lat, user_lon, row['latitude'], row['longitude']) <= radius_input, axis=1)]
        visitors_data_all= df[df['maid'].isin(visitors_data_inplace['maid'])]
        visitors_data_all[['Home_latitude', 'Home_longitude']] = visitors_data_all['homegeohash9'].apply(decode_geohash)
        visitors_data_all[['Work_latitude', 'Work_longitude']] = visitors_data_all['workgeohash'].apply(decode_geohash)
        visitors_data_all['Home_Distance'] = visitors_data_all.apply(lambda row: haversine(user_lat, user_lon, float(row['Home_latitude']), float(row['Home_longitude'])), axis=1)
        visitors_data_all['Work_Distance'] = visitors_data_all.apply(lambda row: haversine(user_lat, user_lon, float(row['Work_latitude']), float(row['Work_longitude'])), axis=1)
        visitors_data_all=visitors_data_all[["maid","latitude","longitude","Home_latitude","Home_longitude","Work_latitude","Work_longitude","Home_Distance","Work_Distance"]]
        # visitors_data_all = visitors_data_all.astype({col: float for col in visitors_data_all.columns})

    st.markdown(f"<font color='orange'><b>Total Count inside Location: {len(visitors_data_all)}</b></font>", unsafe_allow_html=True)

    st.markdown('<p style="color: red;">Select Area for Lookalike</p>', unsafe_allow_html=True)
    lookalike_radius = st.slider("Select radius (in kilometers):", min_value=1, max_value=100, value=2)
    non_visitors_data = df[df.apply(lambda row: haversine(user_lat, user_lon, row['latitude'], row['longitude']) <= lookalike_radius, axis=1)]
    st.markdown(f"<font color='orange'><b>Total Count inside Population: {len(non_visitors_data)}</b></font>", unsafe_allow_html=True)

    non_visitors_data= non_visitors_data[~non_visitors_data['maid'].isin(visitors_data_all['maid'])]
    non_visitors_data[['Home_latitude', 'Home_longitude']] = non_visitors_data['homegeohash9'].apply(decode_geohash)
    non_visitors_data[['Work_latitude', 'Work_longitude']] = non_visitors_data['workgeohash'].apply(decode_geohash)
    non_visitors_data['Home_Distance'] = non_visitors_data.apply(lambda row: haversine(user_lat, user_lon, float(row['Home_latitude']), float(row['Home_longitude'])), axis=1)
    non_visitors_data['Work_Distance'] = non_visitors_data.apply(lambda row: haversine(user_lat, user_lon, float(row['Work_latitude']), float(row['Work_longitude'])), axis=1)
    non_visitors_data=non_visitors_data[["latitude","longitude","Home_latitude","Home_longitude","Work_latitude","Work_longitude","Home_Distance","Work_Distance"]]
    non_visitors_data = non_visitors_data.astype({col: float for col in non_visitors_data.columns})
    PCA_plot(visitors_data_all,non_visitors_data)
    lookalike_audience,nearest_neighbors_indices = find_lookalike_audience(visitors_data_all, non_visitors_data)
    lookalike_audience=lookalike_audience.sort_values('Distance')
    lookalike_audience=lookalike_audience[lookalike_audience['Distance']<int(len(lookalike_audience)*0.1)]

    combined_data1 = pd.concat([visitors_data_all, non_visitors_data])

    # st.write(combined_data1.iloc[nearest_neighbors_indices])


else:
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
        polygon_shapely = Polygon(polygon_coordinates)

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
                matching_records.append([maid,lat,lon])

        def distance_to_polygon(lat, lon):
            point = Point(lat, lon)  # Shapely Point requires (lon, lat) format
            if point.within(polygon_shapely):
                return 0
            else:
                return 1

        visitors_data_inplace=pd.DataFrame(matching_records,columns=['maid','latitude','longitude'])
        visitors_data_all= df[df['maid'].isin(visitors_data_inplace['maid'])]
        visitors_data_all[['Home_latitude', 'Home_longitude']] = visitors_data_all['homegeohash9'].apply(decode_geohash)
        visitors_data_all[['Work_latitude', 'Work_longitude']] = visitors_data_all['workgeohash'].apply(decode_geohash)
        visitors_data_all['Home_Distance'] = visitors_data_all.apply(lambda row: haversine(center_lat, center_lon, float(row['Home_latitude']), float(row['Home_longitude'])), axis=1)
        visitors_data_all['Work_Distance'] = visitors_data_all.apply(lambda row: haversine(center_lat, center_lon, float(row['Work_latitude']), float(row['Work_longitude'])), axis=1)
        visitors_data_all=visitors_data_all[["maid","latitude","longitude","Home_latitude","Home_longitude","Work_latitude","Work_longitude","Home_Distance","Work_Distance"]]
        # visitors_data_all = visitors_data_all.astype({col: float for col in visitors_data_all.columns})

        st.markdown(f"<font color='orange'><b>Total Count inside Location: {len(visitors_data_all)}</b></font>", unsafe_allow_html=True)

    # st.write(visitors_data_all)
    # st.write(len(visitors_data_inplace))
    # st.write(len(visitors_data_all))
    st.markdown('<p style="color: red;">Select Area for Lookalike</p>', unsafe_allow_html=True)
    lookalike_radius = st.slider("Select radius (in kilometers):", min_value=1, max_value=100, value=2)
    non_visitors_data = df[df.apply(lambda row: haversine(center_lat, center_lon, row['latitude'], row['longitude']) <= lookalike_radius, axis=1)]
    st.markdown(f"<font color='orange'><b>Total Count inside Population: {len(non_visitors_data)}</b></font>", unsafe_allow_html=True)

    non_visitors_data= non_visitors_data[~non_visitors_data['maid'].isin(visitors_data_all['maid'])]
    non_visitors_data[['Home_latitude', 'Home_longitude']] = non_visitors_data['homegeohash9'].apply(decode_geohash)
    non_visitors_data[['Work_latitude', 'Work_longitude']] = non_visitors_data['workgeohash'].apply(decode_geohash)
    non_visitors_data['Home_Distance'] = non_visitors_data.apply(lambda row: haversine(center_lat, center_lon, float(row['Home_latitude']), float(row['Home_longitude'])), axis=1)
    non_visitors_data['Work_Distance'] = non_visitors_data.apply(lambda row: haversine(center_lat, center_lon, float(row['Work_latitude']), float(row['Work_longitude'])), axis=1)
    non_visitors_data=non_visitors_data[["latitude","longitude","Home_latitude","Home_longitude","Work_latitude","Work_longitude","Home_Distance","Work_Distance"]]
    non_visitors_data = non_visitors_data.astype({col: float for col in non_visitors_data.columns})

    PCA_plot(visitors_data_all,non_visitors_data)

    lookalike_audience = find_lookalike_audience(visitors_data_all, non_visitors_data)
    
    lookalike_audience=lookalike_audience.sort_values('Distance')
    lookalike_audience=lookalike_audience[lookalike_audience['Distance']<int(len(lookalike_audience)*0.1)]

    # prediction_lookalike=df[~df['maid'].isin(visitors_data_inplace['maid'])]['maid']
    # non_visitors_data.iloc[lookalike_audience.index]


st.write('Lookalike Audience',df.iloc[lookalike_audience.index]['maid'].unique()[:5])
st.markdown(f"<font color='orange'><b>Lookalike Audience Unique Count: {df.iloc[lookalike_audience.index]['maid'].nunique()}</b></font>", unsafe_allow_html=True)








