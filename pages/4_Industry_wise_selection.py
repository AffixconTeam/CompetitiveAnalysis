import pandas as pd
import streamlit as st
import altair as alt
import re

# Page configuration
st.set_page_config(
    page_title="Email Marketing",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")
custom_css = """
<style>
body {
    background-color: #22222E; 
    secondary-background {
    background-color: #FA55AD; 
    padding: 10px; 
}
</style>
"""
st.write(custom_css, unsafe_allow_html=True)
st.markdown(custom_css, unsafe_allow_html=True)
st.title("Data Selection")
st.markdown(":green[**This page filters specific records based on user-selected industries and demographics. It allows users to refine their data analysis by narrowing down their focus to specific industry sectors and demographic characteristics.**]")
df_seg=pd.read_csv('affixcon_segments.csv',encoding='latin-1').dropna(subset=['segment_name'])
df_seg['code'] = df_seg['code'].astype(str)
df_seg.category = df_seg.category.str.upper()
industry_list = df_seg['industry'].dropna().unique().tolist()
selected_industry = st.selectbox(' :bookmark_tabs: Enter Industry:', industry_list)
selected_code= df_seg.loc[df_seg['industry'] == str(selected_industry), 'code'].values[0]

df=pd.read_csv('10000_Movements.csv',usecols=['interests','brands_visited','place_categories','geobehaviour',"Gender","Income","Age_Range"])
def filter_values(df, input_char):
    
    numeric_part = re.search(r'\d+', input_char)
    # Extract unique values from column 'b' that start with the given input_char
    filtered_values = [value for value in df['code'].unique() if value.startswith(input_char)]

    # # If input_char has a dot ('.'), filter values at any level with one more digit after the dot
    if '.' in input_char:
        filtered_values = [value for value in filtered_values if re.match(f"{input_char}\.\d+", value)]
    else:
        if numeric_part: 
            filtered_values = [item for item in filtered_values if str(item).count('.') == 1]
            filtered_values = [value for value in filtered_values if value.split('.')[0] == input_char]
    #     # If input_char is only alphabet, filter values without a dot ('.')
        else:
            filtered_values = [value for value in filtered_values if not re.search(r'\.', value)]
            filtered_values = [item for item in filtered_values if str(item) != input_char]
    return filtered_values


item_list = []
segment_industry_dict = df_seg.groupby('code')['segment_name'].apply(list).to_dict()
def find_similar_codes(input_code, df):
    similar_codes = []
    for index, row in df.iterrows():
        code = row['code']
        if isinstance(code, str) and code.startswith(input_code):
            similar_codes.append(code)
    return similar_codes


user_contain_list = list(set(find_similar_codes(selected_code, df_seg)))

if selected_code in user_contain_list:
    for code in user_contain_list:
        item_list_code = segment_industry_dict[code]
        for item in item_list_code:
            item_list.append(item)
else:
    item_list = []

selected_segments=item_list


industry_list=[]
filtered_codes = filter_values(df_seg, selected_code)
# st.write(filtered_codes)
code_industry_dict = df_seg.groupby('code')['industry'].apply(list).to_dict()

# if selected_code in filtered_codes:
for code in filtered_codes:
    item_list_code = code_industry_dict[code]
    for item in item_list_code:
        industry_list.append(item)

industry_list=list(set(industry_list))
# Determine the number of columns based on the length of industry_list
niche_list=[]

#---------------------------------------------------
if len(industry_list)>0:
    selected_industries = []

    # Calculate the number of rows needed
    num_rows1 = (len(industry_list) + 2) // 3

    # Iterate over each row
    for i in range(num_rows1):
        # Calculate the start and end index for the current row
        start_index = i * 3
        end_index = min(start_index + 3, len(industry_list))
        
        # Create a row for checkboxes
        row = st.columns(3)
        
        # Iterate over each item in the current row and create checkboxes
        for j, industry in enumerate(industry_list[start_index:end_index]):
            # Add checkbox for the current industry to the row
            checkbox_industry_list = row[j].checkbox(industry)
            if checkbox_industry_list:
                selected_code= df_seg.loc[df_seg['industry'] == str(industry), 'code'].values[0]
                selected_industries.append(selected_code)
 
        for code in selected_industries:
            item_list_code = segment_industry_dict[code]
            for item in item_list_code:
                niche_list.append(item)


#--------------------------------------------------
# if len(industry_list)>0:
#     num_columns = len(industry_list)
#     st.write("Select Niche Market Industry")
#     # Create the columns dynamically
#     columns = st.columns(num_columns)
#     selected_industries = []
#     for i, industry in enumerate(industry_list):
#         checkbox_industry_list = columns[i].checkbox(industry)
#         if checkbox_industry_list:
#             selected_code= df_seg.loc[df_seg['industry'] == str(industry), 'code'].values[0]
#             selected_industries.append(selected_code)
#             # st.write(selected_code)  
#     for code in selected_industries:
#         item_list_code = segment_industry_dict[code]
#         for item in item_list_code:
#             niche_list.append(item)

else:
    pass

if len(niche_list)>0:
    selected_segments=niche_list

select_all_segments=st.toggle("Select All Segments", value=True)

# If the "Select All Segments" checkbox is checked, select all segments
if select_all_segments:
    selected_segments = selected_segments
else:
    # Create a multiselect widget
    selected_segments = st.multiselect("Select one or more segments:", selected_segments)

segment_category_dict = df_seg.set_index('segment_name')['category'].to_dict()
result_dict = {}
filtered_dict = {key: value for key, value in segment_category_dict.items() if key in selected_segments}

for key, value in filtered_dict.items():

    if value not in result_dict:
        result_dict[value] = []

    result_dict[value].append(key)
    result_dict = {key: values for key, values in result_dict.items()}
# st.write(result_dict)

if 'BRAND VISITED' in result_dict and 'BRANDS VISITED' in result_dict:
    # Extend the 'a' values with 'a1' values
    result_dict['BRAND VISITED'].extend(result_dict['BRANDS VISITED'])
    # Delete the 'a1' key
    del result_dict['BRANDS VISITED']

selected_category = st.sidebar.radio("Select one option:", list(result_dict.keys()))
if selected_segments:
    if selected_category == 'INTERESTS':
        segment_list=result_dict['INTERESTS']
    elif selected_category == 'BRAND VISITED':
        segment_list=result_dict['BRAND VISITED']
    elif selected_category == 'PLACE CATEGORIES':
        segment_list=result_dict['PLACE CATEGORIES']
    elif selected_category == 'GEO BEHAVIOUR':
        segment_list=result_dict['GEO BEHAVIOUR']
else:
    segment_list=[]

for j in segment_list:
    st.sidebar.write(j)

def filter_condition(df,lst):
    filter_conditions = [df[col_name].apply(lambda x: any(item in str(x).split('|') for item in lst))
        for col_name in ['interests', 'brands_visited', 'place_categories', 'geobehaviour']]
    final_condition = filter_conditions[0]
    for condition in filter_conditions[1:]:
        final_condition = final_condition | condition
    df_new = df[final_condition]
    return df_new
df=filter_condition(df,selected_segments)
def filter_items(column):
    return [item for item in column.split('|') if item in selected_segments]
columns_to_filter = ['interests', 'brands_visited', 'place_categories', 'geobehaviour']
for column in columns_to_filter:
    df[column] = df[column].apply(filter_items)
df[columns_to_filter] = df[columns_to_filter].applymap(lambda x: '|'.join(x))

all_gender=[str(gender_range) for gender_range in df.Gender.dropna().unique()]
all_Income_values = df.Income.dropna().unique()
unique_income_values = sorted(set(value for value in all_Income_values if value != 'unknown_income'))
ordered_categories = [
    "Under $20,799",
    "$20,800 - $41,599",
    "$41,600 - $64,999",
    "$65,000 - $77,999",
    "$78,000 - $103,999",
    "$104,000 - $155,999",
    "$156,000+"
]
sorted_income_values = sorted(unique_income_values, key=lambda x: ordered_categories.index(x))
ordered_income_series = pd.Categorical(sorted_income_values, categories=ordered_categories, ordered=True)


age_range_filter = st.multiselect(f"Select Age Range", ["All"] + sorted([str(age_range) for age_range in df.Age_Range.dropna().unique()]),default=['All'])
Gender_filter = st.multiselect(f"Select Gender Range", ["All"] + sorted(value for value in all_gender if value != '_'),default=['All'])
income_range_filter = st.multiselect(f"Select Income Range", ["All"] + list(ordered_income_series),default=['All'])

all_age_range_values=[str(age_range) for age_range in df.Age_Range.unique()]
all_gender_values=[str(gender_range) for gender_range in df.Gender.unique()]
Income_filter=list(df.Income.unique())

if "All" in age_range_filter:
    # age_range_filter = all_age_range_values
    age_range_filter=all_age_range_values
if "All" in Gender_filter:
    Gender_filter = all_gender_values
if "All" in Income_filter:
    Income_filter = all_Income_values

filtered_df=df.query('Gender ==@Gender_filter & Age_Range==@age_range_filter & Income==@Income_filter')
count=len(filtered_df)
st.text(f"No of records in selected filters: {count}")

with st.expander("View filtered Sample Data"):
    st.write(filtered_df.sample(10))

selections = [
    {"selected_industry": selected_industry},
    {"niche_list": selected_segments},
    {"age_range_filter": age_range_filter} if  'All' not in age_range_filter else {},
    {"Gender_filter": Gender_filter} if not 'All' in Gender_filter else {},
    {"Income_filter": Income_filter} if not 'All' in Income_filter else {},

]
flat_dict = {}
for item in selections:
    flat_dict.update({key: str(value) for key, value in item.items()})
selection_df = pd.DataFrame([flat_dict]).T.rename(columns={0: 'Selections'})
csv=selection_df.to_csv().encode('utf-8')
st.write(selection_df)

st.download_button("Download Selection table",data=csv, file_name="Selection.csv")