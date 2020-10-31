"""
Python Script to render 'Restaurant Inspection Outcomes' on a webpage using Streamlit & a bit of HTML.

The app has two distinct portions that are individually accessible through a sidebar:
1. Predict - For generating restuarant predictions using the model developed, based on the custom criteria that can be entered by the user.
2. Explore Data - Explore the restaurants whose data was used to create the model itself. The information can be filtered using their most        recent inspection result & zipcode. All relevant historical inspection information about the restaurant can be viewed on a map.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression

# Loading Data for the "Explore Data" section
df = pd.read_csv("rest_post_eda.csv",index_col=0)
df.rename(columns={},inplace=True)

# Making the column names more display friendly
df.rename(columns={'businessname':'Restaurant Name', 'review_count':'# Yelp Reviews', 'rating':'Yelp Rating', 'price':'Price Category',
                   'categories_clean':'Food Category', 'address':'Address', 'zip':'Zipcode', 'historic_routine_ins_count':\
                   '# Inspections', 'failed_ins_count':'# Failed', 'most_recent_previous_ins':\
                   'Most Recent Prior Inspecton Result', '2nd_most_recent_previous_ins':'2nd Most Recent Inspection Result', \
                   'target':'Inspection Result','Latitude':'lat','Longitude':'lon'},inplace=True)


#Zipcode is currently encoded in 4 digits.
df['Zipcode'] = df['Zipcode'].map(lambda x: "0"+str(x))


# df_preds = pd.read_csv("test_data_with_predictions.csv")


# Sidebar Condiguration
pages = ["Predict", "Explore Data"]

st.sidebar.markdown("**Boston Restaurant Inspections Prediction**")
page = st.sidebar.radio("Please select a section", options=pages)
st.sidebar.markdown('---')
st.sidebar.write('Created by Navish Agarwal')

sidebar_gradient = """
<style> .sidebar .sidebar-content {
    background-image: linear-gradient(#ff944d,#ff471a);
    color: black; }
</style>
"""
st.sidebar.markdown(sidebar_gradient,  unsafe_allow_html=True)


# Prediction Text
if page == "Predict":
    st.title("Restaurant Inspection Prediction")
    st.write('''
## Welcome!
### This page predicts the outcome of a restaurant's health inspection.  
   
Just select the correct values below to **predict the outcome**!  

    ''')
    
    # Information input by the user & necessary transformation
    historic_ins_count = st.number_input("Number of Past Inspections", min_value = 2, value = 5, step = 1)
    passed_ins_count = st.number_input("Number of Succesful Past Inspections", min_value = 1, value = 3, step = 1)
    most_recent_ins = st.radio("Result of Most Recent Inspection",['Pass','Fail'], index = 0)
    second_most_recent_ins = st.radio("Result of Inspection Prior to last",['Pass','Fail'], index = 0)
    rating = st.slider("Yelp Rating",min_value = 0.0, max_value = 5.0, value = 2.5, step = 0.5)
    yelp_reviews = st.number_input("# of Reviews on Yelp",min_value = 0, value = 57, step = 1)

    review_count_log = 0 if yelp_reviews == 0 else np.log(yelp_reviews)
    most_recent_previous_ins = 1 if most_recent_ins == 'Pass' else 0
    second_most_recent_ins_number = 1 if second_most_recent_ins == 'Pass' else 0

    failed_ins_count = historic_ins_count - passed_ins_count
    passed_ins_ratio = passed_ins_count/historic_ins_count
    failed_ins_ratio = failed_ins_count/historic_ins_count

    #Pre-trained Logistic Regression model to be used for predictions
    with open("logreg_model.pickle", "rb") as model_pickle:
        random_model = pickle.load(model_pickle)
    
# features = ['review_count_log', 'rating', 'historic_routine_ins_count', 'most_recent_previous_ins', \
#             'failed_ins_count', 'passed_ins_count', '2nd_most_recent_previous_ins', 'passed_ins_ratio','failed_ins_ratio',]
            
    variables = np.array([review_count_log, rating, historic_ins_count, most_recent_previous_ins, \
                          failed_ins_count, failed_ins_count, second_most_recent_ins_number, \
                          passed_ins_ratio,failed_ins_ratio]).reshape(1,-1)

    prediction = random_model.predict(variables)

    predict_button = st.button("Predict")
    
    if predict_button:
        try:
            st.write(f'Expected Inspection Outcome:')
            if int(prediction):
                st.markdown('## **PASS**')
#                 st.balloons()
                del predict_button
            else:

                st.markdown('## **FAIL**')
                del predict_button


            st.write("\n\nFeel Free to keep changing the values")
        except:
            st.error("There's some error. Sorry!")
    
#Data Exploration Visual
if page == "Explore Data":
    
    st.title("Historical Inspections Data for Boston Restaurants")
    st.write("")
    st.markdown('''
 **View the inspections data for each restaurant that was used to develop the model.  
 Feel free to use the filter options below to look at narrower set of restaurants**
 ''')
    
    features = ['Restaurant Name', 'Inspection Result', '# Yelp Reviews', 'Yelp Rating', 'Price Category', 'Food Category', \
                'Address', 'Zipcode', '# Inspections', '# Failed', 'Most Recent Prior Inspecton Result', \
                '2nd Most Recent Inspection Result']
    
    filter_status = st.radio("Filter Data?",["Yes","No"],index=1)
    
    st.write("")
    if filter_status == "Yes":
#         ins_num = st.multiselect("Inspection Result",['Pass','Fail'],default=[])
        ins_num = st.multiselect("Inspection Result",[1, 0],default=[])
        zipco = st.multiselect('Select Zip Code', np.sort(df.Zipcode.unique()), default = [])
        
#         ins_results_dict = {'Pass':1,'Fail': 0}                                              
        if not ins_num:
            ins_num = [1,0]
            
# #         else:
#             print(ins_num)
#             print("in else stateemnt")
#             ins_num = [ins_results_dict.get(x,0) for x in ins_num]
            
                                                      
        zipco = np.sort(df.Zipcode.unique()) if len(zipco) == 0 else zipco
        mask = (df['Zipcode'].isin(zipco)) & (df['Inspection Result'].isin(ins_num))
#         print(mask)
        print(ins_num)
                                                      
        rows = len(df.loc[mask,:])
        st.write("")
        st.write(f"**Displaying {rows} restaurants**")

        st.write("")
        st.subheader("Map of Restaurants")
        st.map(df.dropna(subset=['location']).loc[mask,['lat','lon']])
        st.write("")
        st.dataframe(df.loc[mask,features])
      
        
    else:
        st.subheader("Map of Restaurants")
        st.map(df.dropna(subset=['location']).loc[:,['lat','lon']])
        st.write("")
        st.dataframe(df.loc[:,features])
#         
        




