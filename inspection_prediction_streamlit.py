"""
Python Script to render 'Restaurant Inspection Outcome Prediction' on a webpage using Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression

# Loading Dataframes
df = pd.read_csv("rest_post_eda.csv",index_col=0)
df.rename(columns={'Latitude':'lat','Longitude':'lon'},inplace=True)

df_preds = pd.read_csv("test_data_with_predictions.csv")

df.rename(columns={'businessname':'Restaurant Name','review_count':'Yelp Reviews','rating':'Yelp Rating','price':'Price Category',
                      'categories_clean':'Category','address':'Address','zip':'Zipcode','historic_routine_ins_count':'Historic Inspections Count',
                      'failed_ins_count':'Failed Inspection Count','most_recent_previous_ins':'Most Recent Prior Inspecton Result',
                      '2nd_most_recent_previous_ins':'Second Most Recent Inspection Result','target':'Inspection Result'},inplace=True)
df['Zipcode'] = df['Zipcode'].map(lambda x: "0"+str(x))
# df[]
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
# ff3333 #ff5c33
st.sidebar.markdown(sidebar_gradient,  unsafe_allow_html=True)


# "Historical Restaurant Inspection Data","Restaurant Inspection Prediction"

# Background Image for body
Body_html = """
  <style>
   body{
    background-image: url(https://uca1e5566a673f42cc568195e555.previews.dropboxusercontent.com/p/thumb/AA9aE9wrHEeFLmCsFdopoTTB4xfz4Qmz28z6sDCi5PsiOFKknblSdsSoqrJ-qpi7y0mUOe_cWthHDi6nIfQVFFGISFPozme4QEvB6uu-8EqU5ypJvlnchbj4ZMOjm66Pz8PQtrlCgWjLAzO4fqrSAwXle8xBqRoKL72d_mcc5d4sOk0wfqv5EGBMbQVHGoymfQT2I8gDWa9HqImRPxSm6mmq4JImBC4Lc_8nldKbXiqFQVZvCUJB4WfQkryUuowGJOo63OVPWdq7ObFSGyWSb_T1zsvo3jHL5l5SqR0I6Mt6r2BOPohMf176iBdZRPAalv4qjdumZCrKlsLdq-xevwX5RYOd4tKn_sH16iTqo2iSkNLdsbs7keeyBsKGubSHHDEjvK1n3jXvtELMm9RaaTP5/p.jpeg?size=1600x1200&size_mode=3);
    
        background-size: cover;

  
  }
</style>
"""
st.markdown(Body_html, unsafe_allow_html=True) 

# Prediction Text
if page == "Predict":
    st.title("Restaurant Inspection Prediction")
    st.write('''
## Welcome!
### This webpage predicts the outcome of a restaurant's health inspection.  
   
Just select the correct values below to **predict the outcome**!  

    ''')

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

    with open("logreg_model.pickle", "rb") as model_pickle:
        random_model = pickle.load(model_pickle)
    
# features = ['review_count_log', 'rating', 'historic_routine_ins_count', 'most_recent_previous_ins', \
#             'failed_ins_count', 'passed_ins_count', '2nd_most_recent_previous_ins', 'passed_ins_ratio','failed_ins_ratio',]
            
    variables = np.array([review_count_log, rating, historic_ins_count, most_recent_previous_ins, \
                          failed_ins_count, failed_ins_count, second_most_recent_ins_number, passed_ins_ratio, failed_ins_ratio]).reshape(1,-1)

    prediction = random_model.predict(variables)
    print(int(prediction))

    predict_button = st.button("Predict")
    
    if predict_button:
        try:
            st.write(f'Expected Inspection Outcome:')
            if int(prediction):
                st.markdown('## **PASS**')
                st.balloons()
            else:

                st.markdown('## **FAIL**')


            st.write("\n\nFeel Free to keep changing the values")
        except:
            st.error("There's some error. Sorry!")
    
       
if page == "Explore Data":
    st.title("Historical Inspections Data for Boston Restaurants")
    st.write("")
    st.markdown('''
 **View the inspections data for each restaurant that was used to develop the model.  
 Feel free to use thes filter options to look at narrower set of restaurants**
 ''')
    
    features = ['Restaurant Name', 'Inspection Result', 'Yelp Reviews', 'Yelp Rating', 'Price Category', 'Category', 'Address', 'Zipcode', 
                'Historic Inspections Count', 'Failed Inspection Count','Most Recent Prior Inspecton Result','Second Most Recent Inspection Result']
    
    filter_status = st.radio("Filter Data?",["Yes","No"],index=1)
    
    st.write("")
    if filter_status == "Yes":
        ins_num = st.multiselect("Inspection Result",[1,0],default=[])
        zipco = st.multiselect('Select Zip Code', np.sort(df.Zipcode.unique()), default = [])
#         ins_num = [1 if x == 'P' else 0 for x in ins_result]
        
        if not ins_num:
            ins_num = [1,0]
            
        zipco = np.sort(df.Zipcode.unique()) if len(zipco) == 0 else zipco
        mask = (df['Zipcode'].isin(zipco)) & (df['Inspection Result'].isin(ins_num))
        print(mask)
        st.write(ins_num)
        rows = len(df.loc[mask,:])
        st.write(f"Displaying {rows} restaurants")

        st.write("")
        st.subheader("Map of Restaurants")
        st.map(df.dropna(subset=['location']).loc[mask,['lat','lon']])
        st.write("")
        st.dataframe(df.loc[mask,features])
      
        
    else:
        st.dataframe(df.loc[:,features])
        st.write("")
        st.subheader("Map of Restaurants")
        st.map(df.dropna(subset=['location']).loc[:,['lat','lon']])





