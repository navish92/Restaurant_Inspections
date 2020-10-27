"""
Python Script to render 'Restaurant Inspection Outcome Prediction' on a webpage using Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
# from sklearn


st.write('''
# Welcome!

### This webpage predicts the outcome of a restaurant's health inspection.  

Just select the correct values below to **predict the outcome**! 
  
  
Scroll to the bottom of the page to view the Boston restaurants on a map,  
that were used to make this model.  
You can also search for a specific restaurant & look at its historical inspection information.  


''')


historic_ins_count = st.number_input("# of Routine Inspections Till Now", min_value = 2, value = 5, step = 1)
passed_ins_count = st.number_input("# of Routine Inspections Passed", min_value = 1, value = 3, step = 1)
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

st.write(f'Expected Inspection Outcome: {int(prediction[0])}')