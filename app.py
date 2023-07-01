import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor


pipe = pickle.load(open('pipe1.pkl','rb'))

teams = ['Australia',
 'India',
 'Bangladesh',
 'New Zealand',
 'South Africa',
 'England',
 'West Indies',
 'Afghanistan',
 'Pakistan',
 'Zimbabwe',
 'Sri Lanka']

cities = ['Melbourne', 'Adelaide', 'Harare', 'Napier', 'Mount Maunganui',
       'Auckland', 'Southampton', 'Cardiff', 'Chester-le-Street',
       'Nagpur', 'Bangalore', 'Lauderhill', 'Dubai', 'Abu Dhabi',
       'Sydney', 'Hobart', 'Wellington', 'Hamilton', 'Bloemfontein',
       'Potchefstroom', 'Barbados', 'Trinidad', 'Colombo', 'St Kitts',
       'Nelson', 'Ranchi', 'Birmingham', 'Manchester', 'Bristol', 'Delhi',
       'Rajkot', 'Lahore', 'Johannesburg', 'Centurion', 'Cape Town',
       'Indore', 'Mumbai', 'Dhaka', 'Sylhet', 'Sharjah', 'Karachi',
       'East London', 'Brisbane', 'Dehradun', 'Kolkata', 'Lucknow',
       'Chennai', 'Bengaluru', 'Canberra', 'Perth', 'Durban',
       'Port Elizabeth', 'Chandigarh', 'Christchurch', 'Kandy',
       'Chattogram', 'Pune', 'Rawalpindi', 'London', 'Nottingham',
       'King City', 'Guyana', 'St Lucia', 'Antigua', 'Pallekele',
       'Mirpur', 'Hambantota', 'Bulawayo', 'St Vincent', 'Chittagong',
       'Dominica', 'Khulna']

st.title('Cricket Score Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

city = st.selectbox('Select city',sorted(cities))

col3,col4,col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score')
with col4:

        overs = st.number_input('Overs done(works for over>5)')
with col5:
    wickets = st.number_input('Wickets out')

last_five = st.number_input('Runs scored in last 5 overs')

if st.button('Predict Score'):
    balls_left = 120 - (overs*6)
    wickets_left = 10 -wickets
    crr = current_score/overs

    input_df = pd.DataFrame(
     {'batting_team': [batting_team], 'bowling_team': [bowling_team],'city':city, 'current_score': [current_score],'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr], 'last_five': [last_five]})
    result = pipe.predict(input_df)
    st.header("Predicted Score - " + str(int(result[0])))



