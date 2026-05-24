import streamlit as st
import pickle
import pandas as pd

# Teams
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

# Cities
cities = [
    'Hyderabad','Bangalore','Mumbai','Indore','Kolkata','Delhi','Chandigarh',
    'Jaipur','Chennai','Cape Town','Port Elizabeth','Durban','Centurion',
    'East London','Johannesburg','Kimberley','Bloemfontein','Ahmedabad',
    'Cuttack','Nagpur','Dharamsala','Visakhapatnam','Pune','Raipur',
    'Ranchi','Abu Dhabi','Sharjah','Mohali','Bengaluru'
]

# Load model
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title('IPL Win Predictor')

# Team selection
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

# City
selected_city = st.selectbox('Select host city', sorted(cities))

# Match info
target = st.number_input('Target Score')

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Current Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets fallen')

# Prediction
if st.button('Predict Probability'):

    # Runs left
    runs_left = target - score

    # Convert overs → balls correctly
    overs_int = int(overs)
    balls_part = overs - overs_int
    balls = overs_int * 6 + int(round(balls_part * 10))

    balls_left = 120 - balls
    wickets_left = 10 - wickets

    # Current Run Rate (CRR)
    if balls == 0:
        crr = 0
    else:
        crr = (score / balls) * 6

    # Required Run Rate (RRR)
    if balls_left == 0:
        rrr = 0
    else:
        rrr = (runs_left * 6) / balls_left

    # Input dataframe (IMPORTANT: correct feature name)
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Prediction
    result = pipe.predict_proba(input_df)

    loss = result[0][0]
    win = result[0][1]

    st.subheader("Match Prediction")

    st.text(f"{batting_team} Win Chance: {round(win * 100)}%")
    st.text(f"{bowling_team} Win Chance: {round(loss * 100)}%")