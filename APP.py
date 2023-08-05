# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 00:33:04 2023

@author: Hp
"""

import pandas as pd
import numpy as np 
import streamlit as st 
from sklearn.ensemble import RandomForestClassifier
import pickle 

pickle_in = open('Model_Kmeans_new','rb')
# Load the pickled object (e.g., K-Means model) from a file
with open("Model_Kmeans_new", 'rb') as file:
    loaded_model = pickle.load(open("Model_Kmeans_new"))
    
df=pd.read_csv(r"C:\Users\Hp\Downloads\Clustered_ World_Development_Data.csv")
st.title('Prediction on Global Development Measurement')
st.sidebar.header('User Input Parameters')

def user_input_features():
    birth_rate = st.sidebar.number_input("Birth Rate")
    business_tax_rate = st.sidebar.number_input("Business Tax Rate",max_value=60.00)
    co2_emission = st.sidebar.number_input('CO2 Emissions',min_value=0.000 )
    days_to_start_business = st.sidebar.number_input('No. of Days to Start Business',max_value=80.00)
    energy_usage = st.sidebar.number_input('Energy Usage',min_value=0.000)
    GDP = st.sidebar.number_input('GDP ($)',min_value=0.000)
    health_exp_percent_GDP = st.sidebar.number_input('Health Exp % GDP value')
    health_exp_percapita = st.sidebar.number_input('Health Exp/Capita',min_value=0.000)
    hours_to_do_tax = st.sidebar.number_input('Hours to do Tax',max_value=600.00)
    infant_mortality_rate = st.sidebar.number_input('Infant Mortality Rate')
    internet_usage = st.sidebar.number_input('Internet Usage duration')
    lending_interest = st.sidebar.number_input('Lending Interest rate')
    life_expectancy_female = st.sidebar.number_input('Life Expectancy Female')
    life_expectancy_male = st.sidebar.number_input('Life Expectancy Male')
    mobile_phone_usage = st.sidebar.number_input('Mobile Phone Usage duration')
    population_0_14 =  st.sidebar.number_input('Population between 0-14 Years')
    population_15_64 =  st.sidebar.number_input('Population between 15-64 Years')
    population_65_plus =  st.sidebar.number_input('Population between 65+ Years')
    population_total = st.sidebar.number_input('Population Total',min_value=0.000)
    population_urban = st.sidebar.number_input('Population Urban')
    tourism_inbound = st.sidebar.number_input('Tourism Inbound ($)',min_value=0.000)
    tourism_outbound = st.sidebar.number_input('Tourism Outbound ($)',min_value=0.000)
    country = st.sidebar.text_input(' enter country')
    
    data = {'birth_rate' : birth_rate, 'business_tax_rate' : business_tax_rate, 'co2_emission': co2_emission,
       'days_to_start_business' : days_to_start_business, 'energy_usage' : energy_usage,
       'GDP':GDP, 'health_exp_percent_GDP': health_exp_percent_GDP,'health_exp_percapita': health_exp_percapita,
       'hours_to_do_tax':hours_to_do_tax, 'infant_mortality_rate' : infant_mortality_rate, 'internet_usage':internet_usage,
       'lending_interest': lending_interest, 'life_expectancy_female':life_expectancy_female, 'life_expectancy_male':life_expectancy_male,
       'mobile_phone_usage' : mobile_phone_usage, 'population_0_14' : population_0_14 , 'population_15_64' : population_15_64 ,
       'population_65_plus' : population_65_plus , 'population_total' : population_total , 
       'population_urban' : population_urban, 'tourism_inbound' : tourism_inbound,'tourism_outbound': tourism_outbound,'country': country}
    
    features = pd.DataFrame(data,index = [0])
    features = features.drop(columns=['country'], axis=1)
    return features 
df = user_input_features()
st.subheader('User Input parameters')
st.write(df) 



    
# load the model from disk
loaded_model = load(open('Model_Kmeans_new','rb'))
prediction = loaded_model.predict(df)
st.subheader('Predicted Result')


def analysis(prediction):
    if prediction == 0:
        return 'Developing Country'
    elif prediction == 1:
        return 'Under Developed Country'
    elif prediction == 2:
        return 'Developed Country'
    else:
        return 'Small Country'

# st.write(analysis(prediction))

if st.button("Predict"):
    st.write(analysis(prediction))
    
   

