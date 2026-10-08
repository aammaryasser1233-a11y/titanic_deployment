import streamlit as st
import joblib as jb
import pandas as pd
import numpy as np

model =jb.load("Ammar_dt_model.pkl")

st.title("🚢 Titanc project 🚢")
st.write("enter passenger data to now if he/she survived or no")

pclass = st.selectbox("passenger class (Pclass)" , [1,2,3])
Sex = st.selectbox("passenger sex (Sex)" , ["male" , "female"])
Embarked = st.selectbox("Embarked" , ["S", "C", "Q"])

Age = st.number_input("Age" , min_value =0 , max_value =100)
Sibsp = st.number_input("SibSp" , min_value =0 , max_value =10)
Parch = st.number_input("parents/children" , min_value =0 , max_value =10)
Fare = st.number_input("fare" , min_value =0 )

if st.button("predict survival"):
  input = {"pclass" : pclass ,
           "Sex" : Sex ,
           "Embarked" : Embarked ,
           "Age" : Age ,
           "Sibsp" : Sibsp ,
           "Parch" : Parch ,
           "Fare" : Fare
           }
  input_df = pd.DataFrame(input)
  input_df.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

  output = model.predict(input_df)

  if output[0] == 1:
    st.success("the passenger survived ✅")
  else:
    st.error("the passenger didn't survived ❌")

