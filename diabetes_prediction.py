import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

@st.cache
def load_data():
    data = pd.read_csv('./data/diabetes_data_upload.csv')
    data.replace(['Yes','Positive','No','Negative'],[1,1,0,0],inplace=True)
    data.drop(['Gender'], axis=1, inplace=True)
    return data

data = load_data()

st.title('Diabetes Risk Prediction')
# Data app explanation:
#st.write("")

data_head = st.checkbox("Show sample data")
if data_head:
    st.write(data.head())


# Start the form:
st.subheader("Please fill the form below with your age and symptomns to check the diabetes risk: ")

st.write("What is your age?")
age = st.slider("",15,99,25)

st.write("Check all of the following symptoms that apply: ")

cp1 = int(st.checkbox("Polyuria"))
cp2 = int(st.checkbox("Polydipsia"))
csw = int(st.checkbox("Sudden weight loss"))
cw =  int(st.checkbox("Weakness"))
cp3 = int(st.checkbox("Polyphagia"))
cgt = int(st.checkbox("Genital thrush"))
cvb = int(st.checkbox("Visual blurring"))
cit = int(st.checkbox("Itching"))
cir = int(st.checkbox("Irritability"))
cdh = int(st.checkbox("Delayed healing"))
cpp = int(st.checkbox("Partial paresis"))
cms = int(st.checkbox("Muscle stiffness"))
cal = int(st.checkbox("Alopecia"))
cob = int(st.checkbox("Obesity"))

X_pred = [[age,cp1,cp2,csw,cw ,cp3,cgt,cvb,cit,cir,cdh,cpp,cms,cal,cob]]

#Split the data:
y = data['class']
X = data.drop(['class'], axis=1)

# Create random forest model:
forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)
forest.fit(X,y)

if st.button("Check!"):
    y_pred = forest.predict(X_pred)
    if y_pred == 0:
        st.write("The symptoms selected do not seem to be caused by diabetes.")
    else:
        st.write("It is recommended to visit your doctor to check the cause of the symptoms.")