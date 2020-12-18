import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

@st.cache
def load_data():
    data = pd.read_csv('./data/diabetes_data_upload.csv')
    data.replace(['Yes','Positive','No','Negative'],[1,1,0,0],inplace=True)
    data.drop(['Gender'], axis=1, inplace=True)
    return data

data = load_data()

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

intro_markdown = read_markdown_file("./mds/header.md")
st.markdown(intro_markdown, unsafe_allow_html=True)

data_head = st.checkbox("Show sample data")
if data_head:
    st.write(data.head())

about_markdown = read_markdown_file("./mds/about.md")
st.markdown(about_markdown, unsafe_allow_html=True)

st.subheader("This form does not intend diagnose or prevent any disease.")

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
    if sum([cp1,cp2,csw,cw ,cp3,cgt,cvb,cit,cir,cdh,cpp,cms,cal,cob]) == 0:
        st.write("Please check the symptoms that apply!")
    else:
        y_pred = forest.predict(X_pred)
        if y_pred == 0:
            st.write("The symptoms selected do not seem to be caused by diabetes.")
        else:
            st.write("It is recommended to visit your doctor to check the cause of the symptoms.")

