import streamlit as st
import pickle
import os

st.title('Abdu Rahman First ML App')

def load_model():
    with open('C:/Users/hp/Desktop/first_app/iris_model.pkl', 'rb') as f: #f is file name
        model = pickle.load(f)
    return model

#Adding logo
logo_path = "images/parami.jpeg"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

#Adding studnet information
st.sidebar.markdown("**Student Name:** Abdu Rahman")
st.sidebar.markdown("**Student ID:** PIUS20230015")

sepal_length = st.number_input("Sepal Length(cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
sepal_width = st.number_input("Sepal Width(cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
petal_length = st.number_input("Petal Length(cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
petal_width = st.number_input("Petal Width(cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

st.button('Predict')
arr=[[sepal_length,sepal_width,petal_length,petal_width]]
iris_model = load_model()
label_names = ['setosa', 'versicolor', 'virginica']

# Adding images of the flowers according to their names
flowers = {
    'setosa': 'images/setosa.jpeg',
    'versicolor': 'images/versicolor.jpeg',
    'virginica': 'images/virginica.jpeg'
}

results = iris_model.predict(arr)[0]
st.write("The predicted result is", label_names[results])
st.image(flowers[label_names[results]], caption=f"{label_names[results]}", width=400)

