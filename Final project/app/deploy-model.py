import streamlit as st
import pandas as pd
import pickle as pkl


model = pkl.load(open('../models/model.pkl', 'rb'))

st.write("""
# Predicción del precio de la casa
""")

bedrooms = st.number_input('Introduzca el número de habitaciones', value=0)
bathrooms = st.number_input('Introduzca el número de baños', value=0)
floor = st.number_input('Introduzca el número de pisos', value=0)
sqft_living = st.number_input('Introduzca el número de metros cuadrados', value=0)
    

if st.button('Calcular'):
    data = pd.DataFrame({'bedrooms': [bedrooms], 'bathrooms': [bathrooms],
            'floor': [floor],
            'sqft_living': [sqft_living]})


    data

    prediction = model.predict([[sqft_living, bathrooms+floor+bedrooms]])

    st.success('La casa costará {} euros'.format(int(prediction[0])))


           