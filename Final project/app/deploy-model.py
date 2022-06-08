import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Prediction App
This app predicts the **House Price**!
""")
st.write('---')

# Loads the House Price Dataset
houses = datasets.load_boston()
X = pd.DataFrame(houses.data, columns=houses.feature_names)
Y = pd.DataFrame(houses.target, columns=["MEDV"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Input Parameters')

def user_input_features():
    bed = st.sidebar.slider('bedroom', X.bed.min(), X.bed.max(), X.bed.mean())
    bath = st.sidebar.slider('bathroom', X.bath.min(), X.bath.max(), X.bath.mean())
    floor = st.sidebar.slider('floor', X.floor.min(), X.floor.max(), X.floor.mean())
    wf = st.sidebar.slider('waterfront', X.wf.min(), X.wf.max(), X.wf.mean())
    view = st.sidebar.slider('view', X.view.min(), X.view.max(), X.view.mean())
    cond = st.sidebar.slider('conditions', X.cond.min(), X.cond.max(), X.cond.mean())
    grade = st.sidebar.slider('grade', X.grade.min(), X.grade.max(), X.grade.mean())
    price = st.sidebar.slider('price', X.price.min(), X.price.max(), X.price.mean())
    code = st.sidebar.slider('zipcode', X.code.min(), X.code.max(), X.code.mean())
    lat = st.sidebar.slider('lat', X.lat.min(), X.lat.max(), X.lat.mean())
    long = st.sidebar.slider('long', X.long.min(), X.long.max(), X.long.mean())
    data = {'bedrooms': bed,
            'bathroom': bath,
            'floor': INDUS,
            'waterfront': wf,
            'view': view,
            'conditions': cond,
            'grade': grade,
            'price': price,
            'zipcode': code,
            'lat': lat,
            'long': long }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')
