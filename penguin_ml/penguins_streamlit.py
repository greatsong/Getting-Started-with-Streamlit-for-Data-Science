import streamlit as st
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd 
import pickle 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
 
st.title('Penguin Classifier') 

st.write("This app uses 6 inputs to predict the species of penguin using " 

         "a model built on the Palmer's Penguin's dataset. Use the form below" 

         " to get started!") 
  

penguin_file = st.file_uploader('Upload your own penguin data') 

if penguin_file is None: 

    rf_pickle = open('random_forest_penguin.pickle', 'rb') 

    map_pickle = open('output_penguin.pickle', 'rb') 

    rfc = pickle.load(rf_pickle) 

    unique_penguin_mapping = pickle.load(map_pickle) 

    rf_pickle.close() 

    map_pickle.close() 

else: 

    penguin_df = pd.read_csv(penguin_file) 

    penguin_df = penguin_df.dropna() 

    output = penguin_df['species'] 

    features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm', 

                           'flipper_length_mm', 'body_mass_g', 'sex']] 

    features = pd.get_dummies(features) 

    output, unique_penguin_mapping = pd.factorize(output) 

 

    x_train, x_test, y_train, y_test = train_test_split( 

        features, output, test_size=.8) 

    rfc = RandomForestClassifier(random_state=15) 

    rfc.fit(x_train, y_train) 

    y_pred = rfc.predict(x_test) 

    score = round(accuracy_score(y_pred, y_test), 2) 

    st.write('We trained a Random Forest model on these data,' 
             ' it has a score of {}! Use the ' 
             'inputs below to try out the model.'.format(score))

with st.form('user_inputs'): 
  island = st.selectbox('Penguin Island', options=[
    'Biscoe', 'Dream', 'Torgerson']) 
  sex = st.selectbox('Sex', options=[
    'Female', 'Male']) 
  bill_length = st.number_input(
    'Bill Length (mm)', min_value=0) 
  bill_depth = st.number_input(
    'Bill Depth (mm)', min_value=0) 
  flipper_length = st.number_input(
    'Flipper Length (mm)', min_value=0) 
  body_mass = st.number_input(
    'Body Mass (g)', min_value=0) 
  st.form_submit_button() 



island_biscoe, island_dream, island_torgerson = 0, 0, 0 
if island == 'Biscoe': 
  island_biscoe = 1 
elif island == 'Dream': 
  island_dream = 1 
elif island == 'Torgerson': 
  island_torgerson = 1 

sex_female, sex_male = 0, 0 

if sex == 'Female': 
  sex_female = 1 

elif sex == 'Male': 
  sex_male = 1 


new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length, 
  body_mass, island_biscoe, island_dream, 
  island_torgerson, sex_female, sex_male]]) 
prediction_species = unique_penguin_mapping[new_prediction][0]
st.write('We predict your penguin is of the {} species'.format(prediction_species)) 


from mpl_toolkits.mplot3d import Axes3D  # 3D 시각화를 위해 필요
from sklearn.decomposition import PCA

# 시각화 버튼
if st.button("3D Feature Space Visualization"):

    st.subheader("Penguin Dataset in 3D Space (Colored by Species Prediction)")

    # 선택한 주요 특성
    feature_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']
    
    # 예측값 생성
    predicted = rfc.predict(features)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        features['bill_length_mm'],
        features['bill_depth_mm'],
        features['flipper_length_mm'],
        c=predicted,
        cmap='viridis',
        edgecolor='k',
        s=60
    )

    ax.set_xlabel('Bill Length (mm)')
    ax.set_ylabel('Bill Depth (mm)')
    ax.set_zlabel('Flipper Length (mm)')
    ax.set_title('3D Feature Space of Penguins (Predicted Classes)')

    # 범례 추가
    legend_labels = [unique_penguin_mapping[i] for i in range(len(set(predicted)))]
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                 label=label, markerfacecolor=scatter.cmap(scatter.norm(i)),
                                 markersize=10) for i, label in enumerate(legend_labels)]
    ax.legend(handles=legend_handles)

    st.pyplot(fig)


import plotly.express as px

# Plotly 3D 시각화 버튼
if st.button("📊 Show 3D Interactive Visualization with Plotly"):

    st.subheader("Interactive 3D Scatter Plot (by Predicted Species)")

    # 예측값 추가
    features_plot = features.copy()
    features_plot['prediction'] = rfc.predict(features)
    features_plot['species_name'] = features_plot['prediction'].apply(lambda x: unique_penguin_mapping[x])

    # Plotly 3D 산점도
    fig = px.scatter_3d(
        features_plot,
        x='bill_length_mm',
        y='bill_depth_mm',
        z='flipper_length_mm',
        color='species_name',
        symbol='species_name',
        title="Penguins in 3D Feature Space",
        labels={
            'bill_length_mm': 'Bill Length (mm)',
            'bill_depth_mm': 'Bill Depth (mm)',
            'flipper_length_mm': 'Flipper Length (mm)',
            'species_name': 'Predicted Species'
        }
    )

    fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='DarkSlateGrey')))
    fig.update_layout(scene=dict(
        xaxis_title='Bill Length',
        yaxis_title='Bill Depth',
        zaxis_title='Flipper Length'
    ))

    st.plotly_chart(fig, use_container_width=True)

