import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px

st.title('🧊 Penguin Classifier with 3D Visualization')


# 파일 업로드
penguin_file = st.file_uploader('📂 Upload your own penguin data (CSV format)')

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
    features_for_plot = penguin_df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']].copy()  # 시각화용 원본 저장

    features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm',
                           'flipper_length_mm', 'body_mass_g', 'sex']]
    features = pd.get_dummies(features)
    output, unique_penguin_mapping = pd.factorize(output)

    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=.8)
    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(x_train, y_train)

    y_pred = rfc.predict(x_test)
    score = round(accuracy_score(y_pred, y_test), 2)
    st.success(f"🎯 We trained a Random Forest model on these data. It has a score of {score}!")

# 예측 폼
with st.form('user_inputs'):
    island = st.selectbox('🗺️ Penguin Island', options=['Biscoe', 'Dream', 'Torgerson'])
    sex = st.selectbox('⚧️ Sex', options=['Female', 'Male'])
    bill_length = st.number_input('📏 Bill Length (mm)', min_value=0)
    bill_depth = st.number_input('📐 Bill Depth (mm)', min_value=0)
    flipper_length = st.number_input('🏊 Flipper Length (mm)', min_value=0)
    body_mass = st.number_input('⚖️ Body Mass (g)', min_value=0)
    st.form_submit_button()

# 수동 인코딩
island_biscoe = 1 if island == 'Biscoe' else 0
island_dream = 1 if island == 'Dream' else 0
island_torgerson = 1 if island == 'Torgerson' else 0
sex_female = 1 if sex == 'Female' else 0
sex_male = 1 if sex == 'Male' else 0

# 예측
new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length,
                               body_mass, island_biscoe, island_dream,
                               island_torgerson, sex_female, sex_male]])
prediction_species = unique_penguin_mapping[new_prediction][0]
st.write(f'🧠 We predict your penguin is of the **{prediction_species}** species!')

# Plotly 3D 시각화
if penguin_file is not None and st.button("📊 Show 3D Interactive Visualization"):
    st.subheader("3D Feature Space (Predicted Species)")
    features_plot = features_for_plot.copy()
    features_plot['prediction'] = rfc.predict(features)
    features_plot['species'] = features_plot['prediction'].apply(lambda x: unique_penguin_mapping[x])

    fig = px.scatter_3d(
        features_plot,
        x='bill_length_mm',
        y='bill_depth_mm',
        z='flipper_length_mm',
        color='species',
        symbol='species',
        title="Penguins in 3D Feature Space",
        labels={
            'bill_length_mm': 'Bill Length (mm)',
            'bill_depth_mm': 'Bill Depth (mm)',
            'flipper_length_mm': 'Flipper Length (mm)',
            'species': 'Predicted Species'
        }
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color='black')))
    fig.update_layout(scene=dict(
        xaxis_title='Bill Length',
        yaxis_title='Bill Depth',
        zaxis_title='Flipper Length'
    ))

    st.plotly_chart(fig, use_container_width=True)
