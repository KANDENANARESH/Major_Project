# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# # -------------------------------
# # Initialize session state
# # -------------------------------
# if 'page' not in st.session_state:
#     st.session_state.page = 1

# # -------------------------------
# # Load datasets
# # -------------------------------
# @st.cache_data
# def load_data():
#     crop_rec = pd.read_csv('Crop_recommendation.csv')
#     crop_yield = pd.read_csv('crop_yield.csv')
#     fertilizer = pd.read_csv('Fertilizer.csv')
#     return crop_rec, crop_yield, fertilizer

# crop_rec, crop_yield, fertilizer = load_data()

# # Normalize names
# crop_rec['label'] = crop_rec['label'].str.strip().str.lower()
# crop_yield['Crop'] = crop_yield['Crop'].str.strip().str.lower()
# crop_yield['State'] = crop_yield['State'].str.strip().str.lower()
# crop_yield['Season'] = crop_yield['Season'].str.strip().str.lower()
# fertilizer['Fertilizer Name'] = fertilizer['Fertilizer Name'].str.strip().str.lower()

# # -------------------------------
# # Cached model training
# # -------------------------------
# @st.cache_resource
# def train_models(crop_rec, crop_yield):
#     # Crop Recommendation
#     X_crop = crop_rec[['N','P','K','temperature','humidity','ph','rainfall']]
#     y_crop = crop_rec['label']
#     le_crop = LabelEncoder()
#     y_crop_enc = le_crop.fit_transform(y_crop)
#     clf_crop = RandomForestClassifier(random_state=42)
#     clf_crop.fit(X_crop, y_crop_enc)

#     # Production & Yield
#     le_crop_yield = LabelEncoder()
#     crop_yield['Crop_enc'] = le_crop_yield.fit_transform(crop_yield['Crop'])
#     le_state = LabelEncoder()
#     crop_yield['State_enc'] = le_state.fit_transform(crop_yield['State'])
#     le_season = LabelEncoder()
#     crop_yield['Season_enc'] = le_season.fit_transform(crop_yield['Season'])

#     X_prod = crop_yield[['Crop_enc','State_enc','Area','Season_enc']]
#     y_prod = crop_yield['Production']
#     y_yield = crop_yield['Yield']

#     reg_prod = RandomForestRegressor(random_state=42)
#     reg_prod.fit(X_prod, y_prod)

#     reg_yield = RandomForestRegressor(random_state=42)
#     reg_yield.fit(X_prod, y_yield)

#     return clf_crop, le_crop, reg_prod, reg_yield, le_crop_yield, le_state, le_season

# # Train once
# clf_crop, le_crop, reg_prod, reg_yield, le_crop_yield, le_state, le_season = train_models(crop_rec, crop_yield)

# # -------------------------------
# # Fertilizer Recommendation Function
# # -------------------------------
# def recommend_fertilizer(N, P, K, crop_name=None):
#     if crop_name:
#         f = fertilizer[fertilizer['Fertilizer Name'].str.contains(crop_name, case=False, na=False)]
#         if not f.empty:
#             return f['Fertilizer Name'].values[0]
#     fertilizer['diff'] = np.sqrt(
#         (fertilizer['Nitrogen'] - N)**2 +
#         (fertilizer['Phosphorous'] - P)**2 +
#         (fertilizer['Potassium'] - K)**2
#     )
#     best = fertilizer.loc[fertilizer['diff'].idxmin()]
#     return best['Fertilizer Name']

# # -------------------------------
# # Fallback for unseen crops
# # -------------------------------
# def fallback_production_yield(area):
#     if area <= 2:  # small farm
#         production = np.random.uniform(50, 200)
#         yield_val = np.random.uniform(20, 50)
#     elif area <= 5:  # medium farm
#         production = np.random.uniform(200, 500)
#         yield_val = np.random.uniform(50, 100)
#     else:  # large farm
#         production = np.random.uniform(500, 1000)
#         yield_val = np.random.uniform(100, 200)
#     return production, yield_val

# # -------------------------------
# # Page Navigation Functions
# # -------------------------------
# def go_to_page(page_num):
#     st.session_state.page = page_num

# # -------------------------------
# # Page 1: Start Page
# # -------------------------------
# if st.session_state.page == 1:
#     st.title("🌱 Crop & Fertilizer Recommendation System")
#     st.write("Click the button below to start the prediction process.")
#     if st.button("🚀 Start"):
#         go_to_page(2)

# # -------------------------------
# # Page 2: Input Page
# # -------------------------------
# elif st.session_state.page == 2:
#     st.title("📥 Enter Input Details")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90)
#         P = st.number_input("Phosphorous (P)", min_value=0, max_value=200, value=40)
#         K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=40)
#         temperature = st.number_input("Temperature (°C)", min_value=0, max_value=50, value=26)
    
#     with col2:
#         humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=80)
#         ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
#         rainfall = st.number_input("Rainfall (mm)", min_value=0, max_value=500, value=150)
#         area = st.number_input("Area (hectares)", min_value=0.1, max_value=100.0, value=2.5)
    
#     state = st.selectbox("State", crop_yield['State'].unique())
#     season = st.selectbox("Season", crop_yield['Season'].unique())
    
#     if st.button("✅ Predict"):
#         st.session_state.inputs = {
#             'N': N, 'P': P, 'K': K, 'temperature': temperature,
#             'humidity': humidity, 'ph': ph, 'rainfall': rainfall,
#             'area': area, 'state': state, 'season': season
#         }
#         go_to_page(3)

# # -------------------------------
# # Page 3: Result Page
# # -------------------------------
# elif st.session_state.page == 3:
#     st.title("📊 Prediction Results")
    
#     inp = st.session_state.inputs
#     soil_input = [inp['N'], inp['P'], inp['K'], inp['temperature'], inp['humidity'], inp['ph'], inp['rainfall']]
    
#     # Predict crop
#     pred_crop_code = clf_crop.predict([soil_input])[0]
#     pred_crop_name = le_crop.inverse_transform([pred_crop_code])[0]
    
#     # Predict production & yield
#     if pred_crop_name in le_crop_yield.classes_:
#         crop_enc = le_crop_yield.transform([pred_crop_name])[0]
#         state_enc = le_state.transform([inp['state'].strip().lower()])[0]
#         season_enc = le_season.transform([inp['season'].strip().lower()])[0]
#         X_pred_prod = [[crop_enc, state_enc, inp['area'], season_enc]]
#         pred_production = reg_prod.predict(X_pred_prod)[0]
#         pred_yield_val = reg_yield.predict(X_pred_prod)[0]
#     else:
#         pred_production, pred_yield_val = fallback_production_yield(inp['area'])

    
#     # Fertilizer recommendation
#     fert_reco = recommend_fertilizer(inp['N'], inp['P'], inp['K'], pred_crop_name)
    
#     # Display results
#     st.success(f"✅ Recommended Crop: **{pred_crop_name.title()}**")
#     st.info(f"🌾 Predicted Production: **{round(pred_production,2)} units**")
#     st.info(f"📈 Predicted Yield: **{round(pred_yield_val,2)} units/ha**")
#     st.warning(f"💧 Recommended Fertilizer: **{fert_reco.title()}**")
    
#     if st.button("🔄 Start Over"):
#         go_to_page(1)



















from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

app = Flask(__name__)

# ---------------------------------
# Load datasets
# ---------------------------------
crop_rec = pd.read_csv('Crop_recommendation.csv')
crop_yield = pd.read_csv('crop_yield.csv')
fertilizer = pd.read_csv('Fertilizer.csv')

# Normalize text
crop_rec['label'] = crop_rec['label'].str.strip().str.lower()
crop_yield['Crop'] = crop_yield['Crop'].str.strip().str.lower()
crop_yield['State'] = crop_yield['State'].str.strip().str.lower()
crop_yield['Season'] = crop_yield['Season'].str.strip().str.lower()
fertilizer['Fertilizer Name'] = fertilizer['Fertilizer Name'].str.strip().str.lower()

# ---------------------------------
# Train models
# ---------------------------------
def train_models():
    # Crop recommendation
    X_crop = crop_rec[['N','P','K','temperature','humidity','ph','rainfall']]
    y_crop = crop_rec['label']
    le_crop = LabelEncoder()
    y_crop_enc = le_crop.fit_transform(y_crop)
    clf_crop = RandomForestClassifier(random_state=42)
    clf_crop.fit(X_crop, y_crop_enc)

    # Production / Yield
    le_crop_yield = LabelEncoder()
    crop_yield['Crop_enc'] = le_crop_yield.fit_transform(crop_yield['Crop'])
    le_state = LabelEncoder()
    crop_yield['State_enc'] = le_state.fit_transform(crop_yield['State'])
    le_season = LabelEncoder()
    crop_yield['Season_enc'] = le_season.fit_transform(crop_yield['Season'])

    X_prod = crop_yield[['Crop_enc','State_enc','Area','Season_enc']]
    y_prod = crop_yield['Production']
    y_yield = crop_yield['Yield']

    reg_prod = RandomForestRegressor(random_state=42)
    reg_prod.fit(X_prod, y_prod)
    reg_yield = RandomForestRegressor(random_state=42)
    reg_yield.fit(X_prod, y_yield)

    return clf_crop, le_crop, reg_prod, reg_yield, le_crop_yield, le_state, le_season

clf_crop, le_crop, reg_prod, reg_yield, le_crop_yield, le_state, le_season = train_models()

# ---------------------------------
# Helper functions
# ---------------------------------
def recommend_fertilizer(N, P, K, crop_name=None):
    if crop_name:
        f = fertilizer[fertilizer['Fertilizer Name'].str.contains(crop_name, case=False, na=False)]
        if not f.empty:
            return f['Fertilizer Name'].values[0]
    fertilizer['diff'] = np.sqrt(
        (fertilizer['Nitrogen'] - N)**2 +
        (fertilizer['Phosphorous'] - P)**2 +
        (fertilizer['Potassium'] - K)**2
    )
    best = fertilizer.loc[fertilizer['diff'].idxmin()]
    return best['Fertilizer Name']

def fallback_production_yield(area):
    if area <= 2:
        return np.random.uniform(50, 200), np.random.uniform(20, 50)
    elif area <= 5:
        return np.random.uniform(200, 500), np.random.uniform(50, 100)
    else:
        return np.random.uniform(500, 1000), np.random.uniform(100, 200)

# ---------------------------------
# Routes
# ---------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/input')
def input_page():
    states = sorted(crop_yield['State'].unique())
    seasons = sorted(crop_yield['Season'].unique())
    return render_template('input.html', states=states, seasons=seasons)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    N = float(data['N']); P = float(data['P']); K = float(data['K'])
    temperature = float(data['temperature']); humidity = float(data['humidity'])
    ph = float(data['ph']); rainfall = float(data['rainfall'])
    area = float(data['area']); state = data['state'].strip().lower(); season = data['season'].strip().lower()

    soil_input = [N,P,K,temperature,humidity,ph,rainfall]
    pred_crop_code = clf_crop.predict([soil_input])[0]
    pred_crop_name = le_crop.inverse_transform([pred_crop_code])[0]

    if pred_crop_name in le_crop_yield.classes_:
        crop_enc = le_crop_yield.transform([pred_crop_name])[0]
        state_enc = le_state.transform([state])[0]
        season_enc = le_season.transform([season])[0]
        X_pred = [[crop_enc, state_enc, area, season_enc]]
        pred_production = reg_prod.predict(X_pred)[0]
        pred_yield_val = reg_yield.predict(X_pred)[0]
    else:
        pred_production, pred_yield_val = fallback_production_yield(area)

    fert_reco = recommend_fertilizer(N, P, K, pred_crop_name)

    return render_template(
        'result.html',
        crop=pred_crop_name.title(),
        production=round(pred_production,2),
        yield_val=round(pred_yield_val,2),
        fertilizer=fert_reco.title()
    )

if __name__ == '__main__':
    app.run(debug=True)
