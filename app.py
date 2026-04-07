import streamlit as st
import pandas as pd
import joblib

# Настройка страницы
st.set_page_config(page_title="Allergy Risk Predictor", page_icon="🍎")
st.title("Прогноз риска развития пищевой аллергии")

# Загрузка модели
@st.cache_resource
def load_model():
    return joblib.load('allergy_predictor_v1.pkl')

data = load_model()
models = data['models']
X_columns = data['X_columns']

# Форма ввода данных
st.sidebar.header("Данные пациента")
birth_year = st.sidebar.number_input("Год рождения", min_value=1980, max_value=2026, value=2015)
gender = st.sidebar.selectbox("Пол", ["S0 - Male", "S1 - Female"])
race = st.sidebar.selectbox("Раса", ["R0 - White", "R1 - Black", "R2 - Asian", "R3 - Other"])
ethnicity = st.sidebar.selectbox("Этническая принадлежность", ["E0 - Non-Hispanic", "E1 - Hispanic"])
has_asthma = st.sidebar.checkbox("Есть ли астма?")
has_eczema = st.sidebar.checkbox("Есть ли экзема?")

if st.button("Рассчитать риски"):
    # Подготовка входных данных
    user_input = {
        'BIRTH_YEAR': birth_year,
        'GENDER_FACTOR': gender,
        'RACE_FACTOR': race,
        'ETHNICITY_FACTOR': ethnicity,
        'HAS_ASTHMA': int(has_asthma),
        'HAS_ECZEMA': int(has_eczema)
    }

    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df).reindex(columns=X_columns, fill_value=0)

    # Вывод результатов
    st.subheader("Вероятность наличия аллергии:")
    cols = st.columns(3) # Разделим на колонки для красоты

    for i, (name, model) in enumerate(models.items()):
        prob = model.predict_proba(input_encoded)[0][1]
        risk_percent = round(float(prob) * 100, 1)

        with cols[i % 3]:
            # Цветовая индикация риска
            color = "red" if risk_percent > 50 else "orange" if risk_percent > 20 else "green"
            st.markdown(f"**{name.capitalize()}**")
            st.markdown(f"<h2 style='color: {color};'>{risk_percent}%</h2>", unsafe_allow_html=True)

st.info("Примечание: Данная модель носит ознакомительный характер и не является медицинским диагнозом.")