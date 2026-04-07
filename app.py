import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

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

    # 1. Вывод числовых результатов
    st.subheader("Вероятность наличия аллергии:")
    cols = st.columns(3)

    for i, (name, model) in enumerate(models.items()):
        prob = model.predict_proba(input_encoded)[0][1]
        risk_percent = round(float(prob) * 100, 1)

        with cols[i % 3]:
            color = "red" if risk_percent > 50 else "orange" if risk_percent > 20 else "green"
            st.markdown(f"**{name.upper()}**")
            st.markdown(f"<h2 style='color: {color};'>{risk_percent}%</h2>", unsafe_allow_html=True)

    st.divider()

    # --- ГРАФИКИ ---
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # ШАГ 1: ВАЖНОСТЬ ПРИЗНАКОВ
        st.subheader("Факторы влияния")
        target_model = models['peanut']

        if hasattr(target_model, 'coef_'):
            importances = target_model.coef_[0]
        elif hasattr(target_model, 'feature_importances_'):
            importances = target_model.feature_importances_
        else:
            importances = np.zeros(len(X_columns))

        importance_df = pd.DataFrame({'Feature': X_columns, 'Value': importances}).sort_values(by='Value')

        fig1, ax1 = plt.subplots()
        colors = ['#ff9999' if x > 0 else '#66b3ff' for x in importance_df['Value']]
        ax1.barh(importance_df['Feature'], importance_df['Value'], color=colors)
        st.pyplot(fig1)

    with chart_col2:
        # ШАГ 2: ВОЗРАСТНАЯ ДИНАМИКА
        st.subheader("Динамика по годам")

        years = np.arange(1990, 2026, 2)
        trends = []

        # Считаем риск для выбранной аллергии (например, арахис) для разных годов рождения
        for y in years:
            temp_input = user_input.copy()
            temp_input['BIRTH_YEAR'] = y
            temp_df = pd.DataFrame([temp_input])
            temp_encoded = pd.get_dummies(temp_df).reindex(columns=X_columns, fill_value=0)
            prob = models['peanut'].predict_proba(temp_encoded)[0][1]
            trends.append(prob * 100)

        fig2, ax2 = plt.subplots()
        ax2.plot(years, trends, marker='o', color='green', linestyle='--')
        ax2.set_xlabel("Год рождения")
        ax2.set_ylabel("Риск %")
        st.pyplot(fig2)

st.info("Примечание: Данная модель носит ознакомительный характер и не является медицинским диагнозом.")