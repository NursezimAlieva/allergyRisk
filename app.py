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
    # Убедись, что файл модели называется именно так
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

    predictions = {}
    for i, (name, model) in enumerate(models.items()):
        prob = model.predict_proba(input_encoded)[0][1]
        risk_percent = round(float(prob) * 100, 1)
        predictions[name] = risk_percent

        with cols[i % 3]:
            color = "red" if risk_percent > 50 else "orange" if risk_percent > 20 else "green"
            st.markdown(f"**{name.upper()}**")
            st.markdown(f"<h2 style='color: {color};'>{risk_percent}%</h2>", unsafe_allow_html=True)

    st.divider()

    # --- ШАГ 1: ВИЗУАЛИЗАЦИЯ ФАКТОРОВ РИСКА ---
    st.subheader("📊 Анализ факторов влияния")
    st.write("Ниже показано, какие признаки сильнее всего повлияли на прогноз:")

    target_model = models['peanut']

    # Извлекаем важность признаков в зависимости от типа модели
    if hasattr(target_model, 'coef_'):
        # Для Логистической регрессии
        importances = target_model.coef_[0]
    elif hasattr(target_model, 'feature_importances_'):
        # Для Random Forest
        importances = target_model.feature_importances_
    elif hasattr(target_model, 'calibrated_classifiers_'):
        # Для калиброванных моделей
        clf = target_model.calibrated_classifiers_[0].base_estimator
        importances = clf.coef_[0] if hasattr(clf, 'coef_') else clf.feature_importances_
    else:
        importances = np.zeros(len(X_columns))

    # Создаем DataFrame для графика
    importance_df = pd.DataFrame({
        'Feature': X_columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    # Строим график
    fig, ax = plt.subplots(figsize=(10, 6))

    # Если это RandomForest, все значения положительные. Если регрессия — могут быть отрицательные.
    if hasattr(target_model, 'coef_'):
        colors = ['#ff9999' if x > 0 else '#66b3ff' for x in importance_df['Importance']]
    else:
        colors = 'skyblue' # Для леса просто один цвет

    ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    ax.set_xlabel('Степень влияния на результат')
    ax.set_title('Важность признаков в модели')

    st.pyplot(fig)

st.info("Примечание: Данная модель носит ознакомительный характер и не является медицинским диагнозом.")