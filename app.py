import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF

# 1. Настройка страницы
st.set_page_config(page_title="Allergy Predictor Pro", page_icon="NA", layout="wide")

@st.cache_resource
def load_data():
    # Загружаем файл, который теперь содержит Pipeline (Scale + Model)
    return joblib.load('allergy_predictor_v1.pkl')

data = load_data()
models = data['models']
X_columns = data['X_columns']

# 2. Интерфейс
st.title("Прогноз риска развития пищевых аллергий")
st.markdown("Данная система использует машинное обучение для оценки вероятности наличия 15 видов пищевых аллергий на основе демографических данных и истории атопических заболеваний.")

# Боковая панель для ввода
st.sidebar.header("Данные пациента")
birth_year = st.sidebar.number_input("Год рождения", 1980, 2026, 2018)
gender = st.sidebar.selectbox("Пол", ["S0 - Male", "S1 - Female"])
race = st.sidebar.selectbox("Раса", ["R0 - White", "R1 - Black", "R2 - Asian", "R3 - Other"])
has_asthma = st.sidebar.checkbox("Астма в анамнезе")
has_eczema = st.sidebar.checkbox("Экзема в анамнезе")

# Логика расчета
if st.button("Рассчитать риски для всех аллергенов"):
    user_input = {
        'BIRTH_YEAR': birth_year,
        'GENDER_FACTOR': gender,
        'RACE_FACTOR': race,
        'ETHNICITY_FACTOR': 'E0 - Non-Hispanic',
        'HAS_ASTHMA': int(has_asthma),
        'HAS_ECZEMA': int(has_eczema)
    }

    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df).reindex(columns=X_columns, fill_value=0)

    # Вывод карточек с рисками (сетка 5x3)
    st.subheader("Вероятность по типам аллергенов")
    all_names = list(models.keys())

    # Группируем по 3 для отображения в ряд
    for i in range(0, len(all_names), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(all_names):
                name = all_names[i + j]
                # Модель сама применит StandardScaler внутри Pipeline
                prob = models[name].predict_proba(input_encoded)[0][1]
                risk = round(prob * 100, 1)

                with cols[j]:
                    # Цветовой индикатор
                    status = "Высокий" if risk > 40 else "Средний" if risk > 15 else "Низкий"
                    st.metric(label=name.upper(), value=f"{risk}%", delta=status, delta_color="inverse")

    st.divider()

    # Графики и отчет
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Анализ ключевых факторов")
        # Берем коэффициенты из Pipeline (LogisticRegression — второй шаг)
        example_model = models['peanut'].steps[1][1]
        importance_df = pd.DataFrame({
            'Признак': X_columns,
            'Влияние': example_model.coef_[0]
        }).sort_values('Влияние')

        fig, ax = plt.subplots()
        colors = ['#ff9999' if x > 0 else '#66b3ff' for x in importance_df['Влияние']]
        ax.barh(importance_df['Признак'], importance_df['Влияние'], color=colors)
        st.pyplot(fig)
        st.caption("Красный — увеличивает риск, Синий — снижает.")

    with col_right:
        st.subheader("Сохранение отчета")
        st.write("Сгенерируйте документ с результатами для консультации с врачом.")

        # Создание PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Allergy Risk Assessment Report", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Patient Birth Year: {birth_year}", ln=True)
        pdf.ln(5)
        for name in all_names:
            p = models[name].predict_proba(input_encoded)[0][1]
            pdf.cell(200, 10, txt=f"- {name.upper()}: {round(p*100, 1)}%", ln=True)

        pdf_output = pdf.output(dest='S').encode('latin-1')
        st.download_button(label="Скачать PDF отчет", data=pdf_output, file_name="allergy_report.pdf", mime="application/pdf")

st.info("⚠Внимание: Данная система является демонстрационной. Для постановки диагноза обратитесь к специалисту.")