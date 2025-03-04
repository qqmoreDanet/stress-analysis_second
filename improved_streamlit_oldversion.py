import streamlit as st

# Move set_page_config to the very top, before any other Streamlit commands
st.set_page_config(
    page_title="Анализ стресса",
    page_icon="❤️",
    layout="wide"
)

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore, pearsonr
import io
import base64
import os
from functools import lru_cache

# Константы
MODEL_PATH = './models/generator_model.keras'
SCALER_PATH = './models/generator_scaler.pkl'
LABELS = {0: 'M-I', 1: 'M-IIa', 2: 'M-IIb', 3: 'M-III', 4: 'M-IV'}
FEATURES = ['HR', 'RESP', 'foot GSR', 'hand GSR']
SEQ_LENGTH = 10

# Цветовая карта для уровней стресса
STRESS_COLORS = {
    'M-I': '#4CAF50',  # Зеленый - низкий стресс
    'M-IIa': '#8BC34A',  # Светло-зеленый
    'M-IIb': '#FFEB3B',  # Желтый
    'M-III': '#FF9800',  # Оранжевый
    'M-IV': '#F44336'  # Красный - высокий стресс
}

# Создание папки для сессий
os.makedirs('./sessions', exist_ok=True)


# Загрузка модели и скейлера с кешированием
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None


@st.cache_resource
def load_scaler():
    try:
        return joblib.load(SCALER_PATH)
    except Exception as e:
        st.error(f"Ошибка загрузки скейлера: {e}")
        return None


# Инициализация модели и скейлера при запуске
model = load_model()
scaler = load_scaler()


def load_csv(file, column_mapping=None):
    """Загрузка и предобработка данных из CSV/DAT файла"""
    try:
        # Определение разделителя на основе расширения файла
        separator = '\t' if file.name.endswith('.dat') else ','
        data = pd.read_csv(file, encoding='utf-8-sig', sep=separator, decimal='.')

        # Применяем сопоставление колонок, если предоставлено
        if column_mapping:
            data = data.rename(columns=column_mapping)

        # Проверяем наличие необходимых признаков
        available_features = [f for f in FEATURES if f in data.columns]
        if len(available_features) < len(FEATURES):
            missing = set(FEATURES) - set(available_features)
            st.warning(
                f"Внимание: отсутствуют колонки: {', '.join(missing)}. Будут использованы только доступные данные.")

        # Получаем временные метки или генерируем последовательность
        timestamps = data.iloc[:, 0] if data.shape[1] > len(available_features) else pd.Series(range(len(data)))

        # Заполняем пропущенные значения средним
        for feature in available_features:
            if data[feature].isnull().any():
                data[feature] = data[feature].fillna(data[feature].mean())

        # Удаляем выбросы (Z-score > 3)
        for feature in available_features:
            z_scores = zscore(data[feature], nan_policy='omit')
            outliers = np.abs(z_scores) > 3
            if np.sum(outliers) > 0:
                data.loc[outliers, feature] = data[feature].mean()

        # Сохраняем оригинальные данные и масштабируем признаки, если доступны все
        original_data = data[available_features].copy()
        scaled_data = None
        if set(available_features) == set(FEATURES):
            scaled_data = scaler.transform(data[FEATURES])

        return timestamps, original_data, scaled_data
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
        st.stop()


@lru_cache(maxsize=32)
def predict_stress(X_scaled_tuple):
    """Предсказание уровня стресса на основе масштабированных данных"""
    # Преобразуем кортеж обратно в массив
    X_scaled = np.array(X_scaled_tuple)

    if X_scaled is None or len(X_scaled) < SEQ_LENGTH:
        return None

    # Создаем последовательности для модели
    sequences = np.array([X_scaled[i:i + SEQ_LENGTH] for i in range(len(X_scaled) - SEQ_LENGTH + 1)])
    predictions = model.predict(sequences, verbose=0)
    return np.argmax(predictions, axis=1)


def calculate_hrv_metrics(hr_data):
    """Расчет метрик вариабельности сердечного ритма"""
    try:
        rri = (60 * 1000) / hr_data  # RR интервалы в мс
        sdnn = np.std(rri)  # Стандартное отклонение RR интервалов
        rmssd = np.sqrt(np.mean(np.diff(rri) ** 2))  # Корень из среднего квадрата разностей

        return {
            'SDNN': sdnn,
            'RMSSD': rmssd,
            'HR_mean': np.mean(hr_data),
            'HR_std': np.std(hr_data)
        }
    except Exception as e:
        st.error(f"Ошибка при расчете HRV: {e}")
        return {
            'SDNN': None,
            'RMSSD': None,
            'HR_mean': np.mean(hr_data) if len(hr_data) > 0 else None,
            'HR_std': np.std(hr_data) if len(hr_data) > 0 else None
        }


def analyze_hrv_and_trends(df_result):
    """Анализ вариабельности сердечного ритма и тенденций"""
    if 'HR' not in df_result.columns:
        return

    st.write("### Анализ вариабельности сердечного ритма (HRV)")

    # Расчет метрик HRV для каждого уровня стресса
    hrv_results = []
    for level in df_result['Stress Level'].unique():
        level_data = df_result[df_result['Stress Level'] == level]
        if len(level_data) > 0:
            hrv_metrics = calculate_hrv_metrics(level_data['HR'].values)
            hrv_metrics['Stress Level'] = level
            hrv_results.append(hrv_metrics)

    if hrv_results:
        hrv_df = pd.DataFrame(hrv_results)
        st.dataframe(hrv_df)

        # Визуализация метрик HRV
        fig = px.bar(
            hrv_df,
            x='Stress Level',
            y=['SDNN', 'RMSSD'],
            barmode='group',
            title="Метрики HRV по уровням стресса",
            color_discrete_sequence=['#2E86C1', '#F39C12']
        )
        st.plotly_chart(fig, use_container_width=True)


def analyze_correlations(df_result):
    """Анализ корреляций между параметрами и уровнем стресса"""
    st.subheader("Корреляционный анализ")

    # Выбор параметров для анализа
    available_features = [f for f in FEATURES if f in df_result.columns]
    corr_features = st.multiselect(
        "Выберите параметры для корреляционного анализа",
        options=available_features,
        default=available_features
    )

    if corr_features:
        # Матрица корреляций
        corr_matrix = df_result[corr_features].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title="Матрица корреляций"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Корреляция с уровнем стресса
        if len(df_result['Stress Level'].unique()) > 1:
            st.write("### Корреляция с уровнем стресса")
            # Преобразуем уровни стресса в числовые значения
            stress_numeric = pd.factorize(df_result['Stress Level'])[0]

            # Рассчитываем корреляцию для каждого признака
            stress_corr = {}
            for feature in corr_features:
                r, p = pearsonr(df_result[feature], stress_numeric)
                stress_corr[feature] = (r, p)

            # Создаем DataFrame для отображения результатов
            stress_corr_df = pd.DataFrame({
                'Параметр': list(stress_corr.keys()),
                'Корреляция (r)': [stress_corr[f][0] for f in stress_corr],
                'p-значение': [stress_corr[f][1] for f in stress_corr],
                'Статистически значимо': [stress_corr[f][1] < 0.05 for f in stress_corr]
            }).sort_values('Корреляция (r)', ascending=False)

            st.dataframe(stress_corr_df)


def export_data(df_result, format="CSV"):
    """Экспорт данных в выбранном формате"""
    if format == "CSV":
        csv = df_result.to_csv(index=False, encoding='utf-8-sig')
        b64 = base64.b64encode(csv.encode('utf-8-sig')).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="stress_analysis_results.csv">Скачать CSV</a>'
    elif format == "Excel":
        excel_file = io.BytesIO()
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            df_result.to_excel(writer, index=False, sheet_name='Результаты')
        excel_file.seek(0)
        b64 = base64.b64encode(excel_file.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="stress_analysis_results.xlsx">Скачать Excel</a>'
    elif format == "JSON":
        json_file = df_result.to_json(orient='records', force_ascii=False)
        b64 = base64.b64encode(json_file.encode('utf-8-sig')).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="stress_analysis_results.json">Скачать JSON</a>'

    st.markdown(href, unsafe_allow_html=True)


def analyze_gsr_stress_relationship(df_result):
    """Анализ зависимости уровня стресса от GSR показателей"""
    if 'foot GSR' not in df_result.columns or 'hand GSR' not in df_result.columns:
        st.warning("Отсутствуют данные GSR для анализа")
        return

    st.subheader("Зависимость уровня стресса от показателей GSR")

    # Фильтры данных
    st.write("### Настройки фильтров")

    # Выбор формата времени
    time_format = st.radio(
        "Формат времени на графиках",
        options=["Метки времени (timestamp)", "Время в минутах (округленное)"],
        index=0,
        key="gsr_time_format"
    )

    # Фильтр по уровням стресса
    selected_levels = st.multiselect(
        "Выберите уровни стресса",
        options=sorted(df_result['Stress Level'].unique()),
        default=sorted(df_result['Stress Level'].unique()),
        key="gsr_stress_levels"
    )

    # Фильтр по временному диапазону
    time_min, time_max = int(df_result['Timestamp'].min()), int(df_result['Timestamp'].max())
    time_range = st.slider(
        "Выберите диапазон времени",
        time_min,
        time_max,
        (time_min, time_max),
        key="gsr_time_range"
    )

    # Применение фильтров
    filtered_df = df_result[
        (df_result['Stress Level'].isin(selected_levels)) &
        (df_result['Timestamp'].between(time_range[0], time_range[1]))
        ]

    if filtered_df.empty:
        st.warning("Нет данных, соответствующих выбранным фильтрам.")
        return

    # Подготовка данных в зависимости от выбранного формата времени
    if time_format == "Время в минутах (округленное)":
        # Конвертируем время в минуты
        filtered_df = filtered_df.copy()
        filtered_df['Time_minutes_rounded'] = (
                (filtered_df['Timestamp'] - filtered_df['Timestamp'].min()) / 60
        ).round().astype(int)

        # Агрегируем данные
        agg_dict = {
            'foot GSR': 'mean',
            'hand GSR': 'mean',
            'Stress Level': lambda x: x.mode()[0]
        }

        plot_df = filtered_df.groupby('Time_minutes_rounded').agg(agg_dict).reset_index()
        x_axis_column = 'Time_minutes_rounded'
        x_axis_title = "Время (минуты)"
    else:
        x_axis_column = 'Timestamp'
        x_axis_title = "Метки времени"
        plot_df = filtered_df

    # 1. Линейные графики для GSR стопы и руки
    st.write("#### Динамика GSR стопы и руки по времени")

    # График для foot GSR
    fig1 = px.line(
        plot_df,
        x=x_axis_column,
        y='foot GSR',
        color='Stress Level',
        color_discrete_map=STRESS_COLORS,
        title="Динамика GSR стопы по времени",
        labels={
            'foot GSR': 'GSR стопы',
            x_axis_column: x_axis_title,
            'Stress Level': 'Уровень стресса'
        }
    )
    fig1.update_traces(mode='lines+markers')
    st.plotly_chart(fig1, use_container_width=True)

    # График для hand GSR
    fig2 = px.line(
        plot_df,
        x=x_axis_column,
        y='hand GSR',
        color='Stress Level',
        color_discrete_map=STRESS_COLORS,
        title="Динамика GSR руки по времени",
        labels={
            'hand GSR': 'GSR руки',
            x_axis_column: x_axis_title,
            'Stress Level': 'Уровень стресса'
        }
    )
    fig2.update_traces(mode='lines+markers')
    st.plotly_chart(fig2, use_container_width=True)

    # 2. Боксплоты для распределения GSR по уровням стресса
    st.write("#### Распределение значений GSR по уровням стресса")

    # Боксплот для foot GSR
    fig3 = px.box(
        plot_df,
        x='Stress Level',
        y='foot GSR',
        color='Stress Level',
        color_discrete_map=STRESS_COLORS,
        title="Распределение GSR стопы по уровням стресса"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Боксплот для hand GSR
    fig4 = px.box(
        plot_df,
        x='Stress Level',
        y='hand GSR',
        color='Stress Level',
        color_discrete_map=STRESS_COLORS,
        title="Распределение GSR руки по уровням стресса"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # 3. Гистограммы плотности распределения для различных уровней стресса
    st.write("#### Плотность распределения GSR по уровням стресса")

    # Гистограмма плотности для foot GSR
    fig5 = px.histogram(
        plot_df,
        x='foot GSR',
        color='Stress Level',
        marginal='rug',
        opacity=0.7,
        barmode='overlay',
        color_discrete_map=STRESS_COLORS,
        title="Плотность распределения GSR стопы"
    )
    fig5.update_layout(bargap=0.1)
    st.plotly_chart(fig5, use_container_width=True)

    # Гистограмма плотности для hand GSR
    fig6 = px.histogram(
        plot_df,
        x='hand GSR',
        color='Stress Level',
        marginal='rug',
        opacity=0.7,
        barmode='overlay',
        color_discrete_map=STRESS_COLORS,
        title="Плотность распределения GSR руки"
    )
    fig6.update_layout(bargap=0.1)
    st.plotly_chart(fig6, use_container_width=True)


def plot_signals_and_stress(plot_df, x_axis_column, x_axis_title):
    """Построение графиков биосигналов и уровня стресса"""
    # Гистограмма распределения уровней стресса
    fig = px.histogram(
        plot_df,
        x='Stress Level',
        color='Stress Level',
        color_discrete_map=STRESS_COLORS,
        title="Распределение уровней стресса"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Графики биосигналов
    st.write("### Графики биосигналов")
    for feature in [f for f in FEATURES if f in plot_df.columns]:
        fig = px.line(
            plot_df,
            x=x_axis_column,
            y=feature,
            title=f"{feature} по времени",
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_xaxes(title_text=x_axis_title)
        st.plotly_chart(fig, use_container_width=True)

    # График уровней стресса
    st.write("### Уровень стресса по времени")

    # Преобразуем уровни стресса в числовой формат и интерполируем
    stress_numeric = pd.factorize(plot_df['Stress Level'])[0]
    plot_df['Stress Level Numeric'] = stress_numeric
    plot_df['Stress Level Interpolated'] = pd.Series(stress_numeric).interpolate(method='linear').fillna(
        method='bfill').fillna(method='ffill')

    # Создаем цветовую карту для отображения на графике
    plot_df['Color'] = plot_df['Stress Level'].map(STRESS_COLORS)

    # Строим график уровня стресса
    fig = px.line(
        plot_df,
        x=x_axis_column,
        y='Stress Level Numeric',
        title="Уровень стресса по времени",
        labels={'Stress Level Numeric': 'Уровень стресса'},
        color_discrete_sequence=['#1f77b4']
    )

    # Настраиваем внешний вид графика
    fig.update_traces(line=dict(width=3))
    fig.update_yaxes(
        range=[-0.5, 4.5],
        tickvals=[0, 1, 2, 3, 4],
        ticktext=['M-I', 'M-IIa', 'M-IIb', 'M-III', 'M-IV']
    )
    fig.update_xaxes(title_text=x_axis_title)

    st.plotly_chart(fig, use_container_width=True)


def main_interface(df_result):
    """Основной интерфейс с вкладками"""
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Графики и фильтры",
        "Статистический анализ",
        "Корреляционный анализ",
        "GSR и стресс",
        "Экспорт данных"
    ])

    with tab1:
        # Выбор формата времени
        time_format = st.radio(
            "Формат времени на графиках",
            options=["Метки времени (timestamp)", "Время в минутах (округленное)"],
            index=0
        )

        # Фильтры данных
        selected_levels = st.multiselect(
            "Выберите уровни стресса",
            options=sorted(df_result['Stress Level'].unique()),
            default=sorted(df_result['Stress Level'].unique())
        )

        time_min, time_max = int(df_result['Timestamp'].min()), int(df_result['Timestamp'].max())
        time_range = st.slider("Выберите диапазон времени", time_min, time_max, (time_min, time_max))

        # Применение фильтров
        filtered_df = df_result[
            (df_result['Stress Level'].isin(selected_levels)) &
            (df_result['Timestamp'].between(time_range[0], time_range[1]))
            ]

        if not filtered_df.empty:
            # Подготовка данных для отображения
            if time_format == "Время в минутах (округленное)":
                # Конвертируем время в минуты
                filtered_df = filtered_df.copy()
                filtered_df['Time_minutes_rounded'] = (
                        (filtered_df['Timestamp'] - filtered_df['Timestamp'].min()) / 60
                ).round().astype(int)

                # Агрегируем данные
                agg_dict = {f: 'mean' for f in FEATURES if f in filtered_df.columns}
                agg_dict['Stress Level'] = lambda x: x.mode()[0]

                aggregated_df = filtered_df.groupby('Time_minutes_rounded').agg(agg_dict).reset_index()

                x_axis_column = 'Time_minutes_rounded'
                x_axis_title = "Время (минуты)"
                plot_df = aggregated_df
            else:
                x_axis_column = 'Timestamp'
                x_axis_title = "Метки времени"
                plot_df = filtered_df

            # Построение графиков
            plot_signals_and_stress(plot_df, x_axis_column, x_axis_title)

    with tab2:
        analyze_hrv_and_trends(df_result)

    with tab3:
        analyze_correlations(df_result)

    with tab4:
        analyze_gsr_stress_relationship(df_result)

    with tab5:
        st.subheader("Экспорт данных")
        export_format = st.selectbox(
            "Выберите формат экспорта",
            options=["CSV", "Excel", "JSON"]
        )
        export_data(df_result, format=export_format)


# Основной интерфейс приложения
def main():
    st.title("Расширенный анализ уровня стресса")

    # Боковая панель для загрузки и настроек
    with st.sidebar:
        st.header("Настройки")

        # Загрузка файла
        uploaded_file = st.file_uploader("Загрузите CSV или DAT файл", type=["csv", "dat"])

        # Настройки для колонок
        column_mapping = {}
        if uploaded_file:
            try:
                preview = pd.read_csv(uploaded_file, nrows=5, encoding='utf-8-sig')
                uploaded_file.seek(0)
                available_columns = preview.columns.tolist()

                st.subheader("Сопоставление колонок")
                st.write("Сопоставьте колонки из файла с необходимыми полями:")

                for feature in FEATURES:
                    default_index = 0
                    if feature in available_columns:
                        default_index = available_columns.index(feature) + 1

                    column_mapping[feature] = st.selectbox(
                        f"Выберите колонку для {feature}",
                        options=[""] + available_columns,
                        index=default_index
                    )

                column_mapping = {k: v for k, v in column_mapping.items() if v}
            except Exception as e:
                st.error(f"Ошибка при чтении файла: {e}")

    # Обработка загруженного файла
    if uploaded_file or ('df_result' in st.session_state and st.session_state.df_result is not None):
        if uploaded_file:
            with st.spinner("Загрузка и обработка данных..."):
                timestamps, original_data, scaled_data = load_csv(uploaded_file, column_mapping)
                st.session_state.original_data = original_data
                st.session_state.timestamps = timestamps

                if scaled_data is not None:
                    # Преобразуем массив в кортеж для кеширования
                    scaled_data_tuple = tuple(map(tuple, scaled_data))
                    preds = predict_stress(scaled_data_tuple)

                    if preds is None:
                        st.error(f"Недостаточно данных. Требуется минимум {SEQ_LENGTH} записей.")
                    else:
                        # Создаем DataFrame с результатами
                        df_result = pd.DataFrame({
                            'Timestamp': timestamps[SEQ_LENGTH - 1:],
                            **{feat: original_data[feat].iloc[SEQ_LENGTH - 1:].values for feat in
                               original_data.columns},
                            'Stress Level': [LABELS[p] for p in preds]
                        })
                        st.session_state.df_result = df_result
                        st.success("Анализ завершен!")
                else:
                    st.warning(
                        "Невозможно выполнить анализ стресса из-за проблем с данными. Убедитесь, что все необходимые колонки присутствуют.")
    else:
        st.info("""
            ### Инструкции по использованию
            1. Загрузите CSV или DAT файл с данными биометрических сигналов через панель слева.
            2. Укажите соответствие колонок в вашем файле необходимым полям (HR, RESP, foot GSR, hand GSR).
            3. После загрузки будет доступен анализ данных с различными визуализациями и метриками.

            Формат данных: 
            - Файл должен иметь разделитель `,` (для CSV) или `\t` (для DAT).
            - Необходимые поля: данные сердечного ритма (HR), дыхания (RESP), КГР стопы (foot GSR), КГР руки (hand GSR).
        """)
    # Отображение результатов
    if 'df_result' in st.session_state and st.session_state.df_result is not None:
        main_interface(st.session_state.df_result)


if __name__ == "__main__":
    main()