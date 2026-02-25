# Importando librerías
import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
from catboost import Pool
import numpy as np

# 1. --- Configuración y carga ---
st.set_page_config(
    page_title='INTERCONNECT AI',
    layout='wide'
)

schema = []


@st.cache_resource
def load_assets():

    model = joblib.load(
        os.path.join('src', 'churn_model.joblib')
    )

    schema = joblib.load(
        os.path.join('src', 'columns_schema.joblib')
    )

    return model, schema


model, schema = load_assets()


# 2. --- Diseño e interfaz ---
st.title(
    body='SISTEMA DE PREDICCIÓN DE FUGA (CHURN)'
)

st.write(
    'Introduce los datos del cliente para evaluar su nivel de riesgo de fuga:'
)

# Creando 3 columnas (18 variables)
# Columna 1: Perfil del cliente
# Columna 2: Servicios
# Columna 3: Contrato y facturación

col1, col2, col3 = st.columns(spec=3)

# Desarrollando Columna 1: Perfil del cliente
with col1:
    st.subheader(
        body='👤 PERFIL DEL CLIENTE'
    )

    # Categóricas sencillas
    gender = st.selectbox(
        label='Género',
        options=['Female', 'Male']
    )

    senior = st.radio(
        label='¿Es adulto mayor?',
        options=[0, 1],
        help='Seleccione 1 si el cliente tiene más de 65'
    )

    partner = st.selectbox(
        label='¿Tiene pareja?',
        options=['Yes', 'No']
    )

    dependents = st.selectbox(
        label='¿Tiene dependientes?',
        options=['Yes', 'No']
    )

    # Para tenure aplicamos una lógica de conversión de Años meses y días --> días
    st.markdown(body='**Antigüedad del Cliente**')

    # Años
    y = st.number_input(
        label='Años',
        min_value=0,
        max_value=10,
        value=0
    )

    # Meses
    m = st.number_input(
        label='Meses',
        min_value=0,
        max_value=11,
        value=0
    )

    # Días
    d = st.number_input(
        label='Días',
        min_value=0,
        max_value=30,
        value=1
    )

    # Calculando Tenure a partir de Y-M-D
    tenure = (y*365) + (m*30) + d
    st.caption(
        body=f'El usuario ha estado activo {tenure} días'
    )

# Desarrollando Columna 2: Servicios
with col2:
    st.subheader(body="💻 SERVICIOS")

    # Describiendo categorías
    internet = st.selectbox(
        label='Servicio de Internet',
        options=['Fiber optic', 'DSL', 'No']
    )

    online_sec = st.selectbox(
        label='Seguridad Online',
        options=['Yes', 'No']
    )

    online_bak = st.selectbox(
        label='Respaldo Online',
        options=['Yes', 'No']
    )

    dev_prot = st.selectbox(
        label='Protección de dispositivos',
        options=['Yes', 'No']
    )

    tech_support = st.selectbox(
        label='Soporte Técnico',
        options=['Yes', 'No']
    )

    multi_lines = st.selectbox(
        label='Múltiples Líneas',
        options=['Yes', 'No']
    )

    tv = st.selectbox(
        label='Streaming Tv',
        options=['Yes', 'No']
    )

    movies = st.selectbox(
        label='Streaming Movies',
        options=['Yes', 'No']
    )

# Desarrollando Columna 3: Contrato
with col3:
    st.subheader(body="💳 CONTRATO Y FACTURACIÓN")

    # Describiendo tipo de contrato
    type_c = st.selectbox(
        label='Tipo de contrato',
        options=['Month-to-month', 'Two year', 'One year']
    )

    paperless = st.selectbox(
        label='Factura sin Papel',
        options=['Yes', 'No']
    )

    payment = st.selectbox(
        label='Método de Pago',
        options=['Electronic check', 'Mailed check',
                 'Bank transfer (automatic)', 'Credit card (automatic)']
    )

    monthly = st.number_input(
        label='Cargo Mensual ($)',
        min_value=1,
        max_value=300,
        value=15
    )

    total_bill = st.number_input(
        label='Cargos Totales Acumulados ($)',
        min_value=float(0),
        max_value=float(10000),
        value=float(monthly * (tenure / 30))
    )

# 3. --- Construcción del DataFrame ---

# Creando el diccionario con los datos
input_dict = {
    'Type': type_c,
    'PaperlessBilling': paperless,
    'PaymentMethod': payment,
    'MonthlyCharges': float(monthly),
    'TotalCharges': float(total_bill),
    'gender': gender,
    'SeniorCitizen': int(senior),
    'Partner': partner,
    'Dependents': dependents,
    'InternetService': internet,
    'OnlineSecurity': online_sec,
    'OnlineBackup': online_bak,
    'DeviceProtection': dev_prot,
    'TechSupport': tech_support,
    'StreamingTV': tv,
    'StreamingMovies': movies,
    'MultipleLines': multi_lines,
    'Tenure': int(tenure)
}

# Ordenando los datos según schema
df_input = pd.DataFrame(
    data=[input_dict]
)[schema]

# 4. --- Ejecución del modelo ---
st.divider()

if st.button(label='Analizar Riesgo', use_container_width=True):

    prob_array = model.predict_proba(X=df_input)
    prob = float(prob_array[0, 1])

    if prob >= 0.40:
        st.error(
            body=f'### ALTO RIESGO DE FUGA: {prob:.2%}'
        )
        st.progress(value=prob)

    else:
        st.success(
            body=f'### CLIENTE ESTABLE: {prob: 2%}'
        )
        st.progress(value=prob)

    # 4.1 --- Explicabilidad XAI ---

    st.markdown(body="---")
    st.subheader(body='🔍 Diagnóstico: ¿Por qué este resultado?')

    cat_cols_final = [
        'Type', 'PaperlessBilling', 'PaymentMethod', 'gender', 'Partner',
        'Dependents', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'MultipleLines'
    ]

    try:
        data_pool = Pool(
            data=df_input,
            cat_features=cat_cols_final
        )

        # 3. EL CAMBIO CLAVE: Usamos 'ShapValues' para XAI local
        local_imp = model.get_feature_importance(
            data=data_pool,
            type='ShapValues'
        )

        # 4. Procesamos los datos (ShapValues devuelve matriz [n_muestras, n_features + 1])
        # Tomamos la primera fila (0) y todos los valores menos el último ([:-1])
        val_impacto = local_imp[0][:-1]

        imp_df = pd.DataFrame(data={
            'Factor': schema,
            'Impacto': val_impacto
        })

        # Tomamos los 5 que más influyen usando el valor absoluto para rankear
        top_5_imp = imp_df.reindex(
            labels=imp_df['Impacto'].abs().sort_values(ascending=False).index
        ).head(n=5)

        # 5. Creamos el gráfico con Plotly
        fig_imp = px.bar(
            data_frame=top_5_imp,
            x='Impacto',
            y='Factor',
            orientation='h',
            color='Impacto',
            color_continuous_scale='RdYlGn_r',
            template='plotly_dark',
            title='FACTORES CLAVE EN ESTA PREDICCIÓN',
            labels={'Impacto': 'Contribución', 'Factor': 'Variable'}
        )

        fig_imp.update_layout(
            showlegend=False,
            height=400,
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        st.plotly_chart(figure_or_data=fig_imp, use_container_width=True)

        st.info(
            body="💡 **Valor para el Negocio:** Las barras rojas aumentan la probabilidad de fuga, "
                 "mientras que las verdes ayudan a retener al cliente."
        )

    except Exception as e:
        st.error(body=f"Error técnico en el diagnóstico: {e}")
