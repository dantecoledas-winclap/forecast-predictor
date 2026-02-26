import streamlit as st
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
import joblib
from pathlib import Path
import io

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forecast · Predictor",
    page_icon="📊",
    layout="centered",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

.stApp {
    background-color: #0d0d0d;
    color: #e8e0d0;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #e8e0d0 !important;
}

.hero {
    border: 1px solid #2a2a2a;
    border-left: 4px solid #c8a96e;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    background: linear-gradient(135deg, #111111 0%, #161410 100%);
}

.hero h1 {
    font-size: 2.6rem;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
    line-height: 1.1;
}

.hero p {
    color: #888;
    font-size: 0.85rem;
    margin: 0;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.step-label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #c8a96e;
    margin-bottom: 0.5rem;
    font-weight: 500;
}

.result-box {
    background: #111;
    border: 1px solid #2a2a2a;
    border-top: 3px solid #c8a96e;
    padding: 1.5rem;
    margin-top: 1.5rem;
}

.result-box h3 {
    font-family: 'DM Serif Display', serif !important;
    margin: 0 0 0.25rem 0;
    font-size: 1.4rem;
}

.result-box p {
    color: #888;
    font-size: 0.75rem;
    margin: 0 0 1rem 0;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

div[data-testid="stFileUploader"] {
    background: #111;
    border: 1px dashed #2a2a2a;
    border-radius: 0;
    padding: 1rem;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #c8a96e;
}

.stButton > button {
    background: #c8a96e !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: #b8955a !important;
}

div[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: #c8a96e !important;
    border: 1px solid #c8a96e !important;
    border-radius: 0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    font-weight: 500 !important;
}

div[data-testid="stDownloadButton"] > button:hover {
    background: #c8a96e !important;
    color: #0d0d0d !important;
}

.stAlert {
    border-radius: 0 !important;
    border-left: 3px solid #c8a96e !important;
    background: #111 !important;
    color: #e8e0d0 !important;
}

.metric-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}

.metric-card {
    flex: 1;
    background: #0d0d0d;
    border: 1px solid #2a2a2a;
    padding: 1rem;
    text-align: center;
}

.metric-card .val {
    font-size: 1.6rem;
    font-family: 'DM Serif Display', serif;
    color: #c8a96e;
}

.metric-card .lbl {
    font-size: 0.65rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

hr {
    border-color: #1e1e1e !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Forecast<br><i>Predictor</i></h1>
    <p>Revenue inference engine · Powered by ML</p>
</div>
""", unsafe_allow_html=True)

# ─── Load static assets ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "modelo_final1.pkl")

@st.cache_data
def load_normalization():
    return pd.read_csv(ROOT / "normalization_stats.csv", index_col=0)

try:
    modelo = load_model()
    normalization_stats = load_normalization()
except Exception as e:
    st.error(f"Error cargando archivos del modelo: {e}")
    st.stop()

# ─── Upload section ────────────────────────────────────────────────────────────
st.markdown('<div class="step-label">① Subir archivo</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Seleccioná el archivo Forecast (.xlsx o .csv)",
    type=["xlsx", "csv"],
    label_visibility="collapsed"
)

# ─── Process ───────────────────────────────────────────────────────────────────
def run_inference(file):
    # Load
    if file.name.endswith(".csv"):
        base_raw = pd.read_csv(file, decimal='.')
    else:
        base_raw = pd.read_excel(file, decimal='.')

    base = base_raw.copy()

    # ── Preproceso ──────────────────────────────────────────────────────────
    base = base[base['task_hierarchy'] == 'Task']
    base = base[base['status'].str.lower() != 'closed']

    columnas_drop = [
        'archived', 'task_hierarchy', 'team_id', 'permission_level',
        '(aut) TEAMS', 'text_content', 'description', 'date_closed',
        'date_done', 'group_assignees', 'checklists', 'tags', 'parent',
        'priority', 'points', 'time_estimate', 'dependencies', 'linked_tasks',
        'locations', 'space_name', 'waiting_on', 'blocking',
        '[ST] CS 1st Delivery Time', '[D] Type of Package', '[ST] Medidas',
        '[D] LINK: Project Folder', '[ST] LINK: Assets',
        '[ST] LINK: BRKAWAY Collab', '[ST] LINK: Frame',
        'S3_timestamp', 'S3_hours', 'S4_timestamp', 'S4_hours',
        'S5_timestamp', 'S5_hours', 'S6_timestamp', 'S6_hours',
        'S7_timestamp', 'S7_hours', 'url', 'sharing', 'list_id',
        'project_id', 'folder_name', 'folder_id', 'project_name',
        '[D][ST] CLIENT', 'name', '[D][ST] CP Owner', 'creator',
        'watchers', 'assignees', 'orderindex', 'space_id', 'date_created',
        'S1_timestamp', 'S2_timestamp', 'S1_hours', 'S2_hours', 't',
        '[D][ST] GEO', 'PO Number'
    ]
    columnas_drop = [c for c in columnas_drop if c in base.columns]
    base = base.drop(columns=columnas_drop, axis=1)

    base['[D] Deal revenue (USD)'] = base['[D] Deal revenue (USD)'].fillna(0)

    # Client Type dummies
    base['[D] Client Type'] = base['[D] Client Type'].fillna('').astype(str).str.split(',')
    df_exploded = base.explode('[D] Client Type')
    dummies = pd.get_dummies(df_exploded['[D] Client Type'].str.strip())
    dummies = dummies.groupby(df_exploded.index).sum()
    base = base.drop(columns=['[D] Client Type']).join(dummies)

    # Status ordinal
    base['status'] = base['status'].replace({
        'incomes': 1, 'client landing / brief': 2, 'client landing/brief': 2,
        'armado cont plan': 3, 'storyboard prod': 3, 'cplan: feedback & adj': 4,
        'creators prod': 5, 'prod: feedback & adj': 7, 'post produccion': 6,
        'Closed': 8
    })

    # task_type dummies
    base['task_type'] = base['task_type'].replace({1: 'Task'})
    dummies = pd.get_dummies(base['task_type'], prefix='task_type').astype(int)
    base = base.drop(['task_type'], axis=1)
    base = pd.concat([base, dummies], axis=1)

    # list_name dummies
    dummies = pd.get_dummies(base['list_name'], prefix='list_name').astype(int)
    base = base.drop(['list_name'], axis=1)
    base = pd.concat([base, dummies], axis=1)

    # TYPE of ASSET dummies
    base['[D][ST1] TYPE of ASSET'] = base['[D][ST1] TYPE of ASSET'].fillna('').astype(str).str.split(',')
    df_exploded = base.explode('[D][ST1] TYPE of ASSET')
    dummies = pd.get_dummies(df_exploded['[D][ST1] TYPE of ASSET'].str.strip())
    dummies = dummies.groupby(df_exploded.index).sum()
    base = base.drop(columns=['[D][ST1] TYPE of ASSET']).join(dummies, lsuffix='_caller', rsuffix='_other')

    # CHANNEL dummies
    base['[D][ST1] CHANNEL'] = base['[D][ST1] CHANNEL'].fillna('').astype(str).str.split(',')
    df_exploded = base.explode('[D][ST1] CHANNEL')
    dummies = pd.get_dummies(df_exploded['[D][ST1] CHANNEL'].str.strip())
    dummies = dummies.groupby(df_exploded.index).sum()
    base = base.drop(columns=['[D][ST1] CHANNEL']).join(dummies, lsuffix='_caller', rsuffix='_other')

    # Set index
    base = base.set_index("task_id")

    # ── Date engineering (replicating original script lines 212-327) ─────────
    hoy = pd.Timestamp.today().normalize()
    for col in ['date_updated', 'due_date', 'start_date']:
        if col in base.columns:
            base[col] = pd.to_datetime(base[col], errors='coerce')
            base[col] = (base[col] - hoy).dt.days

    base['dias_restantes'] = (pd.to_datetime(base.get('due_date', pd.NaT), errors='coerce') - hoy).dt.days if 'due_date' in base.columns else 0

    # ── Normalize ────────────────────────────────────────────────────────────
    for col in base.columns:
        if col in normalization_stats.index:
            mean = normalization_stats.loc[col, 'mean']
            std = normalization_stats.loc[col, 'std']
            if std != 0:
                base[col] = (base[col] - mean) / std

    # ── Inference ────────────────────────────────────────────────────────────
    expected = modelo.feature_names_in_
    # Add missing columns as 0
    for col in expected:
        if col not in base.columns:
            base[col] = 0
    base = base[expected]

    predicciones = modelo.predict(base)

    media = normalization_stats.loc['Cierre', 'mean']
    std = normalization_stats.loc['Cierre', 'std']
    y_pred_original = (predicciones * std) + media

    predicciones_df = pd.DataFrame(y_pred_original, columns=['Predicciones'], index=base.index)
    predicciones_df['Predicciones'] = predicciones_df['Predicciones'].apply(lambda x: max(0, x))

    # ── Merge con base original para ajustes ─────────────────────────────────
    base_orig = base_raw.copy().set_index('task_id')
    base_orig = base_orig.loc[predicciones_df.index]

    columnas_merge = ['[D] Deal revenue (USD)', 't-1', 't-2', 't-3', 't-4']
    df_pred = predicciones_df.merge(base_orig[columnas_merge], left_index=True, right_index=True)

    df_pred['suma_t'] = df_pred[['t-1', 't-2', 't-3', 't-4']].sum(axis=1)

    def ajustar_prediccion(row):
        diferencia = row['[D] Deal revenue (USD)'] - row['suma_t'] - row['Predicciones']
        if diferencia < 0:
            return max(0, row['[D] Deal revenue (USD)'] - row['suma_t'])
        return row['Predicciones']

    df_pred['predicciones_ajustadas'] = df_pred.apply(ajustar_prediccion, axis=1)
    df_final = df_pred[['predicciones_ajustadas']]

    # ── Reporte final (reporte_Agus) ─────────────────────────────────────────
    file.seek(0)
    if file.name.endswith(".csv"):
        df1 = pd.read_csv(file, decimal='.')
    else:
        df1 = pd.read_excel(file, decimal='.')

    hoy = pd.Timestamp.today().normalize()
    ini_mes = hoy.replace(day=1)
    fin_mes = ini_mes + MonthEnd(1)

    df1['S7_timestamp'] = pd.to_datetime(df1['S7_timestamp'], errors='coerce')
    mask_not_closed = df1['status'].str.lower() != 'closed'
    mask_closed_mes = (df1['status'].str.lower() == 'closed') & df1['S7_timestamp'].between(ini_mes, fin_mes)
    df1 = df1[mask_not_closed | mask_closed_mes]
    df1 = df1.set_index('task_id')

    datos_completos = df1.merge(df_final, left_index=True, right_index=True, how='outer')
    datos_completos['predicciones_ajustadas'] = datos_completos['predicciones_ajustadas'].fillna(0)

    datos_completos['Revenue t-n'] = datos_completos[['t-1', 't-2', 't-3', 't-4']].sum(axis=1)
    datos_completos['Revenue t+1'] = (
        datos_completos['[D] Deal revenue (USD)']
        - datos_completos['Revenue t-n']
        - datos_completos['predicciones_ajustadas']
    )

    mask_closed_final = datos_completos['status'].astype(str).str.lower() == 'closed'
    datos_completos.loc[mask_closed_final, 'predicciones_ajustadas'] = datos_completos.loc[mask_closed_final, 'Revenue t+1']
    datos_completos.loc[mask_closed_final, 'Revenue t+1'] = 0

    datos_completos = datos_completos[datos_completos['[D] Client Type'].isin(["TikTok (PS)", "Meta (PS)"])]

    cols_drop_final = [
        'task_type', 'text_content', 'description', 'CANCELED', 'orderindex',
        'date_created', 'date_updated', 'date_closed', 'date_done', 'archived',
        'creator', 'group_assignees', 'watchers', 'checklists', 'tags', 'parent',
        'task_hierarchy', 'priority', 'due_date', 'start_date', 'points',
        'time_estimate', 'dependencies', 'linked_tasks', 'locations', 'team_id',
        'url', 'sharing', 'permission_level', 'list_name', 'list_id',
        'project_name', 'project_id', 'folder_name', 'folder_id', 'space_name',
        'space_id', 'waiting_on', 'blocking', '[ST] CS 1st Delivery Time',
        '(aut) TEAMS', 'assignees', 'name', '[D][ST1] TYPE of ASSET',
        '[D][ST1] CHANNEL', '[ST] ADAPTATIONS', '[ST] Medidas',
        '[D][ST] CP Owner', '[D] LINK: Project Folder', '[ST] LINK: Assets',
        '[ST] LINK: BRKAWAY Collab', '[ST] LINK: Frame', 'S1_timestamp',
        'S1_hours', 'S2_timestamp', 'S2_hours', 'S3_timestamp', 'S3_hours',
        'S4_timestamp', 'S4_hours', 'S5_timestamp', 'S5_hours', 'S6_timestamp',
        'S6_hours', 'S7_timestamp', 'S7_hours', 'PO Number',
        'Partner Project ID', 'POC Client'
    ]
    cols_drop_final = [c for c in cols_drop_final if c in datos_completos.columns]
    informe = datos_completos.drop(columns=cols_drop_final)
    informe = informe.rename(columns={
        '[D][ST] CLIENT': 'Client',
        '[D][ST] GEO': 'Geo',
        '[ST] CREATION': 'Assets'
    })

    return informe, predicciones_df


# ─── Run button ────────────────────────────────────────────────────────────────
st.markdown('<div class="step-label" style="margin-top:1.5rem;">② Generar predicciones</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    if st.button("Correr modelo"):
        with st.spinner("Procesando..."):
            try:
                resultado, preds_raw = run_inference(uploaded_file)

                # Metrics
                n_tasks = len(resultado)
                total_pred = resultado['predicciones_ajustadas'].sum()
                total_revenue = resultado['[D] Deal revenue (USD)'].sum() if '[D] Deal revenue (USD)' in resultado.columns else 0

                st.markdown(f"""
                <div class="result-box">
                    <h3>Resultado</h3>
                    <p>Inferencia completada exitosamente</p>
                    <div class="metric-row">
                        <div class="metric-card">
                            <div class="val">{n_tasks}</div>
                            <div class="lbl">Tareas procesadas</div>
                        </div>
                        <div class="metric-card">
                            <div class="val">${total_pred:,.0f}</div>
                            <div class="lbl">Revenue predicho</div>
                        </div>
                        <div class="metric-card">
                            <div class="val">${total_revenue:,.0f}</div>
                            <div class="lbl">Deal revenue total</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Export
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    resultado.to_excel(writer, index=True, sheet_name='Predicciones')
                output.seek(0)

                st.markdown('<div class="step-label" style="margin-top:1.5rem;">③ Descargar reporte</div>', unsafe_allow_html=True)
                st.download_button(
                    label="Descargar reporte_Agus.xlsx",
                    data=output,
                    file_name="reporte_Agus.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"Error durante el procesamiento: {e}")
                st.exception(e)
else:
    st.markdown(
        '<p style="color:#444; font-size:0.8rem; margin-top:0.5rem;">Subí un archivo primero para habilitar esta opción.</p>',
        unsafe_allow_html=True
    )

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr style='margin-top:3rem'>", unsafe_allow_html=True)
st.markdown(
    '<p style="color:#333; font-size:0.7rem; text-align:center; letter-spacing:0.1em;">FORECAST PREDICTOR · ML INFERENCE ENGINE</p>',
    unsafe_allow_html=True
)
