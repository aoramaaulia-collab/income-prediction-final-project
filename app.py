import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Income Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. LOAD RESOURCES (Model & Data)
# =====================================================
@st.cache_resource
def load_resources():
    # Load Model
    with open("model_income_xgboost.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Load Explainer
    explainer = shap.TreeExplainer(model)
    return model, explainer

@st.cache_data
def load_global_data():
    # Load data sample untuk Global SHAP
    try:
        df = pd.read_csv("X_sample.csv")
        return df
    except FileNotFoundError:
        return None

# Inisialisasi
try:
    model, explainer = load_resources()
except FileNotFoundError:
    st.error("❌ File 'model_income_xgboost.pkl' tidak ditemukan!")
    st.stop()

global_data = load_global_data()

# =====================================================
# 3. MAPPINGS (Data Dictionary)
# =====================================================
EDUCATION_MAP = {
    "Tidak Sekolah": 1, "SD Kelas 1–4": 2, "SD Kelas 5–6": 3,
    "SMP Kelas 7–8": 4, "SMP Kelas 9": 5, "SMA Kelas 10": 6,
    "SMA Kelas 11": 7, "SMA Kelas 12": 8, "Lulus SMA": 9,
    "Kuliah Tanpa Gelar": 10, "Diploma (D3)": 11, "Diploma (D4)": 12,
    "Sarjana (S1)": 13, "Magister (S2)": 14, "Pendidikan Profesi": 15,
    "Doktor (S3)": 16
}

WORKCLASS_MAP = {
    "Swasta": "Private", "Wiraswasta (Punya Usaha)": "Self-emp-inc",
    "Wiraswasta (Tanpa Usaha)": "Self-emp-not-inc", "Pemerintah Daerah": "Local-gov",
    "Pemerintah Provinsi": "State-gov", "Pemerintah Federal": "Federal-gov",
    "Tidak Pernah Bekerja": "Never-worked", "Tanpa Bayaran": "Without-pay",
    "Tidak Diketahui": "Unknown"
}

MARITAL_MAP = {
    "Belum Menikah": "Never-married", "Menikah (Tinggal Bersama)": "Married-civ-spouse",
    "Menikah (Pasangan Tidak Serumah)": "Married-spouse-absent", "Menikah (Pasangan Militer)": "Married-AF-spouse",
    "Berpisah": "Separated", "Cerai": "Divorced", "Janda / Duda": "Widowed"
}

OCCUPATION_MAP = {
    "Manajerial / Eksekutif": "Exec-managerial", "Profesional / Spesialis": "Prof-specialty",
    "Teknisi / IT Support": "Tech-support", "Penjualan": "Sales",
    "Operator Mesin": "Machine-op-inspct", "Perbaikan / Tukang": "Craft-repair",
    "Petani / Nelayan": "Farming-fishing", "Transportasi": "Transport-moving",
    "Keamanan": "Protective-serv", "Layanan Umum": "Other-service",
    "Asisten Rumah Tangga": "Priv-house-serv", "Pekerja Kasar": "Handlers-cleaners",
    "Militer": "Armed-Forces", "Tidak Bekerja": "No-occupation",
    "Tidak Diketahui": "Unknown"
}

RELATIONSHIP_MAP = {
    "Suami": "Husband", "Istri": "Wife", "Anak": "Own-child",
    "Bukan Keluarga Inti": "Not-in-family", "Kerabat Lain": "Other-relative",
    "Belum Menikah": "Unmarried"
}

RACE_LIST = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]

# =====================================================
# 4. SIDEBAR INPUT USER
# =====================================================
st.sidebar.title("Input Profil")
st.sidebar.caption("Masukkan data individu di sini:")

age = st.sidebar.number_input("Usia", 17, 90, 35)
hours = st.sidebar.number_input("Jam Kerja/Minggu", 1, 100, 40)
capital_gain = st.sidebar.number_input("Capital Gain ($)", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss ($)", 0, 100000, 0)
education_label = st.sidebar.selectbox("Pendidikan", list(EDUCATION_MAP.keys()))
marital_label = st.sidebar.selectbox("Status Pernikahan", list(MARITAL_MAP.keys()))
occupation_label = st.sidebar.selectbox("Pekerjaan", list(OCCUPATION_MAP.keys()))
relationship_label = st.sidebar.selectbox("Hubungan", list(RELATIONSHIP_MAP.keys()))
workclass_label = st.sidebar.selectbox("Kelas Pekerja", list(WORKCLASS_MAP.keys()))
gender = st.sidebar.radio("Jenis Kelamin", ["Male", "Female"], horizontal=True)
race = st.sidebar.selectbox("Ras", RACE_LIST)

# =====================================================
# 5. FUNGSI BUILD INPUT
# =====================================================
def build_input_data():
    feature_names = model.get_booster().feature_names
    X = pd.DataFrame(0, index=[0], columns=feature_names)

    # Data Numerik
    X["Age"] = age
    X["EducationNum"] = EDUCATION_MAP[education_label]
    X["Hours per Week"] = hours
    X["Capital Gain"] = capital_gain
    X["capital loss"] = capital_loss
    X["Has_Capital_Gain"] = 1 if capital_gain > 0 else 0
    X["Has_Capital_Loss"] = 1 if capital_loss > 0 else 0

    # Data Kategorikal (One-Hot Encoding Manual)
    categorical = {
        "Workclass": WORKCLASS_MAP[workclass_label],
        "Marital Status": MARITAL_MAP[marital_label],
        "Occupation": OCCUPATION_MAP[occupation_label],
        "Relationship": RELATIONSHIP_MAP[relationship_label],
        "Race": race,
        "Gender": "Male" if gender == "Male" else None
    }

    for prefix, value in categorical.items():
        if value:
            col_name = f"{prefix}_{value}"
            if col_name in X.columns:
                X[col_name] = 1
    return X

# =====================================================
# 6. MAIN CONTENT
# =====================================================
st.title("💰 Income Prediction & Interpretation")
st.markdown("---")

# PROSES PREDIKSI
X_input = build_input_data()
pred = model.predict(X_input)[0]
prob = model.predict_proba(X_input)[0][1]

# RINGKASAN PROFIL 
with st.container():
    st.subheader("📋 Ringkasan Profil")
    col_summary, _ = st.columns([3, 1]) 
    with col_summary:
        st.info(f"""
        **Demografi:** {gender}, {age} Tahun, Ras {race}.  
        **Pekerjaan:** {occupation_label} ({workclass_label}), {hours} Jam/Minggu.  
        **Status:** {marital_label} ({relationship_label}).  
        **Pendidikan:** {education_label}.  
        **Finansial:** Gain ${capital_gain} | Loss ${capital_loss}.
        """)

# TAMPILAN HASIL PREDIKSI
col_res1, col_res2 = st.columns([1, 2])

with col_res1:
    st.subheader("Hasil Prediksi")
    if pred == 1:
        st.success("### > $50K / Tahun")
        st.write("Individu diprediksi memiliki pendapatan tinggi.")
    else:
        st.warning("### ≤ $50K / Tahun")
        st.write("Individu diprediksi memiliki pendapatan rendah.")
    
    st.metric("Confidence (Probabilitas)", f"{prob*100:.2f}%")

with col_res2:
    st.caption("💡 **Info:** Prediksi ini dihasilkan oleh model XGBoost")

st.markdown("---")

# =====================================================
# 7. TAB ANALISIS (SHAP)
# =====================================================
tab_local, tab_global = st.tabs(["🔍 Analisis Individu (Local)", "🌍 Analisis Model (Global)"])

# --- TAB 1: LOCAL SHAP (Kenapa User INI diprediksi demikian?) ---
with tab_local:
    st.subheader("Mengapa user ini mendapatkan prediksi tersebut?")
    st.caption("Grafik ini menunjukkan fitur spesifik yang menaikkan (hijau) atau menurunkan (merah) peluang user memiliki pendapatan tinggi.")
    
    # Hitung Local SHAP
    shap_values_local = explainer.shap_values(X_input)[0]
    
    # Buat DataFrame untuk visualisasi
    shap_df = pd.DataFrame({
        "Fitur": X_input.columns,
        "Impact": shap_values_local
    }).sort_values("Impact", key=abs, ascending=True).tail(10) # Top 10 Fitur
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#F44336' if x < 0 else '#4CAF50' for x in shap_df['Impact']]
    ax.barh(shap_df['Fitur'], shap_df['Impact'], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Dampak terhadap Probabilitas Income >50K")
    ax.set_title("Top 10 Faktor Penentu untuk Individu Ini")
    st.pyplot(fig)

# --- TAB 2: GLOBAL SHAP (Apa pola umum model?) ---
with tab_global:
    st.subheader("Bagaimana model bekerja secara umum?")
    st.caption("Analisis ini didasarkan pada sampel data pengujian (X_test) untuk melihat pola keputusan model terhadap data baru.")
    
    if global_data is not None:
    
        with st.spinner("Menghitung pola global dari data sampel..."):
            shap_values_global = explainer.shap_values(global_data)
        st.markdown("**Grafik ini menunjukkan peringkat 10 fitur yang paling berpengaruh secara rata-rata.**")
    
        fig_bar, ax = plt.subplots(figsize=(6, 4))
        shap.summary_plot(shap_values_global, global_data, plot_type="bar", show=False, max_display=10)
        
        plt.xlabel("Rata-rata Besarnya Pengaruh (Mean |SHAP|)")     
        st.pyplot(fig_bar, use_container_width=False)
            
    else:
        st.error("⚠️ File 'X_sample.csv' tidak ditemukan.")
        st.warning("Pastikan Anda sudah mendownload file sample dari X_test dan meletakkannya di folder yang sama.")