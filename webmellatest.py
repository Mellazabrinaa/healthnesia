import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
import base64

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Healthnesia",
    page_icon="🧑‍⚕️",
    layout="wide"
)

# ================= BACKGROUND IMAGE =================

def get_base64_image(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()

img = get_base64_image("dokter.png")

# ================= SESSION =================

if "menu" not in st.session_state:
    st.session_state.menu = "Home"

if "page" not in st.session_state:
    st.session_state.page = "home"

# ================= CSS =================

st.markdown(f"""
<style>

[data-testid="stSidebar"] {{
background: linear-gradient(180deg,#1d4e89,#2563a6);
}}

.sidebar-title {{
font-size:28px;
font-weight:bold;
text-align:center;
color:white;
margin-bottom:20px;
}}

.title {{
text-align:center;
font-size:65px;
font-weight:900;
color:#1d4e89;
}}

.subtitle {{
text-align:center;
font-size:22px;
color:gray;
margin-bottom:30px;
}}

.hero {{
background:
linear-gradient(rgba(37,99,166,0.9),rgba(37,99,166,0.9)),
url("data:image/png;base64,{img}");
background-size:50% 100%,50% 100%;
background-position:left,right;
background-repeat:no-repeat;
padding:100px;
border-radius:25px;
color:white;
margin-bottom:40px;
}}

.hero-text {{
width:50%;
}}

.card {{
background:white;
padding:40px;
border-radius:20px;
text-align:center;
font-size:22px;
font-weight:bold;
box-shadow:0 10px 25px rgba(0,0,0,0.1);
}}

</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================

diabetes = pd.read_csv("diabetes.csv")
heart = pd.read_csv("heart_v2(in).csv")

# ================= MODEL DIABETES =================

features_diabetes = [
'Pregnancies','Insulin','BMI','Age','Glucose',
'BloodPressure','DiabetesPedigreeFunction'
]

X = diabetes[features_diabetes]
y = diabetes["Outcome"]

X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.3,random_state=1
)

rf_diabetes = RandomForestClassifier()
rf_diabetes.fit(X_train,y_train)

dt_diabetes = DecisionTreeClassifier()
dt_diabetes.fit(X_train,y_train)

rf_acc_diabetes = accuracy_score(y_test,rf_diabetes.predict(X_test))
dt_acc_diabetes = accuracy_score(y_test,dt_diabetes.predict(X_test))

# ================= MODEL JANTUNG =================

Xh = heart.drop("heart disease",axis=1)
yh = heart["heart disease"]

X_train_h,X_test_h,y_train_h,y_test_h = train_test_split(
Xh,yh,test_size=0.3,random_state=1
)

rf_heart = RandomForestClassifier()
rf_heart.fit(X_train_h,y_train_h)

dt_heart = DecisionTreeClassifier()
dt_heart.fit(X_train_h,y_train_h)

rf_acc_heart = accuracy_score(y_test_h,rf_heart.predict(X_test_h))
dt_acc_heart = accuracy_score(y_test_h,dt_heart.predict(X_test_h))

# ================= SIDEBAR =================

st.sidebar.markdown(
"<div class='sidebar-title'>🧑‍⚕️ Healthnesia</div>",
unsafe_allow_html=True
)

if st.sidebar.button("🏠 Home",use_container_width=True):
    st.session_state.menu="Home"

if st.sidebar.button("🩸 Prediksi Diabetes",use_container_width=True):
    st.session_state.menu="Prediksi Diabetes"

if st.sidebar.button("❤️ Prediksi Jantung",use_container_width=True):
    st.session_state.menu="Prediksi Jantung"

menu = st.session_state.menu

# ================= HOME =================

if menu=="Home":

    if st.session_state.page=="home":

        st.markdown("<div class='title'>Healthnesia</div>",unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>AI Medical Prediction System</div>",unsafe_allow_html=True)

        st.markdown("""
        <div class='hero'>
        <div class='hero-text'>
        <h1>Your Health Is Our Mission</h1>
        <p>Sistem AI untuk membantu mendeteksi risiko Diabetes dan Penyakit Jantung sejak dini.</p>
        </div>
        </div>
        """,unsafe_allow_html=True)

        col1,col2 = st.columns(2)

        with col1:
            st.markdown("<div class='card'>🩸 Informasi Diabetes</div>",unsafe_allow_html=True)

            if st.button("Buka Info Diabetes",use_container_width=True):
                st.session_state.page="diabetes_info"
                st.rerun()

        with col2:
            st.markdown("<div class='card'>❤️ Informasi Penyakit Jantung</div>",unsafe_allow_html=True)

            if st.button("Buka Info Jantung",use_container_width=True):
                st.session_state.page="heart_info"
                st.rerun()

    # ================= INFO DIABETES =================

    elif st.session_state.page=="diabetes_info":

        st.title("🩸 Informasi Penyakit Diabetes")

        st.write("""
Diabetes adalah penyakit kronis yang terjadi ketika tubuh tidak dapat
menggunakan insulin secara efektif sehingga kadar gula darah meningkat.

Gejala:
- Sering haus
- Sering buang air kecil
- Mudah lelah
- Penglihatan kabur

Faktor risiko:
- Obesitas
- Kurang olahraga
- Riwayat keluarga
- Pola makan tidak sehat
""")

        if st.button("⬅ Kembali"):
            st.session_state.page="home"
            st.rerun()

    # ================= INFO JANTUNG =================

    elif st.session_state.page=="heart_info":

        st.title("❤️ Informasi Penyakit Jantung")

        st.write("""
Penyakit jantung adalah gangguan pada jantung dan pembuluh darah yang
dapat menyebabkan serangan jantung atau gagal jantung.

Faktor risiko:
- Kolesterol tinggi
- Tekanan darah tinggi
- Merokok
- Kurang aktivitas fisik
""")

        if st.button("⬅ Kembali"):
            st.session_state.page="home"
            st.rerun()

# ================= PREDIKSI DIABETES =================

elif menu=="Prediksi Diabetes":

    st.title("🩸 Prediksi Diabetes")

    st.write("Random Forest Accuracy:",round(rf_acc_diabetes*100,2),"%")
    st.write("Decision Tree Accuracy:",round(dt_acc_diabetes*100,2),"%")

    col1,col2 = st.columns(2)

    with col1:
        preg = st.number_input("Pregnancies")
        bmi = st.number_input("BMI")
        glucose = st.number_input("Glucose")
        dpf = st.number_input("Diabetes Pedigree Function")

    with col2:
        insulin = st.number_input("Insulin")
        age = st.number_input("Age")
        bp = st.number_input("Blood Pressure")

    if st.button("Predict Diabetes"):

        with st.spinner("AI sedang menganalisis..."):
            time.sleep(2)

            data = pd.DataFrame(
                [[preg,insulin,bmi,age,glucose,bp,dpf]],
                columns=features_diabetes
            )

            pred_rf = rf_diabetes.predict(data)
            pred_dt = dt_diabetes.predict(data)

            prob = rf_diabetes.predict_proba(data)
            risk = prob[0][1]*100

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            title={'text':"Risk %"},
            gauge={'axis':{'range':[0,100]}}
        ))

        st.plotly_chart(fig,use_container_width=True)

        st.subheader("Random Forest")

        if pred_rf[0]==1:
            st.error("⚠ Terindikasi Diabetes")
        else:
            st.success("✅ Tidak Ada Indikasi")

        st.subheader("Decision Tree")

        if pred_dt[0]==1:
            st.error("⚠ Terindikasi Diabetes")
        else:
            st.success("✅ Tidak Ada Indikasi")

# ================= PREDIKSI JANTUNG =================

elif menu=="Prediksi Jantung":

    st.title("❤️ Prediksi Penyakit Jantung")

    st.write("Random Forest Accuracy:",round(rf_acc_heart*100,2),"%")
    st.write("Decision Tree Accuracy:",round(dt_acc_heart*100,2),"%")

    col1,col2 = st.columns(2)

    with col1:
        age = st.number_input("Age")
        bp = st.number_input("Blood Pressure")

    with col2:
        gender = st.selectbox("Gender",["Female","Male"])
        chol = st.number_input("Cholesterol")

    sex = 1 if gender=="Male" else 0

    if st.button("Predict Heart Disease"):

        with st.spinner("AI sedang menganalisis..."):
            time.sleep(2)

            data = pd.DataFrame(
                [[age,sex,bp,chol]],
                columns=Xh.columns
            )

            pred_rf = rf_heart.predict(data)
            pred_dt = dt_heart.predict(data)

            prob = rf_heart.predict_proba(data)
            risk = prob[0][1]*100

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            title={'text':"Risk %"},
            gauge={'axis':{'range':[0,100]}}
        ))

        st.plotly_chart(fig,use_container_width=True)

        st.subheader("Random Forest")

        if pred_rf[0]==1:
            st.error("⚠ Terindikasi Penyakit Jantung")
        else:
            st.success("✅ Tidak Ada Indikasi")

        st.subheader("Decision Tree")

        if pred_dt[0]==1:
            st.error("⚠ Terindikasi Penyakit Jantung")
        else:
            st.success("✅ Tidak Ada Indikasi")
