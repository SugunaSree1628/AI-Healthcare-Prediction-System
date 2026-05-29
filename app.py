import streamlit as st
import pandas as pd
import sqlite3
import re
from datetime import date
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
from streamlit_option_menu import option_menu

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI Healthcare Dashboard",
    layout="wide",
    page_icon="🩺"
)

# ---------------- BRIGHT UI STYLE ---------------- #
st.markdown("""
<style>
.stApp { background-color: #f8fafc; }
h1, h2, h3 { color: #0ea5e9; text-align: center; font-weight: 700; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
div.stButton > button { background: linear-gradient(to right, #00C2FF, #2DD4BF); color: white; font-weight: bold; border-radius: 10px; }
div.stButton > button:hover { transform: scale(1.03); }
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #
st.title("🩺 AI HEALTHCARE PREDICTION SYSTEM")
st.markdown("### Bright Medical Analytics Dashboard")

# ---------------- DATABASE ---------------- #
conn = sqlite3.connect("patients.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    dob TEXT,
    email TEXT,
    glucose REAL,
    haemoglobin REAL,
    cholesterol REAL,
    remarks TEXT
)
""")
conn.commit()

# ---------------- ML DATA & MODEL TRAINING ---------------- #
data = {
    "Glucose": [85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230],
    "Haemoglobin": [15,14,13,14,15,13,12,11,10,14,13,12,11,10,9,8,12,11,10,9,8,12,11,10,9,8,7,10,9,8],
    "Cholesterol": [160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,240,250,260,270,280,290,300,310,320,330,340,260,270,280],
    "Result": ["Normal","Normal","Normal","Normal","Normal","Normal","Normal","Diabetes Risk","Diabetes Risk","Diabetes Risk",
               "Diabetes Risk","Diabetes Risk","Heart Disease Risk","Heart Disease Risk","Anaemia Risk","Anaemia Risk",
               "Diabetes Risk","Diabetes Risk","Heart Disease Risk","Anaemia Risk","Anaemia Risk","Heart Disease Risk",
               "Heart Disease Risk","Heart Disease Risk","Heart Disease Risk","Heart Disease Risk","Anaemia Risk","Heart Disease Risk","Heart Disease Risk","Anaemia Risk"]
}

df_ml = pd.DataFrame(data)
X = df_ml[["Glucose","Haemoglobin","Cholesterol"]]
y = df_ml["Result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------- AI PREDICTION FUNCTION ---------------- #
def predict_health(glucose, haemoglobin, cholesterol):
    return model.predict([[glucose, haemoglobin, cholesterol]])[0]

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2785/2785482.png", width=100)
    menu = option_menu(
        "MENU",
        ["Add Patient", "View & Manage Patients", "Analytics"],
        icons=["person-plus", "table", "bar-chart"],
        default_index=0
    )

# ---------------- 1. CREATE (ADD PATIENT) ---------------- #
if menu == "Add Patient":
    st.subheader("➕ Add Patient Record")

    with st.form("add_form", clear_on_submit=True):
        name = st.text_input("Full Name")
        dob = st.date_input("Date of Birth", min_value=date(1900, 1, 1), max_value=date.today(), value=date(2001, 1, 1))
        email = st.text_input("Email Address")
        glucose = st.number_input("Glucose", min_value=1.0, value=90.0)
        haemoglobin = st.number_input("Haemoglobin", min_value=1.0, value=14.0)
        cholesterol = st.number_input("Cholesterol", min_value=1.0, value=180.0)
        submit = st.form_submit_button("Save Record")

    if submit:
        if not name.strip() or not email.strip():
            st.error("❌ Name and Email fields cannot be empty!")
        elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            st.error("❌ Invalid Email Format!")
        else:
            # 🤖 Automated AI Prediction
            prediction = predict_health(glucose, haemoglobin, cholesterol)
            cursor.execute("""
                INSERT INTO patients (name, dob, email, glucose, haemoglobin, cholesterol, remarks)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, str(dob), email, glucose, haemoglobin, cholesterol, prediction))
            conn.commit()
            
            if prediction == "Normal":
                st.success(f"🟢 Saved Successfully! AI Remarks: {prediction}")
            elif prediction == "Diabetes Risk":
                st.warning(f"🟡 Saved Successfully! AI Remarks: {prediction}")
            else:
                st.error(f"🔴 Saved Successfully! AI Remarks: {prediction}")

# ---------------- 2. READ, UPDATE & DELETE ---------------- #
elif menu == "View & Manage Patients":
    st.subheader("📋 Patient Records Database")

    df = pd.read_sql("SELECT * FROM patients", conn)
    
    if df.empty:
        st.info("No records found. Please add patient details.")
    else:
        # READ
        st.dataframe(df, use_container_width=True)
        st.markdown("---")
        
        # UPDATE SECTION (With Expander for Clean UI)
        with st.expander("✏️ Update Patient Details"):
            col1, col2 = st.columns(2)
            with col1:
                uid = st.number_input("Enter Patient ID to Update", min_value=1, step=1)
                fetch_btn = st.button("Fetch Patient Data")
                
            if fetch_btn or 'edit_uid' in st.session_state:
                if fetch_btn:
                    st.session_state['edit_uid'] = uid
                
                patient = cursor.execute("SELECT * FROM patients WHERE id=?", (st.session_state['edit_uid'],)).fetchone()
                
                if patient:
                    with st.form("update_form"):
                        u_name = st.text_input("New Name", value=patient[1])
                        u_dob = st.date_input("New DOB", value=date.fromisoformat(patient[2]))
                        u_email = st.text_input("New Email", value=patient[3])
                        u_glucose = st.number_input("New Glucose", min_value=1.0, value=float(patient[4]))
                        u_haemoglobin = st.number_input("New Haemoglobin", min_value=1.0, value=float(patient[5]))
                        u_cholesterol = st.number_input("New Cholesterol", min_value=1.0, value=float(patient[6]))
                        
                        update_submit = st.form_submit_button("Confirm Update")
                        
                    if update_submit:
                        if not u_name.strip() or not u_email.strip():
                            st.error("❌ Name and Email fields cannot be empty!")
                        elif not re.match(r"[^@]+@[^@]+\.[^@]+", u_email):
                            st.error("❌ Invalid Email Format!")
                        else:
                            # 🔥 FIX: Recalculating AI prediction automatically on Update!
                            new_prediction = predict_health(u_glucose, u_haemoglobin, u_cholesterol)
                            
                            cursor.execute("""
                                UPDATE patients
                                SET name=?, dob=?, email=?, glucose=?, haemoglobin=?, cholesterol=?, remarks=?
                                WHERE id=?
                            """, (u_name, str(u_dob), u_email, u_glucose, u_haemoglobin, u_cholesterol, new_prediction, st.session_state['edit_uid']))
                            conn.commit()
                            st.success("✅ Patient Details Updated with fresh AI Prediction!")
                            del st.session_state['edit_uid']
                            st.rerun()
                else:
                    st.error("❌ Patient ID not found!")

        # DELETE SECTION
        with st.expander("🗑️ Delete Patient Record"):
            del_id = st.number_input("Enter Patient ID to Delete", min_value=1, step=1)
            if st.button("Delete Permanently", type="secondary"):
                check_user = cursor.execute("SELECT * FROM patients WHERE id=?", (del_id,)).fetchone()
                if check_user:
                    cursor.execute("DELETE FROM patients WHERE id=?", (del_id,))
                    conn.commit()
                    st.warning(f"⚠️ Record ID {del_id} Deleted Successfully.")
                    st.rerun()
                else:
                    st.error("❌ Patient ID not found!")

# ---------------- 3. ANALYTICS ---------------- #
elif menu == "Analytics":
    st.subheader("📊 Bright Analytics Dashboard")
    df = pd.read_sql("SELECT * FROM patients", conn)

    if not df.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Patients", len(df))
        col2.metric("Risk Patients", len(df[df["remarks"] != "Normal"]))
        col3.metric("Normal Patients", len(df[df["remarks"] == "Normal"]))

        fig1 = px.pie(df, names="remarks", title="Health Distribution", hole=0.4,
                      color_discrete_sequence=["#00C2FF", "#FF4D6D", "#FFD60A", "#2DD4BF"])
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.scatter_3d(df, x="glucose", y="haemoglobin", z="cholesterol", color="remarks", title="3D Health Analysis")
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.bar(df, x="name", y="glucose", color="remarks", title="Glucose Levels")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No data available for analytics. Please add records first.")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("🩺 AI Healthcare System • Streamlit + ML + SQLite • Final Production Copy")
