import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="VARK - Sidang Demo", layout="wide")

LABEL_MAP = {
    "V": "Visual",
    "A": "Auditory",
    "R": "Reading/Writing",
    "K": "Kinesthetic",
}

def read_any(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".xlsx"):
        return pd.read_excel(file, engine="openpyxl")
    raise ValueError("Format file tidak didukung. Pakai CSV/XLSX.")

def rekomendasi_asesmen(pred):
    g = str(pred).strip().lower()
    if g in ["v", "visual"]:
        return "Berikan modul Video Pembelajaran & Infografis."
    if g in ["a", "auditory"]:
        return "Berikan diskusi terarah, presentasi lisan, dan forum tanya jawab."
    if g in ["r", "reading", "reading/writing", "read/write"]:
        return "Berikan rangkuman tertulis, review artikel, dan latihan soal berbasis teks."
    if g in ["k", "kinesthetic"]:
        return "Berikan project-based, praktikum/simulasi, dan studi kasus implementatif."
    return "Sesuaikan metode asesmen dengan hasil prediksi."

def score_category(avg_score):
    if avg_score is None or pd.isna(avg_score):
        return "-"
    s = float(avg_score)
    if s < 60:
        return "Rendah"
    if s < 80:
        return "Sedang"
    return "Tinggi"

@st.cache_resource
def load_artifacts():
    model = joblib.load("rf_final.pkl")
    feature_cols = json.load(open("feature_cols.json"))
    return model, feature_cols

model, feature_cols = load_artifacts()

# =========================
# HEADER SIDANG STYLE
# =========================
st.markdown("### --- SISTEM PREDIKSI & VALIDASI GAYA BELAJAR ---")
st.caption("Sistem akan mengambil data mahasiswa, menebak gayanya, lalu mencocokkannya dengan kunci jawaban asli (jika tersedia).")

# =========================
# UPLOAD DATASET
# =========================
st.subheader("1) Upload Dataset untuk Demo Sidang")
ds_file = st.file_uploader("Upload dataset (CSV/XLSX) yang sudah final dari Colab", type=["csv", "xlsx"])

if ds_file is None:
    st.info("Upload dulu file hasil export Colab: streamlit_dataset_vark.csv/xlsx")
    st.stop()

try:
    df = read_any(ds_file)
except Exception as e:
    st.error(f"Gagal membaca file: {e}")
    st.stop()

st.caption(f"Dataset terbaca: {df.shape[0]} baris × {df.shape[1]} kolom")

# =========================
# SET ID COLUMN + LABEL COLUMN
# =========================
id_candidates = [c for c in df.columns if str(c).lower() in ["id_student", "student_id", "id", "user_id", "userid", "id_index"]]
label_candidates = [c for c in df.columns if str(c).lower() in ["learning_style", "label", "target", "vark", "style"]]

col1, col2 = st.columns(2)
with col1:
    id_col = st.selectbox("Kolom ID (disarankan)", ["(pakai index baris)"] + id_candidates)
with col2:
    label_col = st.selectbox("Kolom Label Asli (untuk validasi)", ["(tidak ada)"] + label_candidates)

# =========================
# PILIH TARGET / RANDOM
# =========================
st.subheader("2) Target Mahasiswa")
cbtn1, cbtn2, cbtn3 = st.columns([1, 1, 2])

with cbtn1:
    btn_random = st.button("🎲 Simulasi Mahasiswa Acak")

with cbtn2:
    btn_run = st.button("🔍 Jalankan Prediksi")

with cbtn3:
    st.caption("Tip: pilih mahasiswa via ID atau index, lalu klik Jalankan Prediksi supaya outputnya keluar seperti contoh.")

if btn_random:
    ridx = np.random.randint(0, len(df))
    st.session_state["picked_index"] = int(ridx)

# kalau belum ada picked_index, set default 0
if "picked_index" not in st.session_state:
    st.session_state["picked_index"] = 0

# widget pilih mahasiswa
if id_col != "(pakai index baris)":
    ids = df[id_col].astype(str).tolist()
    chosen_id = st.selectbox("Pilih Mahasiswa (ID)", ids, index=min(st.session_state["picked_index"], len(ids)-1))
    row = df[df[id_col].astype(str) == str(chosen_id)].iloc[0]
    who_text = f"ID Index: {chosen_id}"
else:
    idx = st.number_input("Pilih index baris mahasiswa", min_value=0, max_value=max(0, len(df)-1),
                          value=int(st.session_state["picked_index"]), step=1)
    row = df.iloc[int(idx)]
    who_text = f"ID Index: {int(idx)}"

# =========================
# BUILD X_one (harus sesuai feature_cols)
# =========================
input_dict = {c: 0 for c in feature_cols}
for c in feature_cols:
    if c in row.index and not pd.isna(row[c]):
        input_dict[c] = row[c]

X_one = pd.DataFrame([input_dict]).reindex(columns=feature_cols, fill_value=0)

# =========================
# RUN PRED
# =========================
if not btn_run:
    st.stop()

pred = model.predict(X_one)[0]
pred_str = str(pred)
pred_full = LABEL_MAP.get(pred_str, pred_str)

proba_dict = None
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X_one)[0]
    proba_dict = dict(zip(model.classes_, proba))

# label asli (kalau ada)
actual = None
if label_col != "(tidak ada)" and label_col in row.index:
    actual = str(row[label_col]).strip()

# =========================
# OUTPUT SIDANG STYLE (seperti screenshot)
# =========================
st.markdown(f"#### 👤 DATA MAHASISWA DITEMUKAN ({who_text})")
st.markdown("##### 📊 PROFIL BELAJAR:")

days = row.get("total_study_days", None)
avg_score = row.get("avg_academic_score", None)

colp1, colp2, colp3 = st.columns(3)
with colp1:
    st.write(f"• **Konsistensi** : {int(days)} Hari Aktif" if days is not None and not pd.isna(days) else "• **Konsistensi** : -")
with colp2:
    st.write(f"• **Nilai** : {score_category(avg_score)}")
with colp3:
    st.write(f"• **Prediksi** : **{pred_str} ({pred_full})**")

# Aktivitas top-3 (dari fitur model, kecuali 2 fitur utama)
exclude = {"avg_academic_score", "total_study_days"}
click_feats = [c for c in feature_cols if c not in exclude]
vals = pd.to_numeric(pd.Series({c: input_dict.get(c, 0) for c in click_feats}), errors="coerce").fillna(0)
top = vals.sort_values(ascending=False).head(3)
top = [(k, int(v) if float(v).is_integer() else float(v)) for k, v in top.items() if v != 0]

st.write("• **Aktivitas** :")
if not top:
    st.write("  - (tidak ada aktivitas dominan / semua 0)")
else:
    for k, v in top:
        st.write(f"  - **{k.upper()}** ({v} klik)")

st.divider()

st.markdown("##### 📌 KUNCI JAWABAN / VALIDASI")
if actual is None:
    st.write("📌 **KUNCI JAWABAN (Label Asli)** : *(tidak tersedia di dataset)*")
    st.write(f"🤖 **TEBAKAN AI (Model Predict)** : **{pred_str} ({pred_full})**")
    st.info("✅ **STATUS VALIDASI** : *(tidak bisa dihitung tanpa label asli)*")
else:
    st.write(f"📌 **KUNCI JAWABAN (Label Asli)** : **{actual}**")
    st.write(f"🤖 **TEBAKAN AI (Model Predict)** : **{pred_str} ({pred_full})**")

    ok = (actual.lower() == pred_str.lower()) or (actual.lower() == pred_full.lower())
    if ok:
        st.success("✅ STATUS VALIDASI : VALID (Model menebak dengan tepat!)")
    else:
        st.error("❌ STATUS VALIDASI : TIDAK VALID (Belum tepat)")

st.markdown("##### 🎯 Rekomendasi AI")
st.write(rekomendasi_asesmen(pred_str))

if proba_dict is not None:
    st.markdown("##### 📈 Probabilitas")
    proba_df = pd.DataFrame({
        "Kelas": list(proba_dict.keys()),
        "Probabilitas": [float(v) for v in proba_dict.values()]
    }).sort_values("Probabilitas", ascending=False)
    st.dataframe(proba_df, use_container_width=True, hide_index=True)

st.divider()

st.markdown("##### 🔎 SHAP Explainable AI (Why)")
try:
    import shap
    import matplotlib.pyplot as plt

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_one)  # gaya yang kamu pakai di colab

    # pilih class idx dari pred
    class_idx = list(model.classes_).index(pred)

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0, :, class_idx], max_display=10, show=False)
    st.pyplot(fig, clear_figure=True)

except Exception as e:
    st.warning(
        "SHAP tidak bisa ditampilkan. Pastikan sudah install:\n"
        "- pip install shap matplotlib\n\n"
        f"Detail error: {e}"
    )

with st.expander("🔧 Debug: Input yang masuk ke model"):
    st.dataframe(X_one, use_container_width=True, hide_index=True)
