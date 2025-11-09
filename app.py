# Nama file: app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import librosa
import tsfel # <-- Diperlukan untuk model perintah
import pydub
import io
import os
from streamlit_mic_recorder import mic_recorder

st.set_page_config(page_title="Klasifikasi Suara", layout="wide")
st.title("🎤 Aplikasi Klasifikasi Suara v2.1 (Dual-Feature)")
st.markdown("Model Perintah (Buka/Tutup) menggunakan TSFEL. Model Speaker (Abdi/Alex) menggunakan MFCC.")

# -------------------------------------------------------------------
# Muat Model dan Artefak
# -------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    artifacts = {}
    try:
        # Muat artefak Perintah
        artifacts['model_perintah'] = joblib.load("model_perintah.pkl")
        artifacts['features_perintah'] = joblib.load("features_perintah.pkl")
        artifacts['encoder_perintah'] = joblib.load("encoder_perintah.pkl")
        
        # Muat artefak Speaker
        artifacts['model_speaker'] = joblib.load("model_speaker.pkl")
        artifacts['features_speaker'] = joblib.load("features_speaker.pkl")
        artifacts['encoder_speaker'] = joblib.load("encoder_speaker.pkl")
        
        # Muat TSFEL Cfg (hanya perlu satu)
        artifacts['tsfel_cfg'] = tsfel.get_features_by_domain()
        
    except FileNotFoundError as e:
        st.error(f"Error: File artefak tidak ditemukan. {e}")
        st.error("Pastikan semua 6 file .pkl (perintah_* dan speaker_*) ada di folder ini.")
        return None
    except Exception as e:
        st.error(f"Error memuat artefak: {e}")
        return None

    return artifacts

ARTIFACTS = load_artifacts()

# -------------------------------------------------------------------
# Fungsi Ekstraksi
# -------------------------------------------------------------------

def get_audio_data(audio_bytes):
    """Helper untuk mengonversi bytes ke numpy array"""
    try:
        audio_segment = pydub.AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio_segment = audio_segment.set_channels(1) 
        audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        sr = audio_segment.frame_rate
        audio_data = librosa.util.normalize(audio_data)
        return audio_data, sr
    except Exception as e:
        st.error(f"Error saat memproses audio: {e}")
        return None, None

def extract_features_tsfel_live(audio_data, sr, tsfel_cfg, feature_list):
    """Ekstrak fitur TSFEL untuk Model Perintah"""
    try:
        features_df = tsfel.time_series_features_extractor(tsfel_cfg, audio_data, fs=sr, verbose=0)
        features_df.fillna(0, inplace=True)
        
        # Logika seleksi fitur
        final_features_df = pd.DataFrame(columns=feature_list)
        combined = pd.concat([final_features_df, features_df])
        final_features_df = combined[feature_list].fillna(0)
        return final_features_df.head(1)
        
    except Exception as e:
        st.error(f"Error TSFEL: {e}")
        return None

def extract_features_mfcc_live(audio_data, sr, feature_list, n_mfcc=20):
    """Ekstrak fitur MFCC untuk Model Speaker"""
    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        features = np.concatenate((mfccs_mean, mfccs_std))
        
        features_df = pd.DataFrame(features.reshape(1, -1))
        # Beri nama kolom SEMENTARA agar concat berfungsi
        feature_cols_mean = [f'mfcc_mean_{i+1}' for i in range(n_mfcc)]
        feature_cols_std = [f'mfcc_std_{i+1}' for i in range(n_mfcc)]
        features_df.columns = feature_cols_mean + feature_cols_std
        
        final_features_df = pd.DataFrame(columns=feature_list)
        combined = pd.concat([final_features_df, features_df])
        final_features_df = combined[feature_list].fillna(0)
        return final_features_df.head(1)
        
    except Exception as e:
        st.error(f"Error MFCC: {e}")
        return None

# -------------------------------------------------------------------
# Tata Letak UI Streamlit
# -------------------------------------------------------------------

if ARTIFACTS is None:
    st.stop()

def run_prediction(audio_bytes):
    # 1. Konversi audio (hanya sekali)
    audio_data, sr = get_audio_data(audio_bytes)
    if audio_data is None:
        return

    # 2. Prediksi Perintah (TSFEL)
    with st.spinner("Menganalisis perintah (TSFEL)..."):
        live_features_perintah = extract_features_tsfel_live(
            audio_data, sr,
            ARTIFACTS['tsfel_cfg'],
            ARTIFACTS['features_perintah']
        )
        if live_features_perintah is None: return

        model = ARTIFACTS['model_perintah']
        encoder = ARTIFACTS['encoder_perintah']
        pred_encoded = model.predict(live_features_perintah)
        pred_proba = model.predict_proba(live_features_perintah)
        label_perintah = encoder.inverse_transform(pred_encoded)[0]
        conf_perintah = np.max(pred_proba) * 100
        proba_perintah_raw = pred_proba

    # 3. Prediksi Speaker (MFCC)
    with st.spinner("Mengenali speaker (MFCC)..."):
        live_features_speaker = extract_features_mfcc_live(
            audio_data, sr,
            ARTIFACTS['features_speaker'],
            n_mfcc=20 # Pastikan ini SAMA dengan skrip 1b
        )
        if live_features_speaker is None: return

        model = ARTIFACTS['model_speaker']
        encoder = ARTIFACTS['encoder_speaker']
        pred_encoded = model.predict(live_features_speaker)
        pred_proba = model.predict_proba(live_features_speaker)
        label_speaker = encoder.inverse_transform(pred_encoded)[0]
        conf_speaker = np.max(pred_proba) * 100
        proba_speaker_raw = pred_proba

    # 4. Tampilkan Hasil
    st.success("**Prediksi Berhasil!**")
    st.metric(label="Speaker Terdeteksi", value=label_speaker.capitalize(), 
              help=f"Keyakinan: {conf_speaker:.2f}%")
    st.metric(label="Perintah Terdeteksi", value=label_perintah.capitalize(),
              help=f"Keyakinan: {conf_perintah:.2f}%")
    
    with st.expander("Lihat detail probabilitas"):
        st.write("Probabilitas Speaker:")
        st.dataframe(pd.DataFrame(proba_speaker_raw, columns=ARTIFACTS['encoder_speaker'].classes_))
        st.write("Probabilitas Perintah:")
        st.dataframe(pd.DataFrame(proba_perintah_raw, columns=ARTIFACTS['encoder_perintah'].classes_))


# -- Kolom UI (Tidak berubah) --
col1, col2 = st.columns(2)
with col1:
    st.subheader("Rekam Suara Anda")
    audio_data = mic_recorder(
        start_prompt="🎙️ Klik untuk merekam",
        stop_prompt="⏹️ Klik untuk berhenti",
        key='rekaman_suara',
        format='wav' 
    )
    if audio_data is not None:
        st.audio(audio_data['bytes'])
        if st.button("Prediksi Hasil Rekaman"):
            run_prediction(audio_data['bytes'])

with col2:
    st.subheader("Atau Upload File Manual")
    uploaded_file = st.file_uploader("Pilih file audio (.wav, .mp3, .ogg)", type=["wav", "mp3", "ogg"])
    if uploaded_file is not None:
        audio_bytes = uploaded_file.getvalue()
        st.audio(audio_bytes)
        if st.button("Prediksi File Upload"):
            run_prediction(audio_bytes)