import streamlit as st
import librosa
import numpy as np
import pandas as pd
import tsfel
import joblib
import soundfile as sf
import io
from streamlit_mic_recorder import mic_recorder

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Suara Buka/Tutup",
    page_icon="🎙️",
    layout="centered"
)

# --- Fungsi untuk Memuat Model (dengan cache) ---
@st.cache_resource
def load_artifacts():
    """Memuat artefak model yang sudah dilatih."""
    try:
        artifacts = joblib.load("model_artifacts.joblib")
        return artifacts
    except FileNotFoundError:
        st.error("File 'model_artifacts.joblib' tidak ditemukan. Jalankan 'train_and_save_model.py' terlebih dahulu.")
        return None
    except Exception as e:
        st.error(f"Gagal memuat artefak: {e}")
        return None

# --- Fungsi untuk Ekstraksi & Prediksi (Diperbarui) ---
def predict_audio(audio_bytes, artifacts):
    """Mengekstrak fitur dari audio bytes dan melakukan prediksi."""
    
    # 1. Konversi audio_bytes ke format yang bisa dibaca librosa
    try:
        audio_io = io.BytesIO(audio_bytes)
        y_audio, sr = sf.read(audio_io)
        
        if y_audio.ndim > 1:
            y_audio = y_audio[:, 0]
        
        target_sr = 22050 
        if sr != target_sr:
             y_audio = librosa.resample(y=y_audio, orig_sr=sr, target_sr=target_sr)
             sr = target_sr
             
    except Exception as e:
        st.error(f"Gagal memuat audio: {e}")
        return None, None, None, None # <-- Diperbarui

    # 2. Ekstraksi Fitur TSFEL
    try:
        signal_df = pd.DataFrame({'signal': y_audio})
        features_df_all = tsfel.time_series_features_extractor(
            artifacts['tsfel_cfg'], 
            signal_df, 
            fs=sr, 
            verbose=0
        )
    except Exception as e:
        st.error(f"Gagal mengekstrak fitur: {e}")
        return None, None, None, None # <-- Diperbarui
        
    # 3. Preprocessing (Sesuai Notebook)
    try:
        saved_feature_names = artifacts['feature_names']
        if features_df_all.shape[1] != len(saved_feature_names):
            st.error(f"Ekstraksi fitur menghasilkan {features_df_all.shape[1]} fitur, tapi model dilatih dengan {len(saved_feature_names)}.")
            return None, None, None, None # <-- Diperbarui
            
        features_df_all.columns = saved_feature_names 

        selected_names = artifacts['selected_feature_names']
        features_selected = features_df_all[selected_names]
        
        features_scaled = artifacts['scaler'].transform(features_selected)
        
        # --- PERUBAHAN UTAMA: Gunakan predict_proba ---
        # 4. Prediksi Probabilitas
        probabilities = artifacts['model'].predict_proba(features_scaled)[0]
        all_labels = artifacts['labels']
        
        # 5. Cari hasil terbaik
        best_index = np.argmax(probabilities)
        confidence = probabilities[best_index]
        prediction_label = all_labels[best_index]
        
        # 6. Pisahkan label
        speaker, command = prediction_label.split('_')
        
        # 7. Buat DataFrame probabilitas untuk ditampilkan
        prob_df = pd.DataFrame({
            'Kelas': all_labels,
            'Probabilitas': probabilities
        })
        prob_df = prob_df.sort_values(by='Probabilitas', ascending=False).reset_index(drop=True)
        
        # 8. Kembalikan semua data
        return speaker.capitalize(), command.capitalize(), confidence, prob_df

    except Exception as e:
        st.error(f"Gagal saat preprocessing/prediksi: {e}")
        return None, None, None, None # <-- Diperbarui

# --- Muat Model ---
artifacts = load_artifacts()

# --- UI Streamlit ---
st.title("🎙️ Aplikasi Prediksi Perintah Suara")
st.write("Aplikasi ini memprediksi perintah **'Buka'** atau **'Tutup'** dan siapa **speakernya** (misal: Abdi, Alex).")

if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None

if artifacts:
    tab1, tab2 = st.tabs(["📤 Upload File", "🔴 Rekam Suara"])

    # --- Tab 1: Upload File (Diperbarui) ---
    with tab1:
        st.header("Upload File Audio")
        uploaded_file = st.file_uploader(
            "Pilih file .wav, .mp3, atau .flac", 
            type=["wav", "mp3", "flac"]
        )
        
        if uploaded_file is not None:
            audio_bytes_upload = uploaded_file.getvalue()
            st.audio(audio_bytes_upload)
            
            if st.button("Prediksi File Upload", type="primary"):
                with st.spinner("Menganalisis audio..."):
                    # --- PERUBAHAN DI SINI ---
                    speaker, command, confidence, prob_df = predict_audio(audio_bytes_upload, artifacts)
                    
                    if speaker and command:
                        st.balloons()
                        st.success(f"**Hasil Prediksi:**")
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Speaker", speaker)
                        col2.metric("Perintah", command)
                        
                        # --- TAMBAHAN PROBABILITAS ---
                        st.metric("Tingkat Keyakinan", f"{confidence*100:.2f}%")
                        st.progress(float(confidence))
                        
                        st.subheader("Detail Probabilitas (Semua Kelas)")
                        st.dataframe(prob_df.style.format({'Probabilitas': '{:.2%}'}), use_container_width=True)

    # --- Tab 2: Rekam Suara (Diperbarui) ---
    with tab2:
        st.header("Rekam Suara Langsung")
        st.write("Klik tombol 'Start' untuk merekam (maks 5 detik).")
        
        audio_data = mic_recorder(
            start_prompt="Click to Record ⏺️",
            stop_prompt="Click to Stop ⏹️",
            key='recorder',
            format="wav",
            just_once=False, 
            use_container_width=True
        )
        
        if audio_data is not None:
            st.session_state.audio_bytes = audio_data['bytes']
        
        if st.session_state.audio_bytes is not None:
            st.audio(st.session_state.audio_bytes, format="audio/wav")
            
            if st.button("Prediksi Hasil Rekaman", type="primary"):
                with st.spinner("Menganalisis rekaman..."):
                    # --- PERUBAHAN DI SINI ---
                    speaker, command, confidence, prob_df = predict_audio(st.session_state.audio_bytes, artifacts)
                    
                    if speaker and command:
                        st.balloons()
                        st.success(f"**Hasil Prediksi:**")
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Speaker", speaker)
                        col2.metric("Perintah", command)
                        
                        # --- TAMBAHAN PROBABILITAS ---
                        st.metric("Tingkat Keyakinan", f"{confidence*100:.2f}%")
                        st.progress(float(confidence))
                        
                        st.subheader("Detail Probabilitas (Semua Kelas)")
                        st.dataframe(prob_df.style.format({'Probabilitas': '{:.2%}'}), use_container_width=True)
                    
                    st.session_state.audio_bytes = None 

else:
    st.warning("File 'model_artifacts.joblib' tidak ditemukan. Harap jalankan `train_and_save_model.py` terlebih dahulu.")