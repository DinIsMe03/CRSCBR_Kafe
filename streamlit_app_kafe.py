import streamlit as st
if "next_step" in st.session_state:
    st.session_state.step = st.session_state.next_step
    del st.session_state.next_step

import pandas as pd
import numpy as np
import os
import json
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pygsheets
import tempfile


# Routing antar halaman
st.set_page_config(page_title="Sistem Rekomendasi Kafe", layout="centered")

# Path data
@st.cache_data
def load_review_data():
    df = pd.read_excel("data/hasil_skor_dan_aspek.xlsx")
    df["tokens_negated_indo"] = df["tokens_negated_indo"].apply(eval)
    df["tokens_negated_english"] = df["tokens_negated_english"].apply(eval)
    return df

@st.cache_data
def load_kafe_vector():
    return pd.read_pickle("data/case_vector_df.pkl")

@st.cache_resource
def load_word2vec_model():
    return Word2Vec.load("data/word2vec_model.model")

# Panggil di awal
df_review = load_review_data()
df_kafe = load_kafe_vector()
model_w2v = load_word2vec_model()


# Ambil kolom vektor Word2Vec
vector_cols = [col for col in df_kafe.columns if col.startswith("dim_")]
X_matrix = df_kafe[vector_cols].values

# file_json_handler = "kodeRahasia_jangandiShare.json"


def buat_file_credential_sementara():
    # Ambil secret dari st.secrets
    json_key = dict(st.secrets["gcp_service_account"])

    # Simpan ke file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmp:
        json.dump(json_key, tmp)
        return tmp.name  # Kembalikan path-nya


cred_path = buat_file_credential_sementara()  # Buat file sementara

gc = pygsheets.authorize(service_file=cred_path)  # Otentikasi



# ========================
# KUMPULAN FUNSGI STEP
# ========================

def step_intro():

    st.title("☕ Sistem Rekomendasi Kafe")

    st.markdown("## 🧪 Uji Coba Aplikasi Rekomendasi Kafe")

    st.markdown("""
    Halo! 👋 Terima kasih sudah bersedia ikut uji coba kecil ini.

    Secara garis besar, kamu akan melewati beberapa tahapan berikut:

    1. Mengisi identitas diri  
    2. Mencoba **Aplikasi 1 (Query-Based)**  
    3. Mencoba **Aplikasi 2 (Conversational Recommender System - CRS-CBR)**  
    4. Mengisi **survey perbandingan** (terdiri dari 2 tahap)  
    5. Menyimak **kesimpulan dari hasil uji coba**

    Setiap langkah akan dijelaskan secara ringkas di masing-masing halaman.

    Klik tombol di bawah ini untuk memulai ⬇️
    """)

    if st.button("➡️ Mulai"):
        st.session_state.step = "identity"
        st.rerun()

def step_identity():
    st.subheader("🧍 Identitas Peserta Uji Coba")

    st.markdown("""
    Untuk memastikan hasil uji coba ini valid dan bisa dianalisis, silakan isi data identitas berikut.
    """)

    nama = st.text_input("👤 Nama Lengkap")
    usia = st.number_input("🍰 Usia", min_value=10, max_value=100, step=1)
    gender = st.radio("🚻 Jenis Kelamin", ["Laki-laki", "Perempuan"], horizontal=True)

    kategori_pengguna = st.radio(
        "☕ Seberapa sering kamu pergi ke kafe? (untuk mengelompokkan pengguna)",
        [
            "Casual - Jarang ke kafe atau hanya sesekali",
            "Frequent - Sering mengunjungi kafe dan cukup aktif"
        ]
    )

    email = st.text_input("📧 Email (opsional)", placeholder="Misalnya: kamu@gmail.com")

    if st.button("➡️ Lanjut ke Aplikasi 1"):
        if nama.strip():
            st.session_state.user_identity = {
                "nama": nama.strip(),
                "usia": usia,
                "gender": gender,
                "kategori_pengguna": kategori_pengguna.split(" - ")[0],
                "email": email.strip()
            }
            st.session_state.step = "intro_query"
            st.rerun()
        else:
            st.warning("⚠️ Nama tidak boleh kosong.")

# Halaman pengantar Aplikasi 1 - Query Based
def step_intro_query():
    st.subheader("🔍 Aplikasi 1: Sistem Query-Based")

    st.markdown("""
    Pada sistem ini, kamu cukup memasukkan **kata kunci** yang menggambarkan preferensi suasana atau fasilitas kafe yang kamu cari.

    Sistem akan mencari dan menampilkan **semua kafe** yang memiliki ulasan yang mengandung kata kunci tersebut, **tanpa pengolahan tambahan** seperti sentimen atau pembobotan.

    Cocok untuk kamu yang ingin mencari secara langsung dan eksplisit 🎯
    """)
    
    st.markdown("Apakah kamu ingin lanjut mencoba sistem ini?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Ya, lanjut ke Aplikasi 1"):
            st.session_state.step = "query_based"
            st.rerun()

    # with col2:
    #     if st.button("❌ Nanti aja, lanjut ke Aplikasi 2"):
    #         st.session_state.step = "intro_crs"
    #         st.rerun()

# from kategori_suasana_dict import kategori_suasana  # ⬅️ taruh di bagian import
from kategori_suasana_dict_updated import kategori_suasana

def step_query_based():
    from kategori_suasana_dict_updated import kategori_suasana
    from collections import defaultdict

    st.subheader("🔍 Aplikasi 1: Sistem Rekomendasi Query-Based")

    st.markdown("""
    Masukkan suasana, fasilitas, atau hal yang kamu cari di kafe sesuai keinginanmu.  
    Contoh: cozy, wifi, murah, rame, tenang, dll.

    ⚠️ Sistem akan mencari **kafe yang mengandung minimal satu kata dari tiap preferensi**.  
    💡 Cocok untuk kamu yang ingin mencari suasana tertentu berdasarkan opini pengguna asli.
    """)

    st.markdown("---")

    preferensi_dict = {}
    st.markdown("✅ Checklist sub-aspek dari suasana/fasilitas yang ingin kamu cari:")

    for kategori, sub_dict in kategori_suasana.items():
        with st.expander(kategori):
            for sub_label, keyword_list in sub_dict.items():
                if st.checkbox(f"{sub_label.title()}", key=f"{kategori}_{sub_label}"):
                    preferensi_dict[sub_label] = keyword_list

    st.markdown("---")

    if st.button("🔍 Cari Kafe"):
        if not preferensi_dict:
            st.warning("Masukkan minimal satu sub-aspek dari kategori yang tersedia.")
            return

        kafe_dengan_skor = []

        for nama_kafe, group in df_review.groupby("Nama Kafe"):
            tokens = []
            for _, row in group.iterrows():
                tokens.extend(row["tokens_negated_indo"] + row["tokens_negated_english"])

            subaspek_match_count = 0
            mention_dict = {}

            for sub_label, keywords in preferensi_dict.items():
                matched_keywords = [k for k in keywords if k in tokens]
                if matched_keywords:
                    subaspek_match_count += 1
                    for k in matched_keywords:
                        mention_dict[k] = tokens.count(k)

            total_mentions = sum(mention_dict.values())

            if subaspek_match_count > 0:
                kafe_dengan_skor.append((nama_kafe, mention_dict, subaspek_match_count, total_mentions))

        kafe_dengan_skor = sorted(kafe_dengan_skor, key=lambda x: (-x[2], -x[3]))[:10]

        if not kafe_dengan_skor:
            st.warning("😕 Tidak ditemukan kafe yang sesuai.")
        else:
            st.success("✅ Menampilkan 10 kafe teratas berdasarkan kecocokan:")
            for nama_kafe, mention_dict, subaspek_match, total in kafe_dengan_skor:
                cocok_str = []
                for sub_label, keywords in preferensi_dict.items():
                    count = sum(mention_dict.get(k, 0) for k in keywords)
                    if count > 0:
                        cocok_str.append(f"{sub_label} ({count}x)")
                cocok_str = ", ".join(cocok_str)
                st.markdown(f"### ⭐ {nama_kafe}")
                st.markdown(f"✅ Cocok karena disebut: {cocok_str}")
                st.markdown(f"📊 Total: {total} kali disebut di ulasan.")
                st.markdown("---")

            st.session_state.query_result = kafe_dengan_skor
            st.session_state.query_input = preferensi_dict
            st.session_state.query_has_run = True

    if st.session_state.get("query_has_run"):
        if st.button("➡️ Lanjut ke Aplikasi 2 (CRS)"):
            st.session_state.step = "intro_crs"
            st.rerun()

def step_intro_crs():
    st.subheader("🤖 Aplikasi 2: Conversational Case-Based Reasoning (CRS-CBR)")

    st.markdown("""
    Sistem ini akan membantu kamu menemukan kafe yang paling cocok dengan suasana yang kamu inginkan, bukan hanya berdasarkan kesamaan kata, tapi juga kesamaan makna.
    
    🧠 Bagaimana cara kerjanya?
    
    Sistem menggunakan teknologi yang bisa memahami arti kata-kata yang kamu pilih
    
    Hasil akan dihitung berdasarkan seberapa mirip maknanya dengan suasana di kafe
    
    Jika banyak ulasan negatif atau suasananya tidak nyaman, nilainya akan dikurangi
    
    💡 Apa yang bisa kamu lakukan?
    
    Pilih suasana atau kebutuhan kafe yang kamu cari (misalnya: tenang, ada wifi, cocok untuk kerja)
    
    Jika hasil belum sesuai, kamu bisa melakukan refinement, misalnya: “hindari yang berisik” atau “cari yang lebih sepi”
    
    ✨ Yuk, kita mulai dan temukan kafe yang paling cocok buat kamu!
    """)

    if st.button("➡️ Masukkan Preferensi"):
        st.session_state.step = "crs_cbr"
        st.rerun()

def step_crs_cbr():
    import os
    import json
    from kategori_suasana_dict_updated import kategori_suasana
    from collections import defaultdict

    st.subheader("🤖 Aplikasi 2: Conversational Recommender System (CRS)")

    st.markdown("""
    Masukkan preferensi suasana/fasilitas seperti halnya sebelumnya. Bedanya, kali ini sistem akan
    merekomendasikan berdasarkan **kemiripan makna** menggunakan Word2Vec.

    💡 Cocok untuk kamu yang ingin mendapatkan saran dari sistem secara pintar berdasarkan konteks.
    """)

    # Inisialisasi refine_logs jika belum ada
    if "refine_logs" not in st.session_state:
        st.session_state["refine_logs"] = []
    preferensi_dict = {}
    preferensi_label = {}

    st.markdown("✅ Checklist preferensi kamu:")

    for kategori, sub_dict in kategori_suasana.items():
        with st.expander(kategori):
            for sub_label, keyword_list in sub_dict.items():
                if st.checkbox(f"{sub_label.title()}", key=f"crs_{kategori}_{sub_label}"):
                    preferensi_dict[sub_label] = keyword_list
                    preferensi_label[kategori] = sub_label

    if not preferensi_dict:
        return

    all_keywords = [k for kws in preferensi_dict.values() for k in kws]

    # Cek apakah preferensi ini pernah disimpan user lain
    # def cari_case_sama(casebase, keywords, preferensi_label):
    #     for case in casebase:
    #         if sorted(case.get("crs_keywords", [])) == sorted(keywords) and case.get("preferensi_label", {}) == preferensi_label:
    #             return case
    #     return None



    # if os.path.exists("casebase.json"):
    #     with open("casebase.json", "r") as f:
    #         casebase = json.load(f)
    # else:
    #     casebase = []


    casebase = baca_casebase_dari_gsheet(
        spreadsheet_id="1RlsZ4h9FLSX_2J5wNuDn_fBQcVhSAnLe3A7eXqoB9HI", 
        sheet_name="Sheet2"
    )





    case_match = cari_case_sama(casebase, all_keywords, preferensi_label)
    force_crs_run = False  # Flag untuk memaksa proses rekomendasi

    if case_match:
        st.markdown("🔁 Preferensi kamu **pernah dicari oleh user sebelumnya.**")
        # st.markdown(f"📌 Kafe yang dipilih oleh user sebelumnya: **{case_match['selected_kafe']}**")

        lihat_lama = st.button("👀 Lihat Hasil User Sebelumnya")
        rekomendasi_baru = st.button("🧠 Cari Rekomendasi Baru")

        if lihat_lama:
            st.session_state.crs_keywords = all_keywords
            st.session_state.crs_preferensi_label = case_match["preferensi_label"]
            st.session_state.crs_refine_excluded = case_match.get("refine_excluded", [])

            query_vec = make_query_vector(all_keywords, model_w2v)
            similarity_scores = cosine_similarity(query_vec, X_matrix)[0]

            df_result = df_kafe.copy()
            df_result["Similarity"] = similarity_scores
            df_result["FinalScore"] = df_result["Similarity"]

            df_selected = df_result[df_result["Nama Kafe"] == case_match["selected_kafe"]]

            if not df_selected.empty:
                st.session_state.crs_result_before_refine = df_selected.to_dict(orient="records")
                st.session_state.kritik_dari_top5 = get_kritik_negatif(case_match["selected_kafe"], df_review, kata_kritik_umum, return_dict=True)
                st.session_state.crs_has_run = True
                st.rerun()
            else:
                st.warning("Kafe dari hasil user sebelumnya tidak ditemukan.")
                return

        elif rekomendasi_baru:
            force_crs_run = True  # paksa hitung ulang rekomendasi

    if not case_match:
        if st.button("🎯 Dapatkan Rekomendasi"):
            force_crs_run = True

    if force_crs_run:
        query_vec = make_query_vector(all_keywords, model_w2v)
        similarity_scores = cosine_similarity(query_vec, X_matrix)[0]

        df_result = df_kafe.copy()
        df_result["Similarity"] = similarity_scores
        df_result["FinalScore"] = df_result["Similarity"]

        df_sorted = df_result.sort_values(by="FinalScore", ascending=False).head(5)

        st.session_state.crs_keywords = all_keywords
        st.session_state.crs_result_before_refine = df_sorted.to_dict(orient="records")
        st.session_state.crs_has_run = True
        st.session_state.crs_preferensi_label = preferensi_label
        st.session_state.crs_refine_excluded = []
    # Log iterasi 0
    st.session_state.refine_logs = [{
        "iteration": 0,
        "keywords": all_keywords,
        "refine_added": [],
        "refine_excluded": [],
        "preferensi_label": preferensi_label
    }]

    if st.session_state.get("crs_has_run"):
        st.success("✨ TOP 5 REKOMENDASI KAFE:")

        for row in st.session_state.crs_result_before_refine:
            tampilkan_kafe_dengan_detail(
                row,
                st.session_state.crs_keywords,
                model_w2v,
                df_review,
                vector_cols=[col for col in df_kafe.columns if col.startswith("dim_")],
                kata_kritik_umum=kata_kritik_umum,
                preferensi_dict={
                    sub: kategori_suasana[kat][sub]
                    for kat, sub in st.session_state.crs_preferensi_label.items()
                }
            )

        kritik_counter = defaultdict(int)
        for row in st.session_state.crs_result_before_refine:
            nama_kafe = row["Nama Kafe"]
            kritik_dict = get_kritik_negatif(nama_kafe, df_review, kata_kritik_umum, return_dict=True)
            for k, v in kritik_dict.items():
                kritik_counter[k] += v

        st.session_state.kritik_dari_top5 = dict(kritik_counter)

        st.markdown("---")

        pilihan = st.radio(
            "Apakah kamu sudah puas dengan hasil rekomendasi ini?",
            options=["Sudah puas", "Belum puas, ingin refinement"],
            key="puas_crs"
        )

        if pilihan == "Sudah puas":
            top_kafe = [row["Nama Kafe"] for row in st.session_state.crs_result_before_refine]

            selected_kafe = st.radio(
                "📌 Pilih salah satu kafe yang mau kamu simpan ke Casebase:",
                top_kafe,
                key="crs_final_choice"
            )

            if st.button("💾 Simpan Pilihan"):
                case = {
                    "selected_kafe": selected_kafe,
                    "crs_keywords": st.session_state.crs_keywords,
                    "preferensi_label": st.session_state.crs_preferensi_label,
                    "refine_added": [],
                    "refine_excluded": st.session_state.get("crs_refine_excluded", []),
                    "compare_choice": "Langsung puas",
                    "user_identity": st.session_state.get("user_identity", {}),
                    "refine_logs": st.session_state.get("refine_logs", []),
                    "refine_iteration_count": len(st.session_state.get("refine_logs", []))
                }

                st.session_state.crs_final_case = case

                # simpan_case_ke_gsheet_casebase(case, spreadsheet_id="1RlsZ4h9FLSX_2J5wNuDn_fBQcVhSAnLe3A7eXqoB9HI", sheet_name="Sheet2")
                ok, msg = simpan_case_ke_gsheet_casebase(case, spreadsheet_id="1RlsZ4h9FLSX_2J5wNuDn_fBQcVhSAnLe3A7eXqoB9HI", sheet_name="Sheet2")
                # st.success(msg) if ok else st.error(msg)
                # st.write("Debug msg:", msg)
                # st.write("Type of msg:", type(msg))
                # st.success(str(msg)) if ok else st.error(str(msg))




                # if os.path.exists("casebase.json"):
                #     with open("casebase.json", "r") as f:
                #         existing = json.load(f)
                # else:
                #     existing = []

                # existing.append(case)

                # with open("casebase.json", "w") as f:
                #     json.dump(existing, f, indent=4)

                st.success(f"Pilihan '{selected_kafe}' berhasil disimpan ke casebase.json ✅")
                st.session_state.case_dari_crs_cbr_flat = case
                st.session_state.simpan_has_clicked = True

            if st.session_state.get("simpan_has_clicked"):
                if st.button("➡️ Lanjut ke Survey"):
                    st.session_state.step = "survey_1_app1"
                    st.rerun()

        elif pilihan == "Belum puas, ingin refinement":
            if st.button("🔁 Coba Refine"):
                st.session_state.step = "crs_refine"
                st.rerun()


def step_crs_refine():
    import pandas as pd
    from collections import defaultdict
    from kategori_suasana_dict_updated import kategori_suasana

    st.subheader("🛠️ Refinement: Tambah atau Hindari Preferensi")

    prev_keywords = st.session_state.get("crs_keywords", [])
    st.markdown(f"💬 Preferensi awal kamu: **{', '.join(prev_keywords)}**")


    if "refine_logs" not in st.session_state:
        st.session_state.refine_logs = []

    # ✅ Checkbox UI untuk tambah/kurangi preferensi (sudah disamakan)
    tambah_keywords_dict = {}
    st.markdown("➕ Tambahkan atau ubah preferensi (berdasarkan preferensi awal kamu):")
    for kategori, sub_dict in kategori_suasana.items():
        with st.expander(kategori):
            for sub_label, keyword_list in sub_dict.items():
                checkbox_key = f"refine_{kategori}_{sub_label}"
                # ✅ Preferensi yang sudah ada di awal otomatis tercentang
                already_selected = all(kw in prev_keywords for kw in keyword_list)
                if st.checkbox(sub_label.title(), key=checkbox_key, value=already_selected):
                    tambah_keywords_dict[sub_label] = keyword_list

    # 🚫 Pilih kritik dari rekomendasi awal yang ingin dihindari
    st.markdown("🚫 Pilih kritik yang ingin dihindari dari ulasan sebelumnya:")
    kritik_check = []
    for k, v in st.session_state.get("kritik_dari_top5", {}).items():
        if st.checkbox(f"{v} menyebut **{k}**", key=f"chk_{k}"):
            kritik_check.append(k)
    hindari_input = kritik_check

    if st.button("🔁 Proses Ulang Rekomendasi"):
        # 🔄 Ambil semua preferensi hasil checkbox
        tambah_keywords = [kw for kws in tambah_keywords_dict.values() for kw in kws]
        full_keywords = list(set(tambah_keywords))  # Hanya ambil yang dicentang sekarang (bukan gabungan manual)

        # 🔍 Hitung similarity dan penalti
        query_vec = make_query_vector(full_keywords, model_w2v)
        similarity_scores = cosine_similarity(query_vec, X_matrix)[0]

        df_result = df_kafe.copy()
        df_result["Similarity"] = similarity_scores

        penalti_list = []
        for nama in df_result["Nama Kafe"]:
            total_kritik = hitung_total_kritik(nama, df_review, hindari_input)
            penalti_list.append(total_kritik)

        df_result["Penalti"] = penalti_list
        df_result["FinalScore"] = df_result["Similarity"] - 0.01 * df_result["Penalti"]

        df_sorted = df_result.sort_values(by="FinalScore", ascending=False)

        # ❌ Filter kafe yang mengandung kata yang ingin dihindari
        filtered_rows = []
        for _, row in df_sorted.iterrows():
            nama = row["Nama Kafe"]
            df_kafe_rows = df_review[df_review["Nama Kafe"] == nama]
            tokens_all = sum((r["tokens_negated_indo"] + r["tokens_negated_english"] for _, r in df_kafe_rows.iterrows()), [])
            if not any(k in tokens_all for k in hindari_input):
                filtered_rows.append(row)

        # ⛔ Jika hasil < 5, coba longgarin filter
        if len(filtered_rows) < 5:
            max_kritik_awal = 0
            for row in st.session_state.get("crs_result_before_refine", []):
                nama = row["Nama Kafe"]
                max_kritik_awal = max(max_kritik_awal, hitung_total_kritik(nama, df_review, hindari_input))

            for _, row in df_sorted.iterrows():
                if row in filtered_rows:
                    continue
                nama = row["Nama Kafe"]
                df_kafe_rows = df_review[df_review["Nama Kafe"] == nama]
                tokens_all = sum((r["tokens_negated_indo"] + r["tokens_negated_english"] for _, r in df_kafe_rows.iterrows()), [])
                total_kritik = sum(tokens_all.count(k) for k in hindari_input)
                if total_kritik < max_kritik_awal:
                    filtered_rows.append(row)
                if len(filtered_rows) >= 5:
                    break

        top_kafe = pd.DataFrame(filtered_rows).head(5)

        # ✅ Tampilkan hasil
        st.success("🔍 Berikut hasil rekomendasi setelah refinement:")
        for _, row in top_kafe.iterrows():
            tampilkan_kafe_dengan_detail(
                row,
                full_keywords,
                model_w2v,
                df_review,
                vector_cols=[col for col in df_kafe.columns if col.startswith("dim_")],
                kata_kritik_umum=kata_kritik_umum,
                preferensi_dict=st.session_state.get("preferensi_dict", {})
            )

        # 🟡 Simpan hasil ke session_state
        st.session_state.crs_keywords = full_keywords
        st.session_state.crs_result_after_refine = top_kafe.to_dict(orient="records")
        st.session_state.crs_refine_added = full_keywords  # Yang dicentang saat ini
        st.session_state.crs_refine_excluded = hindari_input
        st.session_state.crs_refine_added_label = list(tambah_keywords_dict.keys())
        # 🔁 Buat preferensi_label yang baru dari yang dicentang
        preferensi_label = {}
        for kategori, sub_dict in kategori_suasana.items():
            for sub_label in sub_dict:
                if f"refine_{kategori}_{sub_label}" in st.session_state and \
                st.session_state[f"refine_{kategori}_{sub_label}"]:
                    preferensi_label[kategori] = sub_label

        st.session_state.crs_preferensi_label = preferensi_label

        # Hitung iterasi sekarang
        iteration_now = len(st.session_state.get("refine_logs", []))
        
        # Buat log iterasi saat ini
        log_iterasi = {
            "iteration": iteration_now,
            "keywords": full_keywords,
            "refine_added": full_keywords,
            "refine_excluded": hindari_input,
            "preferensi_label": preferensi_label
        }
        
        # Simpan ke refine_logs
        if "refine_logs" not in st.session_state:
            st.session_state.refine_logs = []
        
        st.session_state.refine_logs.append(log_iterasi)


        st.session_state.step = "crs_compare"
        st.rerun()


def step_crs_compare():
    import os
    import json
    from kategori_suasana_dict_updated import kategori_suasana

    st.subheader("🔍 Perbandingan Rekomendasi Sebelum & Setelah Refinement")

    before = st.session_state.get("crs_result_before_refine")
    after = st.session_state.get("crs_result_after_refine")
    keywords = st.session_state.get("crs_keywords", [])

    if not before or not after:
        st.warning("Hasil rekomendasi belum lengkap. Silakan lakukan pencarian dan refinement terlebih dahulu.")
        return

    preferensi_label = st.session_state.get("crs_preferensi_label", {})
    label_keywords = list(preferensi_label.values())
    st.markdown(f"💬 **Preferensi setelah refinement:** {', '.join(label_keywords) or '-'}")
    jumlah_refine = max(0, len(st.session_state.get("refine_logs", [])) - 1)
    st.markdown(f"🔁 Jumlah iterasi refinement: {jumlah_refine}")

    if refine_logs:
    for log in refine_logs:
        iterasi = log.get("iteration", "-") + 1  # +1 biar user-friendly mulai dari 1
        label_keys = list(log.get("preferensi_label", {}).values())
        refine_excluded = log.get("refine_excluded", [])

        st.markdown(f"### Iterasi {iterasi}:")

        # Preferensi ditambahkan ditampilkan dalam bentuk sub_label
        if label_keys:
            st.markdown(f"- ➕ Preferensi Ditambahkan: {', '.join(label_keys)}")
        else:
            st.markdown(f"- ➕ Preferensi Ditambahkan: (tidak ada)")

        if refine_excluded:
            st.markdown(f"- 🚫 Kata Dihindari: {', '.join(refine_excluded)}")
        else:
            st.markdown(f"- 🚫 Kata Dihindari: (tidak ada)")

    st.json(st.session_state.get("refine_logs", []))

    added = st.session_state.get("crs_refine_added", [])
    excluded = st.session_state.get("crs_refine_excluded", [])

    if preferensi_label:
        st.markdown(f"➕ **Preferensi yang dipilih:** {', '.join(preferensi_label.values())}")
    if excluded:
        st.markdown(f"🚫 **Kata yang dihindari:** {', '.join(excluded)}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # ✅ Build preferensi_dict yang benar: sub_label → keyword_list
    preferensi_dict = {
        sub_label: kategori_suasana[kategori][sub_label]
        for kategori, sub_label in preferensi_label.items()
    }

    with col1:
        st.markdown("### 🔵 Sebelum Refinement")
        for row in before:
            tampilkan_kafe_dengan_detail(
                row,
                keywords,
                model_w2v,
                df_review,
                vector_cols=[col for col in df_kafe.columns if col.startswith("dim_")],
                kata_kritik_umum=kata_kritik_umum,
                preferensi_dict=preferensi_dict
            )

    with col2:
        st.markdown("### 🟢 Setelah Refinement")
        for row in after:
            tampilkan_kafe_dengan_detail(
                row,
                keywords,
                model_w2v,
                df_review,
                vector_cols=[col for col in df_kafe.columns if col.startswith("dim_")],
                kata_kritik_umum=kata_kritik_umum,
                preferensi_dict=preferensi_dict
            )

    st.markdown("### ✅ Menurut kamu, hasil mana yang lebih cocok?")
    pilihan = st.radio("Jawaban kamu:", ["Sebelum refinement", "Setelah refinement", "Sama saja"], key="pilih_compare")

    if pilihan == "Sebelum refinement":
        kafe_list = [row["Nama Kafe"] for row in before]
    elif pilihan == "Setelah refinement":
        kafe_list = [row["Nama Kafe"] for row in after]
    else:
        kafe_list = [row["Nama Kafe"] for row in after]

    st.markdown("---")
    st.info("Kalau hasil ini masih belum cocok juga, kamu bisa refine lagi.")
    
    if st.button("🔁 Refinement Masih Belum Memuaskan"):
        st.session_state.step = "crs_refine"
        st.rerun()

    
    st.markdown("---")
    st.markdown("### 📌 Pilih salah satu kafe final yang menurut kamu PALING cocok:")
    selected_kafe = st.radio("Pilih kafe:", kafe_list, key="kafe_final")

    if "simpan_compare_has_clicked" not in st.session_state:
        st.session_state.simpan_compare_has_clicked = False

    if st.button("💾 Simpan Pilihan"):
        case = {
            "selected_kafe": selected_kafe,
            "crs_keywords": keywords,
            "preferensi_label": preferensi_label,
            "refine_added": added,
            "refine_excluded": excluded,
            "compare_choice": pilihan,
            "user_identity": st.session_state.get("user_identity", {}),
            "refine_logs": st.session_state.get("refine_logs", []),
            "refine_iteration_count": len(st.session_state.get("refine_logs", [])),
        }

        st.session_state.crs_final_case = case


        ok, msg = simpan_case_ke_gsheet_casebase(case, spreadsheet_id="1RlsZ4h9FLSX_2J5wNuDn_fBQcVhSAnLe3A7eXqoB9HI", sheet_name="Sheet2")
        # st.success(msg) if ok else st.error(msg)
        # st.write("💬 Isi msg:")
        # st.write(type(msg), msg)

        # st.success(str(msg)) if ok else st.error(str(msg))


        # if os.path.exists("casebase.json"):
        #     with open("casebase.json", "r") as f:
        #         existing = json.load(f)
        # else:
        #     existing = []

        # existing.append(case)

        # with open("casebase.json", "w") as f:
        #     json.dump(existing, f, indent=4)

        st.success(f"Pilihan kamu '{selected_kafe}' berhasil disimpan ke casebase.json ✅")
        st.session_state.simpan_compare_has_clicked = True

    if st.session_state.simpan_compare_has_clicked:
        if st.button("➡️ Lanjut ke Survey"):
            st.session_state.compare_choice = pilihan
            st.session_state.step = "survey_1_app1"
            st.rerun()


def step_survey_1_app1():
    st.subheader("📋 Survei Pengalaman - Aplikasi ke-1 (Query-Based)")

    st.markdown("""
    Survei ini bertujuan untuk mengetahui pengalamanmu saat menggunakan **Aplikasi 1**.

    Jika lupa, Aplikasi 1 adalah sistem rekomendasi kafe berdasarkan pencarian kata kunci secara eksplisit di ulasan pengguna.  
    Kamu memilih kata kunci tertentu (misalnya 'tenang', 'wifi', 'murah'), lalu sistem mencari kafe berdasarkan ulasan yang menyebutkan kata-kata tersebut.
    """)

    st.markdown("---")
    st.markdown("✅ Beri centang pada pernyataan yang kamu **setujui**, dan biarkan kosong jika **tidak setuju**.")
    st.markdown("---")

    statements = {
        "prq_1": "Saya sangat menyukai rekomendasi kafe yang saya dapatkan.",
        "prq_2": "Saya tidak menyukai cara interaksi dengan sistem ini.",  # ✘ negatif
        "pe_1": "Saya bisa menemukan kafe yang sesuai preferensi dengan cepat.",
        "tr_1": "Saya benar-benar mempertimbangkan untuk mengunjungi kafe ini suatu saat nanti.",
        "tr_2": "Saya tertarik menggunakan sistem ini lagi di lain waktu.",
        "inf_1": "Saya dapat dengan mudah menemukan informasi tentang kafe yang direkomendasikan.",
        "etu_1": "Secara keseluruhan, saya kesulitan menemukan kafe yang sesuai dengan keinginan saya.",  # ✘ negatif
        "etu_2": "Saya tidak mengalami kesulitan dalam menggunakan sistem ini.",
        "eou_1": "Pertanyaan dan pilihan yang diberikan sistem mudah dipahami.",
        "eou_2": "Saya memahami semua instruksi yang diberikan dalam sistem."
    }

    survey_answers_app1 = {}

    for key, text in statements.items():
        survey_answers_app1[key] = st.checkbox(f"{text}", key=f"survey1_{key}")

    saran = st.text_area("📝 Saran atau komentar tambahan (opsional)", key="survey1_saran")
    survey_answers_app1["saran"] = saran

    if st.button("➡️ Lanjut ke Survei Aplikasi 2"):
        st.session_state.survey_1_app1_feedback = survey_answers_app1
        st.session_state.step = "survey_1_app2"
        st.rerun()


def step_survey_1_app2():
    st.subheader("📋 Survei Pengalaman - Aplikasi ke-2 (Case-Based/CRS)")

    st.markdown("""
    Survei ini bertujuan untuk mengetahui pengalamanmu saat menggunakan **Aplikasi 2**.

    Jika lupa, Aplikasi 2 adalah sistem rekomendasi kafe berdasarkan **kemiripan makna** menggunakan pendekatan Word2Vec.  
    Sistem mencoba memahami makna dari preferensimu, dan merekomendasikan kafe yang memiliki makna serupa dalam ulasan.
    """)

    st.markdown("---")
    st.markdown("✅ Beri centang pada pernyataan yang kamu **setujui**, dan biarkan kosong jika **tidak setuju**.")
    st.markdown("---")

    statements = {
        "prq_1": "Saya sangat menyukai rekomendasi kafe yang saya dapatkan.",
        "prq_2": "Saya tidak menyukai cara interaksi dengan sistem ini.",  # ✘ negatif
        "pe_1": "Saya bisa menemukan kafe yang sesuai preferensi dengan cepat.",
        "tr_1": "Saya benar-benar mempertimbangkan untuk mengunjungi kafe ini suatu saat nanti.",
        "tr_2": "Saya tertarik menggunakan sistem ini lagi di lain waktu.",
        "inf_1": "Saya dapat dengan mudah menemukan informasi tentang kafe yang direkomendasikan.",
        "etu_1": "Secara keseluruhan, saya kesulitan menemukan kafe yang sesuai dengan keinginan saya.",  # ✘ negatif
        "etu_2": "Saya tidak mengalami kesulitan dalam menggunakan sistem ini.",
        "eou_1": "Pertanyaan dan pilihan yang diberikan sistem mudah dipahami.",
        "eou_2": "Saya memahami semua instruksi yang diberikan dalam sistem."
    }

    survey_answers_app2 = {}

    for key, text in statements.items():
        survey_answers_app2[key] = st.checkbox(f"{text}", key=f"survey2_{key}")

    saran = st.text_area("📝 Saran atau komentar tambahan (opsional)", key="survey2_saran")
    survey_answers_app2["saran"] = saran

    if st.button("➡️ Lanjut ke Survei Perbandingan"):
        st.session_state.survey_1_app2_feedback = survey_answers_app2
        st.session_state.step = "survey_2"
        st.rerun()


def step_survey_2():
    st.subheader("⚖️ Survei Perbandingan Sistem")

    st.markdown("Sistem mana yang paling kamu sukai secara keseluruhan?")
    favorit = st.radio(
        options=["Aplikasi 2 (Case-Based/CRS)", "Aplikasi 1 (Query-Based)"],
        label="",
        key="fav_survey2"
    )

    st.markdown("Kenapa kamu lebih menyukai sistem tersebut?")
    alasan = st.text_area("📝 Jelaskan alasanmu secara singkat", key="alasan_survey2")

    st.markdown("Sistem mana yang menurutmu paling akurat dalam menghasilkan rekomendasi?")
    akurat = st.radio(
        options=["Aplikasi 2 (Case-Based/CRS)", "Aplikasi 1 (Query-Based)"],
        label="",
        key="eff_survey2"
    )

    if st.button("✅ Selesai & Tampilkan Rangkuman"):
        st.session_state.survey_2_feedback = {
            "favorit": favorit,
            "alasan": alasan,
            "akurat": akurat
        }
        st.session_state.step = "pamit"
        st.rerun()


# def step_pamit():
#     st.subheader("🎉 Terima kasih telah berpartisipasi!")

#     st.markdown("Berikut adalah rangkuman data dari seluruh proses yang telah kamu lalui:")

#     gc = pygsheets.authorize(service_account_file="kodeRahasia_jangandiShare.json")


#     # 1. Identitas
#     st.markdown("### 👤 Identitas Pengguna")
#     st.json(st.session_state.get("user_identity", "Belum diisi"))

#     st.markdown("---")

#     # 2. Aplikasi 1: Query-Based
#     st.markdown("### 🔍 Preferensi & Hasil - Aplikasi 1 (Query-Based)")
#     if "query_input" in st.session_state:
#         st.markdown("**Preferensi yang dimasukkan:**")
#         st.json(st.session_state["query_input"])
#     else:
#         st.info("Kamu tidak mencoba Aplikasi 1.")

#     if "query_result" in st.session_state:
#         st.markdown("**Hasil Rekomendasi:**")
#         for nama_kafe, mention_dict, subaspek_match, total in st.session_state["query_result"]:
#             st.markdown(f"- ⭐ **{nama_kafe}** — {total} sebutan relevan, {subaspek_match} aspek cocok")

#     st.markdown("---")

#     # 3. Aplikasi 2: CRS
#     st.markdown("### 🤖 Preferensi & Hasil - Aplikasi 2 (CRS-CBR)")

#     if "crs_keywords" in st.session_state:
#         st.markdown("**Preferensi (semua keyword):**")
#         st.code(", ".join(st.session_state["crs_keywords"]))

#     if "crs_preferensi_label" in st.session_state:
#         st.markdown("**Preferensi label (kategori → sub):**")
#         st.json(st.session_state["crs_preferensi_label"])

#     # Menampilkan hasil akhir dari CRS (hanya 1 kafe)
#     if "crs_final_case" in st.session_state:
#         st.markdown("**Hasil rekomendasi terakhir yang dipilih:**")
#         kafe_final = st.session_state["crs_final_case"]["selected_kafe"]
#         hasil = st.session_state.get("crs_result_after_refine") or st.session_state.get("crs_result_before_refine")

#         if hasil:
#             for row in hasil:
#                 if row["Nama Kafe"] == kafe_final:
#                     st.markdown(f"- ⭐ **{kafe_final}** (sim: {row['Similarity']:.2f}, score: {row['FinalScore']:.2f})")
#                     break
#         else:
#             st.markdown(f"- ⭐ **{kafe_final}**")
#     else:
#         st.info("Kamu belum memilih kafe akhir dari CRS.")

#     st.markdown("---")

#     # 4. Refinement (jika ada)
#     st.markdown("### 🛠️ Refinement (Jika dilakukan)")
#     if st.session_state.get("crs_refine_added") or st.session_state.get("crs_refine_excluded"):
#         st.markdown("**Preferensi tambahan (dari checkbox):**")
#         st.code(", ".join(st.session_state.get("crs_refine_added", [])))

#         st.markdown("**Kata yang dihindari:**")
#         st.code(", ".join(st.session_state.get("crs_refine_excluded", [])))
#     else:
#         st.info("Kamu tidak melakukan refinement.")

#     st.markdown("---")

#     # 5. Survey Aplikasi 1
#     st.markdown("### 📝 Survei Aplikasi 1 (Query-Based)")
#     if "survey_1_app1_feedback" in st.session_state:
#         st.json(st.session_state["survey_1_app1_feedback"])
#     else:
#         st.info("Belum mengisi survei ini.")

#     # 6. Survey Aplikasi 2
#     st.markdown("### 📝 Survei Aplikasi 2 (CRS-CBR)")
#     if "survey_1_app2_feedback" in st.session_state:
#         st.json(st.session_state["survey_1_app2_feedback"])
#     else:
#         st.info("Belum mengisi survei ini.")

#     # 7. Survey Perbandingan
#     st.markdown("### ⚖️ Survei Perbandingan")
#     if "survey_2_feedback" in st.session_state:
#         st.json(st.session_state["survey_2_feedback"])
#     else:
#         st.info("Belum mengisi survei ini.")

#     st.markdown("---")
#     st.success("🎉 Terima kasih banyak! Semua data sudah terekam. 🙏")

#     data_user = {
#         "dataIdentitas_user" : st.session_state.get("user_identity", "Belum diisi"),
#         "dataQuery_input" : st.session_state["query_input"],
#         "dataQuery_result" : st.session_state["query_result"],
#         "dataCrs_keywords" : st.session_state["crs_keywords"],
#         "dataCrs_preferensi" : st.session_state["crs_preferensi_label"],
#         "dataCrs_result_akhir" : st.session_state["crs_final_case"],
#         "dataCrs_refine_added" : st.session_state.get("crs_refine_added", {}),
#         "dataCrs_refine_excluded" : st.session_state.get("crs_refine_excluded", {}),
#         "dataSurvey_1_app1_feedback" : st.session_state["survey_1_app1_feedback"],
#         "dataSurvey_1_app2_feedback" : st.session_state["survey_1_app2_feedback"],
#         "dataSurvey_2_feedback" : st.session_state["survey_2_feedback"]
#     }


#     if st.button("💾 Simpan Data ke JSON"):
#         simpan_data_user()
#         # append_dict_ke_gsheet(data_user, sheet_name="Sheet1", gsheet_index=0)
#         success, message = kirim_data_ke_gsheet(
#             data_user,
#             spreadsheet_id="1RlsZ4h9FLSX_2J5wNuDn_fBQcVhSAnLe3A7eXqoB9HI",
#             sheet_name="Sheet1"  # Nama sheet di dalam spreadsheet
#         )
#         st.success(message) if success else st.error(message)

def step_pamit():
    st.subheader("🎉 Terima kasih telah berpartisipasi!")

    st.markdown("Berikut adalah rangkuman data dari seluruh proses yang telah kamu lalui:")

    # ✅ Bagian GSheet: Tidak perlu authorize manual di sini, dilakukan di fungsi kirim
    # GC pygsheets sudah dipakai di fungsi kirim_data_ke_gsheet()

    # 1. Identitas
    st.markdown("### 👤 Identitas Pengguna")
    st.json(st.session_state.get("user_identity", "Belum diisi"))

    st.markdown("---")

    # 2. Aplikasi 1: Query-Based
    st.markdown("### 🔍 Preferensi & Hasil - Aplikasi 1 (Query-Based)")
    if "query_input" in st.session_state:
        st.markdown("**Preferensi yang dimasukkan:**")
        st.json(st.session_state["query_input"])
    else:
        st.info("Kamu tidak mencoba Aplikasi 1.")

    if "query_result" in st.session_state:
        st.markdown("**Hasil Rekomendasi:**")
        for nama_kafe, mention_dict, subaspek_match, total in st.session_state["query_result"]:
            st.markdown(f"- ⭐ **{nama_kafe}** — {total} sebutan relevan, {subaspek_match} aspek cocok")

    st.markdown("---")

    # 3. Aplikasi 2: CRS-CBR
    st.markdown("### 🤖 Preferensi & Hasil - Aplikasi 2 (CRS-CBR)")

    if "crs_keywords" in st.session_state:
        st.markdown("**Preferensi (semua keyword):**")
        st.code(", ".join(st.session_state["crs_keywords"]))

    if "crs_preferensi_label" in st.session_state:
        st.markdown("**Preferensi label (kategori → sub):**")
        st.json(st.session_state["crs_preferensi_label"])

    # Tampilkan hasil akhir CRS
    if "crs_final_case" in st.session_state:
        st.markdown("**Hasil rekomendasi terakhir yang dipilih:**")
        kafe_final = st.session_state["crs_final_case"]["selected_kafe"]
        hasil = st.session_state.get("crs_result_after_refine") or st.session_state.get("crs_result_before_refine")

        if hasil:
            for row in hasil:
                if row["Nama Kafe"] == kafe_final:
                    st.markdown(f"- ⭐ **{kafe_final}** (sim: {row['Similarity']:.2f}, score: {row['FinalScore']:.2f})")
                    break
        else:
            st.markdown(f"- ⭐ **{kafe_final}**")
    else:
        st.info("Kamu belum memilih kafe akhir dari CRS.")

    st.markdown("---")

    # 4. Refinement
    st.markdown("### 🛠️ Refinement (Jika dilakukan)")
    if st.session_state.get("crs_refine_added") or st.session_state.get("crs_refine_excluded"):
        st.markdown("**Preferensi tambahan:**")
        st.code(", ".join(st.session_state.get("crs_refine_added", [])))
        st.markdown("**Kata yang dihindari:**")
        st.code(", ".join(st.session_state.get("crs_refine_excluded", [])))
    else:
        st.info("Kamu tidak melakukan refinement.")

    st.markdown("---")

    # 5. Survei
    st.markdown("### 📝 Survei Aplikasi 1 (Query-Based)")
    st.json(st.session_state.get("survey_1_app1_feedback", {}))

    st.markdown("### 📝 Survei Aplikasi 2 (CRS-CBR)")
    st.json(st.session_state.get("survey_1_app2_feedback", {}))

    st.markdown("### ⚖️ Survei Perbandingan")
    st.json(st.session_state.get("survey_2_feedback", {}))

    st.markdown("---")

    # ✅ Data yang akan dikirim ke GSheet
    data_user = {
        "dataIdentitas_user": st.session_state.get("user_identity", {}),
        "dataQuery_input": st.session_state.get("query_input", {}),
        "dataQuery_result": st.session_state.get("query_result", []),
        "dataCrs_keywords": st.session_state.get("crs_keywords", []),
        "dataCrs_preferensi": st.session_state.get("crs_preferensi_label", {}),
        "dataCrs_result_akhir": st.session_state.get("crs_final_case", {}),
        "dataCrs_refine_logs": st.session_state.get("refine_logs", []),
        "dataCrs_refine_added": st.session_state.get("crs_refine_added", []),
        "dataCrs_refine_excluded": st.session_state.get("crs_refine_excluded", []),
        "dataSurvey_1_app1_feedback": st.session_state.get("survey_1_app1_feedback", {}),
        "dataSurvey_1_app2_feedback": st.session_state.get("survey_1_app2_feedback", {}),
        "dataSurvey_2_feedback": st.session_state.get("survey_2_feedback", {})
    }

    # ✅ Tombol Simpan ke JSON + GSheet
    if st.button("💾 Simpan Data ke JSON & GSheet"):
        # simpan_data_user()  # Simpan ke lokal JSON juga (opsional)

        

        # Kirim ke GSheet
        success, message = kirim_data_ke_gsheet(
            data_user,
            spreadsheet_id="1RlsZ4h9FLSX_2J5wNuDn_fBQcVhSAnLe3A7eXqoB9HI",
            sheet_name="Sheet1"
        )
        # st.success(message) if success else st.error(message)
        
        st.success("🎉 Terima kasih banyak! Semua data sudah terekam. 🙏")
        st.success("🔁 Kalau mau coba lagi, silakan **refresh halamannya** ya.")







# ==================
# KUMPULAN FUNGSI SUPPORTING ALGORITHM
# ==================
def query_filter_kafe(df_review, keyword_list):
    keyword_list = [k.lower().strip() for k in keyword_list if k.strip()]
    df_filtered = df_review.copy()

    if keyword_list:
        def contains_keywords(row):
            tokens = row['tokens_negated_indo'] + row['tokens_negated_english']
            return all(k in tokens for k in keyword_list)

        df_filtered = df_filtered[df_filtered.apply(contains_keywords, axis=1)]

    return df_filtered.groupby("Nama Kafe").first().reset_index()

# 🟢 Ini fungsi global, aman untuk cache
def int_default():
    return defaultdict(int)

#@st.cache_data
#def get_keyword_mentions_per_kafe(df_review, keywords):
#    hasil = defaultdict(int_default)  # gunakan fungsi global tadi
#    hasil = {}
#
#    for _, row in df_review.iterrows():
#        nama_kafe = row['Nama Kafe']
#        tokens = row['tokens_negated_indo'] + row['tokens_negated_english']
#        nama_kafe = row["Nama Kafe"]
#        tokens = row["tokens_negated_indo"] + row["tokens_negated_english"]
#
#        if nama_kafe not in hasil:
#            hasil[nama_kafe] = {}
#
#        for k in keywords:
#            hasil[nama_kafe][k] += tokens.count(k.lower())
#            hasil[nama_kafe][k] = hasil[nama_kafe].get(k, 0) + tokens.count(k.lower())
#
#    return hasil
from collections import defaultdict

@st.cache_data
def get_keyword_mentions_per_kafe(df_review, keywords):
    # Nested defaultdict: hasil[nama_kafe][keyword] = jumlah
    hasil = defaultdict(lambda: defaultdict(int))

    for _, row in df_review.iterrows():
        nama_kafe = row['Nama Kafe']
        tokens = row['tokens_negated_indo'] + row['tokens_negated_english']
        for k in keywords:
            hasil[nama_kafe][k] += tokens.count(k.lower())

    return dict(hasil)  # Convert to normal dict biar aman buat caching


def make_query_vector(keywords, model, vector_size=100):
    vectors = [model.wv[k] for k in keywords if k in model.wv]
    if vectors:
        return np.mean(vectors, axis=0).reshape(1, -1)
    return np.zeros((1, vector_size))

kata_kritik_umum = ["mahal", "rame", "berisik", "bising", "lambat", "pelayan_lama", "kotor",
    "sempit", "panas", "gerah", "jutek", "antri", "macet", "crowded",
    "tidak_bersih", "tidak_aman", "overpriced"]

def get_kritik_negatif(nama_kafe, df_review, kritik_list, return_dict=False):
    from collections import defaultdict

    kritik_dict = defaultdict(int)
    df_kafe = df_review[df_review["Nama Kafe"] == nama_kafe]

    for _, row in df_kafe.iterrows():
        tokens = row["tokens_negated_indo"] + row["tokens_negated_english"]
        for k in kritik_list:
            kritik_dict[k] += tokens.count(k)

    kritik_filtered = {k: v for k, v in kritik_dict.items() if v > 0}

    if return_dict:
        return kritik_filtered

    if kritik_filtered:
        kritik_lines = "\n".join([f"- {v} menyebut **{k}**" for k, v in sorted(kritik_filtered.items(), key=lambda x: -x[1])])
        return f"⚠️ Kritik umum ditemukan:\n{kritik_lines}"
    else:
        return "⚠️ Tidak ditemukan kritik umum di review."



    # Filter hanya yang > 0
    kritik_filtered = {k: v for k, v in kritik_dict.items() if v > 0}

    if kritik_filtered:
        kritik_str = ", ".join([f"{v} menyebut '{k}'" for k, v in sorted(kritik_filtered.items(), key=lambda x: -x[1])])
    else:
        kritik_str = "Tidak ditemukan kritik umum di review."

    return kritik_str

def tampilkan_kafe_dengan_detail(row, keywords, model, df_review, vector_cols, kata_kritik_umum, preferensi_dict):
    import numpy as np
    from collections import defaultdict

    nama_kafe = row["Nama Kafe"]
    kafe_vec = np.array([row[col] for col in vector_cols])  # aman

    similarity = row["Similarity"]
    sentiment = row.get("avg_sentiment", None)
    final_score = row["FinalScore"]

    # Hitung similarity per sub-aspek (gabungan keyword dalam preferensi_dict)
    cocok_sub = []
    for sub_label, keyword_list in preferensi_dict.items():
        keyword_vec = make_query_vector(keyword_list, model)
        sim_score = cosine_similarity(keyword_vec, [kafe_vec])[0][0]
        if sim_score > 0.3:  # ambang minimal relevansi
            cocok_sub.append((sub_label, sim_score))

    # Susun kalimat cocok dengan preferensi
    if cocok_sub:
        cocok_sub = sorted(cocok_sub, key=lambda x: -x[1])  # urutkan dari sim tertinggi
        sub_names = [s[0] for s in cocok_sub]
        avg_sim = np.mean([s[1] for s in cocok_sub])
        if len(sub_names) == 1:
            cocok_str = sub_names[0]
        else:
            cocok_str = ", ".join(sub_names[:-1]) + " dan " + sub_names[-1]
        cocok_str += f" (sim: {avg_sim:.2f})"
    else:
        cocok_str = "-"

    # Mention ulasan
    mention_dict = get_keyword_mentions_per_kafe(df_review[df_review["Nama Kafe"] == nama_kafe], keywords).get(nama_kafe, {})
    mention_str_parts = [f"{v} menyebut '{k}'" for k, v in mention_dict.items() if v > 0]
    mention_str = ", ".join(mention_str_parts) if mention_str_parts else "Tidak ada ulasan relevan."

    # Kritik umum
    kritik_str = get_kritik_negatif(nama_kafe, df_review, kata_kritik_umum)

    # Tampilkan
    st.markdown(f"### ⭐ {nama_kafe}")
    st.markdown(f"- Similarity Score     : `{similarity:.4f}`")
    if sentiment is not None:
        st.markdown(f"- Avg Sentiment        : `{sentiment:.2f}`")
    st.markdown(f"- Final Score (penalti): `{final_score:.4f}`")
    st.markdown(f"- ✅ Cocok dengan preferensi: {cocok_str}")
    st.markdown(f"- 📊 {mention_str}")
    st.markdown(f"- {kritik_str}")
    # Tambahkan kata yang dihindari kalau ada
    excluded_words = st.session_state.get("crs_refine_excluded", [])
    if excluded_words:
        st.markdown(f"- 🚫 Menghindari kata: `{', '.join(excluded_words)}`")
    st.markdown("---")



def aspek_yang_cocok(kafe_vec, keywords, model):
    from kategori_suasana_dict_updated import kategori_suasana
    label_sim = {}

    for k in keywords:
        if k in model.wv:
            sim = cosine_similarity([model.wv[k]], [kafe_vec])[0][0]
            label = get_label_dari_keyword(k)  # ← ini map raw ke label
            if label not in label_sim or sim > label_sim[label]:
                label_sim[label] = sim

    cocok = sorted(label_sim.items(), key=lambda x: -x[1])
    return [(label, f"{sim:.2f}") for label, sim in cocok]


def hitung_total_kritik(nama_kafe, df_review, kritik_list):
    kritik_dict = defaultdict(int)
    df_kafe = df_review[df_review["Nama Kafe"] == nama_kafe]
    for _, row in df_kafe.iterrows():
        tokens = row["tokens_negated_indo"] + row["tokens_negated_english"]
        for k in kritik_list:
            kritik_dict[k] += tokens.count(k)
    return sum(kritik_dict.values())

def ambil_kritik_dari_top_kafe(top_kafe, df_review, kritik_list):
    kritik_terpakai = set()
    for row in top_kafe:
        nama_kafe = row["Nama Kafe"]
        df_kafe = df_review[df_review["Nama Kafe"] == nama_kafe]
        for _, r in df_kafe.iterrows():
            tokens = r["tokens_negated_indo"] + r["tokens_negated_english"]
            for k in kritik_list:
                if k in tokens:
                    kritik_terpakai.add(k)
    return sorted(list(kritik_terpakai))

def ambil_kritik_dict(nama_kafe, df_review, kritik_list):
    kritik_dict = defaultdict(int)
    df_kafe = df_review[df_review["Nama Kafe"] == nama_kafe]
    for _, row in df_kafe.iterrows():
        tokens = row["tokens_negated_indo"] + row["tokens_negated_english"]
        for k in kritik_list:
            kritik_dict[k] += tokens.count(k)
    return {k: v for k, v in kritik_dict.items() if v > 0}

def get_labels_dari_keywords(keywords, kategori_suasana):
    label_set = set()
    for kategori, sub_dict in kategori_suasana.items():
        for sub_label, keyword_list in sub_dict.items():
            if any(k in keyword_list for k in keywords):
                label_set.add(sub_label)
    return list(label_set)

def get_label_dari_keyword(keyword):
    from kategori_suasana_dict_updated import kategori_suasana
    for kategori, sub_dict in kategori_suasana.items():
        for sub_label, keyword_list in sub_dict.items():
            if keyword in keyword_list:
                return sub_label
    return keyword  # fallback kalau gak ketemu


def tampilkan_rekomendasi_berdasarkan_preferensi(preferensi_dict):
    if "saved_choices" in st.session_state:
        for saved in st.session_state.saved_choices:
            if saved["preferensi"] == preferensi_dict:
                st.info("🔁 Kamu memiliki preferensi yang sama dengan pengguna sebelumnya!")
                st.success(f"✅ Rekomendasi dari pengguna sebelumnya: **{saved['kafe']}**")
                return True
    return False

def simpan_ke_casebase(preferensi_dict, nama_kafe, path="data/casebase.json"):
    case = {"preferensi": preferensi_dict, "kafe": nama_kafe}

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(case)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# def cari_dari_casebase(preferensi_dict, path="data/casebase.json"):
#     if not os.path.exists(path):
#         return None

#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     # Cek apakah ada preferensi yang persis sama
#     kandidat = []
#     for item in data:
#         if item["preferensi"] == preferensi_dict:
#             kandidat.append(item["kafe"])

#     return kandidat[0] if kandidat else None

import os

def tampilkan_rekomendasi_dari_casebase(preferensi_dict):
    casebase_path = "data/casebase.json"
    if not os.path.exists(casebase_path):
        return False

    with open(casebase_path, "r") as f:
        casebase = json.load(f)

    for case in casebase:
        if case["preferensi"] == preferensi_dict:
            st.info(f"🔁 Preferensi ini pernah dipilih oleh **{case['user']}**")
            st.success(f"✅ Rekomendasi dari pengguna sebelumnya: **{case['kafe']}**")

            tampilkan_kafe_dengan_detail_dict(case["detail"])
            return True
    return False

def simpan_ke_casebase(user, preferensi, nama_kafe, detail_kafe):
    casebase_path = "data/casebase.json"
    if os.path.exists(casebase_path):
        with open(casebase_path, "r") as f:
            casebase = json.load(f)
    else:
        casebase = []

    casebase.append({
        "user": user,
        "preferensi": preferensi,
        "kafe": nama_kafe,
        "detail": detail_kafe
    })

    with open(casebase_path, "w") as f:
        json.dump(casebase, f, indent=2)

def ambil_detail_kafe(nama_kafe):
    row = df_kafe[df_kafe["Nama Kafe"] == nama_kafe].iloc[0]
    return row.to_dict()

def tampilkan_kafe_dengan_detail_dict(row):
    st.markdown(f"### ⭐ {row['Nama Kafe']}")
    st.markdown(f"- Similarity Score     : `{row.get('Similarity', '-')}`")
    st.markdown(f"- Avg Sentiment        : `{row.get('avg_sentiment', '-')}`")
    st.markdown(f"- Final Score (penalti): `{row.get('FinalScore', '-')}`")
    # Bisa tambahkan detail lain jika ingin

def cari_case_sama(casebase, keywords, preferensi_label):
    for case in casebase:
        if sorted(case.get("crs_keywords", [])) == sorted(keywords) and case.get("preferensi_label", {}) == preferensi_label:
            return case
    return None


# def baca_casebase_dari_gsheet(spreadsheet_id, sheet_name="Sheet2"):
#     import pygsheets
#     import json
#     from tempfile import NamedTemporaryFile

#     try:
#         json_key = dict(st.secrets["gcp_service_account"])
#         with NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmp:
#             json.dump(json_key, tmp)
#             gc = pygsheets.authorize(service_file=tmp.name)


#         # Ubah kolom JSON string jadi dict/list
#         json_cols = ["crs_keywords", "preferensi_label", "refine_added", "refine_excluded", "user_identity"]
#         for col in json_cols:
#             if col in df.columns:
#                 df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) and x.strip().startswith("{") or x.strip().startswith("[") else x)

#         return df.to_dict(orient="records")

#     except Exception as e:
#         st.error(f"❌ Gagal membaca CaseBase dari GSheet: {e}")
#         return []

# def baca_casebase_dari_gsheet(spreadsheet_id, sheet_name="Sheet2"):
#     import pygsheets
#     import json
#     import pandas as pd
#     from tempfile import NamedTemporaryFile

#     try:
#         # Ambil kredensial dari secrets dan simpan ke file temporer
#         json_key = dict(st.secrets["gcp_service_account"])
#         with NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmp:
#             json.dump(json_key, tmp)
#             gc = pygsheets.authorize(service_file=tmp.name)

#         # Buka spreadsheet
#         sh = gc.open_by_key(spreadsheet_id)
#         wks = sh.worksheet_by_title(sheet_name)
#         df = wks.get_as_df()

#         # Kolom-kolom yang akan di-deserialize dari string JSON
#         json_cols = ["crs_keywords", "preferensi_label", "refine_added", "refine_excluded", "user_identity"]

#         def safe_json_load(x):
#             try:
#                 if isinstance(x, str):
#                     x = x.strip()
#                     if x.startswith("{") or x.startswith("["):
#                         return json.loads(x)
#                 return x
#             except:
#                 return x  # fallback kalau gagal parsing

#         for col in json_cols:
#             if col in df.columns:
#                 df[col] = df[col].apply(safe_json_load)

#         return df.to_dict(orient="records")

#     except Exception as e:
#         st.error(f"❌ Gagal membaca CaseBase dari GSheet: {e}")
#         return []


def baca_casebase_dari_gsheet(spreadsheet_id, sheet_name="Sheet2"):
    import pygsheets
    import json

    try:
        # Gunakan secrets di deploy Streamlit Cloud
        json_key = dict(st.secrets["gcp_service_account"])
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmp:
            json.dump(json_key, tmp)
            tmp_path = tmp.name

        gc = pygsheets.authorize(service_file=tmp_path)
        sh = gc.open_by_key(spreadsheet_id)
        wks = sh.worksheet_by_title(sheet_name)
        df = wks.get_as_df()

        json_cols = ["crs_keywords", "preferensi_label", "refine_added", "refine_excluded", "user_identity"]
        for col in json_cols:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x.strip() and (x.strip().startswith("{") or x.strip().startswith("[")) else x
                )

        return df.to_dict(orient="records")

    except Exception as e:
        st.error(f"❌ Gagal membaca CaseBase dari GSheet: {e}")
        return []




def simpan_data_user(filepath="hasil_testing_user.json"):
    data = {}

    # 1. Identitas
    data["user_identity"] = st.session_state.get("user_identity", {})

    # 2. Query-based
    data["query_input"] = st.session_state.get("query_input", {})
    data["query_result"] = st.session_state.get("query_result", [])

    # 3. CRS
    data["crs_keywords"] = st.session_state.get("crs_keywords", [])
    data["crs_preferensi_label"] = st.session_state.get("crs_preferensi_label", {})
    data["crs_result_akhir"] = st.session_state.get("crs_final_case", {})

    # 4. Refinement
    data["crs_refine_added"] = st.session_state.get("crs_refine_added", [])
    data["crs_refine_excluded"] = st.session_state.get("crs_refine_excluded", [])

    # 5. Survei
    data["survey_1_app1_feedback"] = st.session_state.get("survey_1_app1_feedback", {})
    data["survey_1_app2_feedback"] = st.session_state.get("survey_1_app2_feedback", {})
    data["survey_2_feedback"] = st.session_state.get("survey_2_feedback", {})

    # Simpan ke file JSON
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

    st.success(f"✅ Data berhasil disimpan ke `{filepath}`")


def format_data_for_gsheet(data_dict):
    formatted = {}
    for k, v in data_dict.items():
        if v is None:
            formatted[k] = "N/A"
        elif isinstance(v, (dict, list)):
            try:
                formatted[k] = json.dumps(v, ensure_ascii=False)
            except:
                formatted[k] = str(v)
        elif isinstance(v, (pd.Series, pd.DataFrame)):
            formatted[k] = str(v.to_dict())
        else:
            formatted[k] = str(v)
    return formatted


def kirim_data_ke_gsheet(data_dict, spreadsheet_id, sheet_name="hasil_user_testing"):
    import pygsheets
    import json
    import pandas as pd
    import tempfile

    try:
        # gc = pygsheets.authorize(service_file=file_json_handler)  # 🟢 untuk lokal
        if "gcp_service_account" in st.secrets:
            json_key = dict(st.secrets["gcp_service_account"])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmp:
                json.dump(json_key, tmp)
                service_file_path = tmp.name

        sh = gc.open_by_key(spreadsheet_id)
        wks = sh.worksheet_by_title(sheet_name)

        formatted_data = format_data_for_gsheet(data_dict)
        wks.append_table(list(formatted_data.values()), dimension='ROWS')

        return True, "✅ Data berhasil dikirim ke Google Sheets."
    except Exception as e:
        return False, f"❌ Gagal mengirim data ke Google Sheets: {e}"


# def simpan_case_ke_gsheet_casebase(case_dict, spreadsheet_id, sheet_name="Sheet2"):
#     import pygsheets
#     import json
#     from datetime import datetime

#     try:
#         gc = pygsheets.authorize(service_file=file_json_handler)  # ganti kalau udah deploy

#         sh = gc.open_by_key(spreadsheet_id)
#         wks = sh.worksheet_by_title(sheet_name)

#         # Tambahkan timestamp kalau belum ada
#         if "timestamp" not in case_dict:
#             case_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#         # Format semua kolom jadi string / json string
#         formatted = {}
#         for k, v in case_dict.items():
#             if v is None:
#                 formatted[k] = "N/A"
#             elif isinstance(v, (dict, list)):
#                 formatted[k] = json.dumps(v, ensure_ascii=False)
#             else:
#                 formatted[k] = str(v)

#         # Kirim ke GSheet (tambah baris)
#         wks.append_table(list(formatted.values()), dimension="ROWS")
#         return True, "✅ Case berhasil ditambahkan ke CaseBase."
#     except Exception as e:
#         return False, f"❌ Gagal menambahkan case ke GSheet: {e}"


def simpan_case_ke_gsheet_casebase(case_dict, spreadsheet_id, sheet_name="Sheet2"):
    import pygsheets
    import json
    import tempfile
    from datetime import datetime
    import streamlit as st

    try:
        # 🔐 Gunakan secrets saat di-deploy
        if "gcp_service_account" in st.secrets:
            json_key = dict(st.secrets["gcp_service_account"])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmp:
                json.dump(json_key, tmp)
                service_file_path = tmp.name
        else:
            # 🧪 Untuk local dev, pakai file langsung
            service_file_path = "client_secret.json"

        # ✅ Autentikasi pygsheets
        gc = pygsheets.authorize(service_file=service_file_path)
        sh = gc.open_by_key(spreadsheet_id)
        wks = sh.worksheet_by_title(sheet_name)

        # ⏱️ Tambahkan timestamp kalau belum ada
        if "timestamp" not in case_dict:
            case_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 📦 Format semua field jadi string
        formatted = {}
        for k, v in case_dict.items():
            if v is None:
                formatted[k] = "N/A"
            elif isinstance(v, (dict, list)):
                formatted[k] = json.dumps(v, ensure_ascii=False)
            else:
                formatted[k] = str(v)

        # 📤 Append ke Google Sheets
        wks.append_table(list(formatted.values()), dimension="ROWS")
        return True, "✅ Case berhasil ditambahkan ke CaseBase."

    except Exception as e:
        return False, f"❌ Gagal menambahkan case ke GSheet: {e}"



# =================
# MAIN APP CONTROLLER
# =================

if "step" not in st.session_state:
    st.session_state.step = "intro"

if st.session_state.step == "intro":
    step_intro()
elif st.session_state.step == "identity":
    step_identity()
elif st.session_state.step == "intro_query":
    step_intro_query()
elif st.session_state.step == "query_based":
    step_query_based()
elif st.session_state.step == "intro_crs":
    step_intro_crs()
elif st.session_state.step == "crs_cbr":
    step_crs_cbr()
elif st.session_state.step == "crs_refine":
    step_crs_refine()
elif st.session_state.step == "crs_compare":
    step_crs_compare()
elif st.session_state.step == "survey_1_app1":
    step_survey_1_app1()
elif st.session_state.step == "survey_1_app2":
    step_survey_1_app2()
elif st.session_state.step == "survey_2":
    step_survey_2()
elif st.session_state.step == "pamit":
    step_pamit()
