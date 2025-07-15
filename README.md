Sebuah aplikasi sistem rekomendasi kafe yang menggunakan konsep knowledge-based dan conversational.
Untuk knowledge-based, digunakan pendekatan case-based reasoning yang menggunakan algoritma cosine similarity untuk menghitung skor kemiripan antara keinginan user dengan data yang tersedia.
Untuk conversational, digunakan pendekatan critique-based yang menggunakan konsep system-suggested dimana sistem akan memberikan pilihan kritik kepada user untuk
mengolah kembali nilai yang sudah diberikan terus menerus hingga iterasi kritik oleh user selesai.

Sistem ini juga dilengkapi dengan analisis sentimen berbasis lexicon untuk menyaring hasil rekomendasi berdasarkan ulasan positif yang relevan terhadap preferensi pengguna.

Untuk mencoba demo aplikasi: https://rekomendasi-kafe-crscbr.streamlit.app/

Untuk menjalankan program secara lokal, file yang digunakan adalah:

streamlit_app_kafe.py

hasil_skor_dan_aspek.xlsx

case_vector_df.pkl

tokens_library.txt

word2vec_model.model

Dengan catatan: pada file streamlit_app_kafe.py, harus memodifikasi terlebih dahulu beberapa fungsi agar sistem berjalan normal.
Fungsi yang digunakan di dalam aplikasi saat ini menggunakan data yang disimpan pada cloud service untuk menambah dan membaca case-base.
Jika fungsi membaca/menambah case-base dihilangkan, sistem masih bisa bekerja untuk menyediakan rekomendasi dengan menggunakan cosine similarity.
