import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from flask import jsonify


def load_csv(file_path):
    return pd.read_csv(file_path)


def modeling(test_data):
    # Memuat semua sequence
    # Misalnya untuk satu subjek: normal walking dan fast walking sequences
    normal_seq1 = load_csv("dataset_mfei_hog_fn00.csv")
    normal_seq2 = load_csv("dataset_mfei_hog_fn01.csv")
    bag_seq1 = load_csv("dataset_mfei_hog_fb00.csv")


    # Menggabungkan semua data ke dalam satu dataframe
    # Data Pelatihan: normal_seq1 dan normal_seq
    train_data = pd.concat([normal_seq1,normal_seq2,bag_seq1])

    # Data Pengujian: normal_seq3, normal_seq4, fast_seq1, dan fast_seq2


    # Memisahkan fitur dan label
    X_train = train_data.drop('label', axis=1)  # Semua kolom kecuali label
    y_train = train_data['label']  # Kolom label

    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']

    # Membuat model K-NN dengan k=3 (misalnya)
    knn = KNeighborsClassifier(n_neighbors=1)

    # Melatih model dengan data pelatihan
    knn.fit(X_train, y_train)

    # Melakukan prediksi pada data pengujian
    predictions = knn.predict(X_test)

    # Menghitung akurasi
    accuracy = accuracy_score(y_test, predictions)
    score_f1 = f1_score(y_test, predictions, average=None)
    score_f1_avg = f1_score(y_test, predictions, average="micro")

    print(f"Akurasi model: {accuracy * 100:.2f}%")
    print(f"F1 Score: {score_f1_avg},{score_f1}")
    return score_f1_avg



def compute_svc():
        # Memuat semua sequence
    # Misalnya untuk satu subjek: normal walking dan fast walking sequences
    folder_path = os.path.join(os.path.dirname(__file__), 'training_data',)
    normal_seq1 = load_csv(os.path.join(folder_path, 'dataset_mfei_hog_fn00.csv'))
    normal_seq2 = load_csv(os.path.join(folder_path, 'dataset_mfei_hog_fn01.csv'))
    bag_seq1 = load_csv(os.path.join(folder_path, 'dataset_mfei_hog_fb00.csv'))
    testing_data = load_csv("testing_data.csv")


    # Menggabungkan semua data ke dalam satu dataframe
    # Data Pelatihan: normal_seq1 dan normal_seq
    train_data = pd.concat([normal_seq1,normal_seq2,bag_seq1])

    # Data Pengujian: normal_seq3, normal_seq4, fast_seq1, dan fast_seq2


    # Memisahkan fitur dan label
    X_train = train_data.drop('label', axis=1)  # Semua kolom kecuali label
    y_train = train_data['label']  # Kolom label

    # X_test = test_data.drop('label', axis=1)
    # y_test = test_data['label']
    svc_calculate = make_pipeline(SVC(gamma="auto",kernel="linear"))
    svc_calculate.fit(X_train, y_train)
    y_pred = svc_calculate.predict(testing_data)

    print(f"predicted {y_pred[0]}")
    # 6. Cek akurasi
    # accuracy = accuracy_score(y_test, y_pred)
    # score_f1 = f1_score(y_test, y_pred, average=None)
    # score_f1_avg = f1_score(y_test, y_pred, average="micro")
    # print(f'Akurasi: {accuracy * 100:.2f}%')
    # print(f"F1 Score: {score_f1_avg},{score_f1}")
    return f"predicted {y_pred[0]}"