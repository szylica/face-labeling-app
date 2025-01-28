# Face Labeling App 🎯

Aplikacja webowa do klasyfikacji emocji na podstawie zdjęć twarzy z wykorzystaniem uczenia maszynowego.

## Opis projektu 📋

Aplikacja pozwala na wykrywanie twarzy na zdjęciach oraz klasyfikację emocji przy użyciu sieci neuronowej. Użytkownicy mogą wgrywać zdjęcia, klasyfikować emocje oraz pobierać sklasyfikowane zdjęcia w paczce ZIP.

## Funkcje 🚀

- Rejestracja i logowanie użytkowników
- Automatyczna detekcja twarzy (MTCNN)
- Klasyfikacja 8 podstawowych emocji
- Wskaźniki pewności predykcji
- Ręczna weryfikacja wyników
- Możliwość pominięcia zdjęć
- Eksport sklasyfikowanych zdjęć

## Technologie 💻

- Python 3.8+
- TensorFlow
- Flask
- OpenCV
- MTCNN
- Bootstrap 5

## Wymagania systemowe 🔧

- Python 3.8 lub nowszy
- Biblioteki wymienione w `requirements.txt`

## Instalacja 📥

1. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/twoj_uzytkownik/twoje_repozytorium.git
   cd twoje_repozytorium
   ```
   Zainstaluj wymagane biblioteki:

   ```bash
   pip install -r requirements.txt
   ```
   
   Uruchomienie ▶️
   ```bash
   python app.py
   ```
   
Struktura projektu 📁
   
twoje_repozytorium/
│
├── app.py # Główny plik aplikacji Flask

├── tuner_ai_model_face_emotion.py # Skrypt do strojenia modelu AI

├── podzial_zbioru_zdjec_na_valid_test_train.py # Skrypt do podziału danych

├── requirements.txt # Lista zależności

├── emotions/ # Katalog z danymi treningowymi, walidacyjnymi i testowymi

├── models/ # Katalog z zapisanymi modelami

├── users/ # Katalog z danymi użytkowników

├── static/ # Pliki statyczne (CSS, JS, obrazy)
│ └── ...

├── templates/ # Szablony HTML

│ ├── index.html # Strona główna

│ ├── login.html # Strona logowania

│ ├── register.html # Strona rejestracji

│ ├── photos.html # Strona z przesłanymi zdjęciami

│ ├── 404.html # Strona błędu 404

│ └── 500.html # Strona błędu 500

└── README.md # Ten plik

Model AI 🧠

Model wykorzystuje architekturę CNN z:
4 warstwy konwolucyjne
Normalizacja wsadowa
Maxpooling
Warstwy dropout
Funkcja aktywacji ReLU
Klasyfikacja 8 emocji

Autorzy👨‍💻:
plspry, Aquaier, Szylica, Kamil
