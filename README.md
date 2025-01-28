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
