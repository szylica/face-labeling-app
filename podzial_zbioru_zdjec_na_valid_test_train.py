import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import cv2
import os
import shutil
import random

# Definicja klas emocji po polsku
emocje = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def podziel_dane(katalog_zrodlowy, katalog_docelowy, proporcje=(0.7, 0.2, 0.1)):
    """
    Funkcja dzieląca dane na zbiory treningowy, walidacyjny i testowy
    
    Parametry:
    katalog_zrodlowy: ścieżka do katalogu z wszystkimi obrazami
    katalog_docelowy: ścieżka gdzie mają być zapisane podzielone dane
    proporcje: krotka (trening, walidacja, test) określająca podział danych
    """
    
    # Sprawdzenie czy proporcje sumują się do 1
    if not sum(proporcje) >= 0.99 and not sum (proporcje) <= 1.01:
        raise ValueError(f"Proporcje podziału muszą sumować się do 1 {sum(proporcje)}")
    
    # Tworzenie struktury katalogów
    podzialy = ['trening', 'walidacja', 'test']
    for podzial in podzialy:
        for emocja in emocje:
            sciezka = os.path.join(katalog_docelowy, podzial, emocja)
            os.makedirs(sciezka, exist_ok=True)
    
    # Przejście przez każdą klasę emocji
    for emocja in emocje:
        print(f"Przetwarzanie emocji: {emocja}")
        
        # Ścieżka do katalogu z obrazami dla danej emocji
        katalog_emocji = os.path.join(katalog_zrodlowy, emocja)
        if not os.path.exists(katalog_emocji):
            print(f"Pominięto {emocja} - brak katalogu")
            continue
        
        # Lista wszystkich plików dla danej emocji
        pliki = [f for f in os.listdir(katalog_emocji) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(pliki)
        
        # Obliczenie liczby plików dla każdego podzbioru
        total = len(pliki)
        n_trening = int(total * proporcje[0])
        n_walidacja = int(total * proporcje[1])
        
        # Podział plików
        pliki_trening = pliki[:n_trening]
        pliki_walidacja = pliki[n_trening:n_trening + n_walidacja]
        pliki_test = pliki[n_trening + n_walidacja:]
        
        # Kopiowanie plików do odpowiednich katalogów
        for nazwa_pliku in pliki_trening:
            src = os.path.join(katalog_emocji, nazwa_pliku)
            dst = os.path.join(katalog_docelowy, 'trening', emocja, nazwa_pliku)
            shutil.copy2(src, dst)
            
        for nazwa_pliku in pliki_walidacja:
            src = os.path.join(katalog_emocji, nazwa_pliku)
            dst = os.path.join(katalog_docelowy, 'walidacja', emocja, nazwa_pliku)
            shutil.copy2(src, dst)
            
        for nazwa_pliku in pliki_test:
            src = os.path.join(katalog_emocji, nazwa_pliku)
            dst = os.path.join(katalog_docelowy, 'test', emocja, nazwa_pliku)
            shutil.copy2(src, dst)
        
        print(f"{emocja}: {len(pliki_trening)} trening, {len(pliki_walidacja)} walidacja, {len(pliki_test)} test")


podziel_dane("./input", "./emotions")
