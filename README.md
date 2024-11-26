# face-labeling-app

## Description
### A significant problem in the field of machine learning is datasets of facial images that are incorrectly labeled with emotions (bad labeling). Together with my team, we decided to create an application that, using a specially trained module, allows facial images to be reclassified based on the emotions they actually display. As input, users simply need to upload the JPG files they want to sort, and as output, they receive a ZIP file with the images sorted by emotion.


## Wyjaśnienie pliku tuner_ai_model_face_emotion.py

1. **Import bibliotek:**
   Importowane są niezbędne biblioteki do przetwarzania obrazów, budowy modelu, optymalizacji oraz logowania procesu trenowania.

2. **Ścieżki do danych:**
   Określono lokalizacje katalogów zawierających dane treningowe, walidacyjne i testowe.

3. **Parametry obrazu:**
   Obrazy są skalowane do wymiarów 150x150 pikseli, co jest zgodne z wymaganiami modelu VGG16. Batch size wynosi 64.

4. **Przygotowanie danych:**
   - **Augmentacja danych treningowych:**
     Dane są normalizowane (podzielone przez 255), a dodatkowo stosuje się augmentację, by zwiększyć różnorodność danych.
   - **Przygotowanie danych walidacyjnych:**
     Dane walidacyjne są tylko normalizowane.
   - **Generatory:**
     Generatory wczytują dane z odpowiednich katalogów i przygotowują je do treningu.

5. **Wczytanie modelu VGG16:**
   Wykorzystano pretrenowany model VGG16 (bez górnych warstw), aby skorzystać z już wytrenowanych cech.

6. **Zamrożenie wag:**
   Zablokowano warstwy modelu VGG16, by nie aktualizować ich podczas trenowania.

7. **Rozbudowa modelu:**
   Dodano nowe warstwy w pełni połączone, by dostosować model do klasyfikacji emocji. Ostatnia warstwa ma tyle neuronów, ile klas w danych treningowych.

8. **Kompilacja modelu:**
   Model jest kompilowany z optymalizatorem Adam, funkcją strat `categorical_crossentropy` i metryką `accuracy`.

9. **Callbacki:**
   - `EarlyStopping`: Zatrzymuje trenowanie, gdy wynik walidacyjny nie poprawia się przez 5 epok.
   - `ReduceLROnPlateau`: Redukuje współczynnik uczenia, gdy wynik walidacyjny nie poprawia się przez 3 epoki.
   - `TensorBoard`: Loguje informacje o treningu do wizualizacji.

10. **Trenowanie modelu:**
    Model jest trenowany przez maksymalnie 30 epok, wykorzystując dane treningowe i walidacyjne.

11. **Zapis modelu:**
    Wytrenowany model jest zapisywany jako plik `.h5`.

##
    
### **Augmentacja danych treningowych**

Augmentacja danych to technika używana w uczeniu maszynowym, aby zwiększyć różnorodność danych treningowych bez faktycznego zbierania nowych przykładów. W kontekście tego skryptu augmentacja polega na wprowadzaniu drobnych modyfikacji do obrazów wejściowych. Używane metody to:

- rescale=1./255: Normalizuje wartości pikseli do zakresu od 0 do 1 (zamiast 0-255), co ułatwia proces trenowania.
- rotation_range=20: Obraca obraz losowo o maksymalnie ±20 stopni, aby model był odporny na różne orientacje twarzy.
- width_shift_range=0.2 i height_shift_range=0.2: Przesuwa obraz w poziomie i pionie o maksymalnie 20% jego wymiarów, aby model radził sobie z przesunięciami.
- horizontal_flip=True: Odbija obraz w poziomie, co pozwala modelowi rozpoznać symetryczne cechy twarzy.

Te operacje pomagają modelowi generalizować i poprawiają jego zdolność do rozpoznawania emocji na obrazach, które różnią się od tych użytych podczas treningu.
Model VGG16

VGG16 to jeden z najbardziej znanych modeli głębokiego uczenia opracowany przez Visual Geometry Group (VGG) z Uniwersytetu Oksfordzkiego. W tym skrypcie pełni rolę pretrenowanego modelu bazowego do ekstrakcji cech z obrazów.

## **Cechy charakterystyczne VGG16:**
- Składa się z 16 warstw konwolucyjnych i w pełni połączonych.
- Wykorzystuje małe filtry konwolucyjne (3x3), co pozwala na uchwycenie drobnych cech obrazu.
- Pretrenowany na dużym zbiorze danych ImageNet, co oznacza, że "nauczył się" rozpoznawać uniwersalne cechy wizualne, takie jak krawędzie czy tekstury.

## **Jak jest używany w skrypcie:**
- weights='imagenet': Wagi modelu są inicjalizowane na podstawie wcześniejszego treningu na ImageNet, co pozwala na szybsze i bardziej efektywne uczenie na nowych danych.
- include_top=False: Usuwa górne warstwy (klasyfikator), aby umożliwić dostosowanie modelu do zadania rozpoznawania emocji.
- input_shape=(150, 150, 3): Definiuje rozmiar obrazów wejściowych (150x150 pikseli, 3 kanały kolorów RGB).

## **Zamrożenie wag:**

- conv_base.trainable = False

Oznacza, że wagi modelu VGG16 nie będą aktualizowane podczas treningu, co pozwala skupić się na nauce nowych warstw klasyfikujących emocje.

## **Zalety VGG16 w tym zadaniu:**

- Model jest dobrze wytrenowany do wykrywania ogólnych cech wizualnych.
- Jego wykorzystanie przyspiesza trenowanie, ponieważ nie trzeba uczyć modelu od podstaw (transfer learning).
