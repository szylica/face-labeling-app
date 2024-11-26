# face-labeling-app

## Description
### A significant problem in the field of machine learning is datasets of facial images that are incorrectly labeled with emotions (bad labeling). Together with my team, we decided to create an application that, using a specially trained module, allows facial images to be reclassified based on the emotions they actually display. As input, users simply need to upload the JPG files they want to sort, and as output, they receive a ZIP file with the images sorted by emotion.



## **Augmentacja danych treningowych**

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
