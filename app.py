from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, send_file, session
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import random
import shutil
import zipfile
import os
import io   

## -- KONFIGURACJA SEED -- ##
# Ustawienie ziarna dla powtarzalności wyników
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

## -- KONIEC KONFIGURACJI SEED -- ##

from mtcnn import MTCNN
import cv2

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from functools import wraps
from datetime import timedelta

import logging

# Konfiguracja logowania, aby wykluczyć komunikaty debugowania PIL
logging.basicConfig(level=logging.INFO)
logging.getLogger('PIL').setLevel(logging.INFO)

## ----------- OBSŁUGA FOLDERÓW ---------- ##

# Definiowanie ścieżek do ważnych katalogów
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "emotions", "trening")
VAL_DIR = os.path.join(BASE_DIR, "emotions", "walidacja")
TEST_DIR = os.path.join(BASE_DIR, "emotions", "test")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_detection_model.h5")
USERS_DIR = os.path.join(BASE_DIR, "users")

train_dir = TRAIN_DIR
val_dir = VAL_DIR
test_dir = TEST_DIR
model_path = MODEL_PATH

# Upewnij się, że wymagane katalogi istnieją
def ensure_directories():
    """Upewnij się, że wszystkie wymagane katalogi istnieją"""
    directories = [
        os.path.join(BASE_DIR, 'static'),
        os.path.join(BASE_DIR, 'templates'),
        os.path.join(BASE_DIR, 'uploads'),
        os.path.join(BASE_DIR, 'users'),
        MODEL_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
os.makedirs(USERS_DIR, exist_ok=True)

## ----------- KONIEC OBSŁUGI FOLDERÓW ---------- ##

## ----------------- KOD MODELU ----------------- ##

# Hiperparametry modelu
batch_size = 64
img_height, img_width = 96, 96
epochs = 100
learning_rate = 0.0005

# Generatory danych
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Definicja modelu CNN
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(train_generator.num_classes, activation='softmax')
    ])

    # Kompilacja modelu
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Sprawdzenie, czy model istnieje i próba jego załadowania
def load_or_create_model():
    try:
        if os.path.exists(MODEL_PATH):
            print(f"Ładownie modelu z: {MODEL_PATH}")
            model = tf.keras.models.load_model(MODEL_PATH)
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        else:
            print("Nie znaleziono modelu, tworzenie nowego.")
            return create_model()
    except Exception as e:
        print(f"Błąd ładowania modelu: {e}")
        print("Tworzenie nowego modelu.")
        return create_model()

# Inicjalizacja modelu
model = load_or_create_model()

## ----------------- KONIEC KODU Z MODELU ----------------- ##

## ----------------- KOD APLIKACJI FLASK ----------------- ##

# Inicjalizacja aplikacji Flask
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

# Wyłączanie automatycznego przeładowywania aplikacji
app.config['DEBUG'] = True
app.config['USE_RELOADER'] = True

# Funkcja dekorująca sprawdzająca, czy użytkownik jest zalogowany
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Strona główna
@app.route('/')
@login_required
def index():
    matrix_exists = os.path.exists(os.path.join('static', 'confusion_matrix.png'))
    return render_template('index.html', 
                         prediction=request.args.get('prediction', ''),
                         username=session.get('username'),
                         show_matrix=matrix_exists,
                         matrix_path='confusion_matrix.png' if matrix_exists else None)

# Tworzenie struktury katalogów
def create_user_dataset_structure(username):
    """Utwórz strukturę katalogów dla zbioru danych użytkownika"""
    emotions = ['happy', 'sad', 'angry', 'disgust', 'fear', 'surprise', 'neutral', 'contempt']
    user_dataset = os.path.join(USERS_DIR, username, 'dataset')
    
    for emotion in emotions:
        emotion_dir = os.path.join(user_dataset, emotion)
        os.makedirs(emotion_dir, exist_ok=True)

# Klasyfikacja obrazów
@app.route('/classify_image', methods=['POST'])
def classify_image():
    try:
        data = request.get_json()
        filename = data.get('filename')
        emotion = data.get('emotion')
        username = session.get('username')

        # Log dla celów debugowania
        print(f"Otrzymano prośbę: plik={filename}, emocja={emotion}, nazwa użytkownika={username}")

        if not all([filename, emotion, username]):
            return jsonify({'success': False, 'error': 'Brakuje parametrów'})

        # Ścieżka źródłowa pliku z photos
        source_path = os.path.join(USERS_DIR, username, 'photos', filename)
        print(f"Ścieżka źródłowa: {source_path}")
        
        if not os.path.exists(source_path):
            return jsonify({'success': False, 'error': f'Nie znaleziono pliku źródłowego: {source_path}'})

        # Ścieżka docelowa pliku w folderze dataset
        target_dir = os.path.join(USERS_DIR, username, 'dataset', emotion.lower())
        os.makedirs(target_dir, exist_ok=True)
        print(f"Katalog docelowy: {target_dir}")

        target_path = os.path.join(target_dir, filename)
        print(f"Ścieżka docelowa: {target_path}")

        # Handler błędów dla przenoszenia pliku
        try:
            shutil.move(source_path, target_path)
            print(f"Pomyślnie przeniesiono plik {source_path} do {target_path}")
        except Exception as move_error:
            print(f"Błąd przenoszenia pliku: {str(move_error)}")
            return jsonify({'success': False, 'error': f'Błąd przenoszenia pliku: {str(move_error)}'})

        return jsonify({
            'success': True,
            'message': f'Zdjęcie przeniesione do folderu {emotion}',
            'new_path': target_path
        })

    except Exception as e:
        print(f"Błąd w klasyfikacji zdjęć: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# Rejestracja
@app.route('/register', methods=['GET', 'POST'])
def register():
    app.logger.debug(f"Zarejestruj ścieżkę wywołaną metodą: {request.method}")
    if request.method == 'GET':
        app.logger.debug("Renderowanie szablonu rejestracji")
        return render_template('register.html')
    
    # Obsługa formularza rejestracji
    username = request.form.get('username')
    password = request.form.get('password')

    if not username or not password:
        return jsonify({"error": "Nazwa użytkownika i hasło są wymagane"}), 400

    if not username.isalnum():
        return jsonify({"error": "Nazwa użytkownika może zawierać tylko litery i cyfry"}), 400

    # Utwórz katalog użytkownika
    user_dir = os.path.join(USERS_DIR, username)
    if os.path.exists(user_dir):
        return jsonify({"error": "Użytkownik już istnieje"}), 400

    try:
        os.makedirs(user_dir)
        user_file = os.path.join(user_dir, "user.txt")
        
        # Utwórz strukturę danych użytkownika
        create_user_dataset_structure(username)
        
        # Zapisz dane użytkownika
        with open(user_file, 'w') as f:
            f.write(f"Username:{username}\nPassword:{password}")

            create_user_dataset_structure(username)

        session['username'] = username  # Zapisz użytkownika w sesji
        flash("Rejestracja udana! Zalogowano.", "success")  # Dodaj komunikat
        return redirect(url_for('index'))  # Przekieruj na stronę główną
    except Exception as e:
        return jsonify({"error": f"Rejestracja nie powiodła się: {str(e)}"}), 500

# Logowanie
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    # Obsługa formularza logowania
    username = request.form.get('username')
    password = request.form.get('password')

    if not username or not password:
        return jsonify({"error": "Nazwa użytkownika i hasło są wymagane"}), 400

    user_dir = os.path.join(USERS_DIR, username)
    user_file = os.path.join(user_dir, "user.txt")

    if not os.path.exists(user_file):
        return jsonify({"error": "Użytkownik nie istnieje"}), 404

    # Odczyt danych użytkownika
    with open(user_file, 'r') as f:
        stored_password = f.readlines()[1].split(":")[1].strip()

    # Sprawdzenie hasła
    if stored_password != password:
        return jsonify({"error": "Błędne hasło"}), 400

    # Zaloguj użytkownika
    session['username'] = username
    flash("Pomyślnie zalogowano!", "success")
    return jsonify({"success": True, "redirect": url_for('index')})

# Wylogowanie
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Zostałeś wylogowany.", "success")
    return redirect(url_for('login'))  # Przekierowanie na stronę logowania

# Model detekcji twarzy
detektor = MTCNN()

# Przetwarzanie zdjęć
@app.route('/upload_and_predict', methods=['POST'])
@login_required
def upload_and_predict():
    # Sprawdź, czy przesłano plik
    if 'file' not in request.files:
        flash('Nie przesłano pliku', 'danger')
        return redirect(url_for('index'))
    
    # Pobierz przesłane pliki
    files = request.files.getlist('file')
    uploaded_files = []
    predictions_list = []
    
    user_folder = os.path.join(USERS_DIR, session['username'], 'photos')
    os.makedirs(user_folder, exist_ok=True)
    
    for file in files:
        if file.filename != '':
            file_path = os.path.join(user_folder, file.filename)
            file.save(file_path)

            # Wczytaj obraz
            img = cv2.imread(file_path)

            # Detekcja twarzy na zdjęciu
            try:
                results = detektor.detect_faces(img)
            except Exception as e:
                print("Błąd detekcji twarzy:", e)
                os.remove(file_path)

            faces = []
            # Wycięcie twarzy z obrazu
            for result in results:
                x, y, width, height = result['box']
                face = img[y-5:y+height+5, x-5:x+width+5]
                try:
                    # Zmniejszenie rozmiaru twarzy do 64x64
                    face_resized = cv2.resize(face, (64, 64), interpolation=cv2.INTER_AREA)
                    small_image = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))
                    
                    # Stworzenie białego tła 96x96
                    background = Image.new('RGB', (96, 96), (255, 255, 255))

                    # Pozycjonowanie twarzy na środku
                    position = ((96 - small_image.width) // 2, (96 - small_image.height) // 2)

                    # Wklejenie twarzy na białe tło
                    background.paste(small_image, position)
                    
                    # Dodanie twarzy na białym tle do listy
                    faces.append(background)
                except Exception as e:
                    print("Błąd przetwarzania twarzy:", e)

            # Zapisanie twarzy do plików
            for i, face in enumerate(faces):
                face_filename = f"{os.path.splitext(file.filename)[0]}_face_{i}.jpg"
                face_path = os.path.join(user_folder, face_filename)
                face.save(face_path)
                uploaded_files.append(face_filename)

                # Przetworzenie obrazu i predykcja emocji
                img_array = preprocess_image(face_path)
                predictions = model.predict(img_array)[0]
                
                # Sortowanie emocji według prawdopodobieństwa
                emotions = list(train_generator.class_indices.keys())
                emotion_predictions = [
                    {"emotion": emotion, "probability": float(prob)}
                    for emotion, prob in zip(emotions, predictions)
                ]
                emotion_predictions.sort(key=lambda x: x["probability"], reverse=True)
                predictions_list.append(emotion_predictions)

    # Przekierowanie na stronę z przesłanymi zdjęciami
    if uploaded_files:
        return render_template('photos.html',
                             username=session.get('username'),
                             photos=uploaded_files,
                             predictions=predictions_list)
    
    # Komunikat o błędzie
    flash('Nie przesłano żadnych poprawnych plików, na których można rozpoznać twarz', 'warning')
    return redirect(url_for('index'))

# Strona z przesłanymi zdjęciami
@app.route('/photos')
@login_required
def photos():
    return render_template('photos.html', username=session.get('username'))

@app.route('/users/<username>/photos/<filename>')
@login_required
def user_photo(username, filename):
    if username != session.get('username'):
        return redirect(url_for('login'))
    return send_from_directory(os.path.join(USERS_DIR, username, 'photos'), filename)

def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_emotion(model, img_array):
    prediction = model.predict(img_array)[0]
    top_indices = prediction.argsort()[-2:][::-1]
    top_emotions = [(list(train_generator.class_indices.keys())[i], prediction[i]) for i in top_indices]
    return top_emotions

# Trenowanie modelu
@app.route('/train', methods=['POST'])
@login_required
def train_model():
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        flash('Training and validation directories not found!', 'danger')
        return redirect(url_for('index'))

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    try:
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[early_stopping, reduce_lr]
        )

        # Ensure model directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Save the model
        model.save(MODEL_PATH)
        flash('Model trained and saved successfully!', 'success')
    except Exception as e:
        flash(f'Error during training: {str(e)}', 'danger')
    
    return redirect(url_for('index'))

# Obsługa pobierania ZIP
@app.route('/download_zip', methods=['POST'])
@login_required
def download_zip():
    try:
        # Odczyt danych z żądania
        data = request.get_json()
        files_to_zip = data.get('files', [])
        files_to_zip2 = data.get('files2', [])
        username = session.get('username')

        print(files_to_zip)
        print(files_to_zip2)

        if not files_to_zip or not username:
            return jsonify({'success': False, 'error': 'Brak plików do spakowania w ZIP'}), 400

        # Utwórz plik ZIP w pamięci
        zip_filename = f"{username}_classified_images.zip"
        memory_file = io.BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w') as zipf:
            for file_info in files_to_zip:
                file_path = os.path.join(USERS_DIR, username, 'dataset', file_info['emotion'], file_info['filename'])
                
                # Sprawdź, czy plik istnieje i zapisz jego ścieżkę
                if os.path.exists(file_path):
                    print(f"Dodawanie pliku do ZIP: {file_path}")
                    if file_path.split(os.sep)[4] == "dataset":
                        zipf.write(file_path, os.path.join(*file_path.split(os.sep)[5:]))
                    else:
                        zipf.write(file_path, os.path.join(*file_path.split(os.sep)[4:]))
                else:
                    print(f"Plik nie istnieje: {file_path}")
            
            for file_info in files_to_zip2:
                file_path = os.path.join(USERS_DIR, username, 'dataset', 'skipped', file_info)
                
                # Sprawdź, czy plik istnieje i zapisz jego ścieżkę
                if os.path.exists(file_path):
                    print(f"Dodawanie pliku do ZIP: {file_path}")
                    if file_path.split(os.sep)[4] == "dataset":
                        zipf.write(file_path, os.path.join(*file_path.split(os.sep)[5:]))
                    else:
                        zipf.write(file_path, os.path.join(*file_path.split(os.sep)[4:]))
                else:
                    print(f"Plik nie istnieje: {file_path}")

        memory_file.seek(0)
        
        # Wyślij ZIP do klienta
        return send_file(memory_file, as_attachment=True, download_name=zip_filename, mimetype='application/zip')
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Usuwanie plików
@app.route('/clear_files', methods=['POST'])
@login_required
def clear_files():
    try:
        # Odczyt danych z żądania
        files_to_delete = request.json.get('files', [])
        files_to_delete2 = request.json.get('files2', [])
        username = session.get('username')

        if not files_to_delete or not username:
            return jsonify({'success': False, 'error': 'Brak plików do usunięcia lub użytkownika'}), 400

        for file_info in files_to_delete:
            file_path = os.path.join(USERS_DIR, username, 'dataset', file_info['emotion'], file_info['filename'])
            if os.path.exists(file_path):
                os.remove(file_path)

        for file_info in files_to_delete2:
            file_path = os.path.join(USERS_DIR, username, 'dataset', 'skipped', file_info)
            if os.path.exists(file_path):
                os.remove(file_path)

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Obsługa pomijania zdjęć
@app.route('/skip_image', methods=['POST'])
def skip_image():
    data = request.get_json()
    filename = data.get('filename')
    username = session.get('username')
    
    if not username:
        return jsonify({"success": False, "error": "Użytkownik niezidentyfikowany"}), 401

    # Log dla celów debugowania
    print(f"Otrzymano prośbę: plik={filename}, skip, nazwa użytkownika={username}")

    # Ścieżka źródłowa pliku z photos
    source_path = os.path.join(USERS_DIR, username, 'photos', filename)
    print(f"Ścieżka źródłowa: {source_path}")

    if not os.path.exists(source_path):
        return jsonify({'success': False, 'error': f'Nie znaleziono pliku źródłowego: {source_path}'})
    
    # Ścieżka docelowa pliku w folderze dataset
    target_dir = os.path.join(USERS_DIR, username, 'dataset', 'skipped')
    os.makedirs(target_dir, exist_ok=True)
    print(f"Katalog docelowy: {target_dir}")

    target_path = os.path.join(target_dir, filename)
    print(f"Ścieżka docelowa: {target_path}")

    # Przenieś plik do folderu skipped i obsłuż błędy
    try:
        shutil.move(source_path, target_path)
        print(f"Pomyślnie przeniesiono plik {source_path} do {target_path}")
    except Exception as move_error:
        print(f"Błąd podczas przenoszenia pliku: {str(move_error)}")
        return jsonify({'success': False, 'error': f'Błąd podczas przenoszenia pliku: {str(move_error)}'})

    return jsonify({
        'success': True,
        'message': f'Zdjęcie przeniesione do folderu skipped',
        'new_path': target_path
    })

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Upewnij się, że wszystkie wymagane katalogi istnieją
    ensure_directories()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)