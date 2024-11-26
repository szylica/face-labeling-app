from flask import Flask, render_template, request, send_file, redirect, url_for
import zipfile
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('files[]') # Pobiera listę przesłanych plików z formularza

    if 'files[]' not in request.files:
        return redirect(url_for('index'))

    if not files:
        return redirect(url_for('index'))

    zip_buffer = BytesIO() # Tworzy bufor pamieci, w ktorym tymczasowo przechowywany bedzie plik zip, dzieki temu nie musi tworzyc pliku na dysku
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file: 
        folders = [
            'temporary', 'anger', 'contempt', 'disgust', 
            'fear', 'happy', 'neutral', 'sad', 'surprise'
        ] # tworzy foldery w zipie z emocjami jakie rozpoznaje model, dodany jest tymczasowy folder "temporary" do momentu aż nie zostanie to połączone z wytrenowanym modelem ktory automatycznie bedzie przypisywal zdjecia do folderow
        for folder in folders:
            zip_file.writestr(f"{folder}/", "")  # Tworzy te foldery
            
        # Zapisuje wszystkie zdjecia do folderu temporary, tymczasowe rozwiazanie
        for file in files:
            if file.filename:
                zip_file.writestr(f"temporary/{file.filename}", file.read())

    zip_buffer.seek(0) # ustawia wskaznik bufora na poczatek, aby poprawnie odczytac dane
    return send_file(zip_buffer, as_attachment=True, download_name="classified_images.zip") # Wysyla plik ZIP jako zalacznik o nazwie classified_images.zip


if __name__ == "__main__":
    app.run(debug=True)
