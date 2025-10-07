from flask import Flask, request, send_from_directory, jsonify, url_for, render_template, session
from controllers.about_controller import about_page_logic, index_logic
from config import Config
import os

app = Flask(__name__)
app.config.from_object(Config)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'generated_music')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = os.urandom(24)

from controllers.generate import generate_music, get_status, download_file, model_status
import base64
from io import BytesIO
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

def get_songs_list():
    """Get list of all songs in the music directory."""
    music_dir = app.config.get('MUSIC_DIR', os.path.join('static', 'music'))
    songs = []
    if os.path.exists(music_dir):
        for file in os.listdir(music_dir):
            if file.endswith(('.mp3', '.wav', '.ogg')):
                song_path = url_for('static', filename=f'music/{file}')
                song_name = os.path.splitext(file)[0]
                songs.append({
                    'name': song_name,
                    'artist': 'Unknown',
                    'path': song_path,
                    'cover': url_for('static', filename='images/icon.jpg')
                })
    return songs

# Register routes
@app.route('/', methods=['GET', 'POST'])
def home():
    """Home page route with Spotify clone UI."""
    songs = get_songs_list()
    return about_page_logic(app, songs)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    """Recommendation endpoint."""
    # Get emotion from query parameter or session
    emotion = request.args.get('emotion') or session.get('emotion', 'neutral')
    # Pass emotion to your recommendation logic
    return index_logic(request, app, emotion=emotion)

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    """Music generation page and API endpoint."""
    # Get emotion from query parameter or session
    emotion = request.args.get('emotion') or session.get('emotion', 'neutral')
    if request.method == 'GET':
        return render_template('generate.html', emotion=emotion)
    # For POST requests (API), pass emotion to generate_music
    data = request.get_json() or {}
    if 'emotion' not in data and emotion:
        data['emotion'] = emotion
    return generate_music(data)

@app.route('/status/<generation_id>')
def status(generation_id):
    """Get generation status endpoint"""
    return get_status(generation_id)

@app.route('/download/<filename>')
def download(filename):
    """Download generated music file"""
    return download_file(filename)

@app.route('/api/songs')
def get_songs():
    """API endpoint to get list of songs."""
    songs = get_songs_list()
    return jsonify(songs)


# --------------------
# HuggingFace emotion predictor (lazy-loaded)
# --------------------
MODEL_NAME = "abhilash88/face-emotion-detection"
_hf_processor = None
_hf_model = None

def load_hf_model():
    """Lazily load the Hugging Face image processor and model."""
    global _hf_processor, _hf_model
    if _hf_processor is None or _hf_model is None:
        _hf_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        _hf_model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    return _hf_processor, _hf_model

def predict_emotion_pil(image: Image.Image):
    """Predict emotion label and confidence from a PIL Image.

    Returns (label, confidence)
    """
    # Define the emotion labels in the correct order
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    processor, model = load_hf_model()
    # Prepare inputs
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    top_idx = torch.argmax(probs, dim=-1).item()
    
    # Use the proper emotion label instead of label_0, label_1, etc.
    if top_idx < len(emotions):
        label = emotions[top_idx].lower()  # Convert to lowercase for consistency
    else:
        label = 'neutral'  # Fallback
    
    confidence = float(probs[0][top_idx].item())
    return label, confidence


@app.route('/api/predict_emotion', methods=['POST'])
def api_predict_emotion():
    """Accept a base64-encoded image (data URL or raw base64) and return predicted emotion.

    POST JSON: { "image": "data:image/jpeg;base64,..." }
    Response: { status: 'ok', emotion: 'happy', confidence: 0.87 }
    """
    data = request.get_json() or {}
    img_b64 = data.get('image')
    if not img_b64:
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400

    # Strip data URL prefix if present
    if isinstance(img_b64, str) and img_b64.startswith('data:'):
        try:
            img_b64 = img_b64.split(',', 1)[1]
        except Exception:
            pass

    try:
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Invalid image data: {e}'}), 400

    try:
        label, confidence = predict_emotion_pil(img)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Model error: {e}'}), 500

    # Save to session so other endpoints can reuse it
    session['emotion'] = label
    session['emotion_confidence'] = confidence

    return jsonify({'status': 'ok', 'emotion': label, 'confidence': confidence})

@app.route('/api/play', methods=['POST'])
def api_play_song():
    """
    API endpoint to play a song.
    Expects JSON: { "filename": "song.mp3" }
    Returns a direct URL to stream the song.
    """
    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'Filename is required'}), 400

    music_dir = app.config.get('MUSIC_DIR', os.path.join('static', 'music'))
    file_path = os.path.join(music_dir, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'Song not found'}), 404

    stream_url = url_for('play_song', filename=filename)
    return jsonify({'stream_url': stream_url})

@app.route('/library')
def library():
    """User's music library."""
    return about_page_logic(app)

@app.route('/playlists')
def playlists():
    """User's playlists."""
    return about_page_logic(app)

@app.route('/liked')
def liked_songs():
    """User's liked songs."""
    return about_page_logic(app)

@app.route('/play/<path:filename>')
def play_song(filename):
    """Stream a song from the music directory."""
    music_dir = app.config.get('MUSIC_DIR', os.path.join('static', 'music'))
    return send_from_directory(music_dir, filename)

# Routes for emotion detection
@app.route('/api/emotion', methods=['POST'])
def save_emotion():
    """Save detected emotion to session"""
    data = request.get_json()
    if not data or 'emotion' not in data:
        return jsonify({'status': 'error', 'message': 'Invalid request data'}), 400
    # Save emotion in session
    session['emotion'] = data['emotion']
    session['emotion_confidence'] = data.get('confidence', 0.0)
    session['emotion_timestamp'] = data.get('timestamp', '')
    return jsonify({'status': 'ok'})

@app.route('/model_status')
def check_model_status():
    """Check if the music generation model is loaded"""
    return model_status()

if __name__ == '__main__':
    app.run(debug=True)
