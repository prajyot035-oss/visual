import os

class Config:
    # Security
    SECRET_KEY = 'your-secret-key-here'  # Change this in production
    
    # Application paths
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
    STATIC_DIR = os.path.join(BASE_DIR, 'static')
    
    # Model and data paths
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    MOOD_ENCODER_PATH = os.path.join(MODELS_DIR, 'mood_encoder.pkl')
    ARTIST_ENCODER_PATH = os.path.join(MODELS_DIR, 'artist_encoder.pkl')
    SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
    MODEL_PATH = os.path.join(MODELS_DIR, 'optimized_random_forest_model.pkl')
    LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'sentiment_analysis_lstm_model.keras')
    DATA_PATH = os.path.join(BASE_DIR, 'data_moods.csv')
    
    # Media paths
    MUSIC_DIR = os.path.join(BASE_DIR, 'static', 'music')
    IMAGES_DIR = os.path.join(BASE_DIR, 'static', 'images')
    # Flask configuration
    DEBUG = True  # Set to False in production
    TEMPLATES_AUTO_RELOAD = True
    
    # Model feature columns
    FEATURE_COLUMNS = [
        'danceability', 'acousticness', 'energy',
        'instrumentalness', 'liveness', 'valence',
        'loudness', 'speechiness', 'tempo', 'key',
        'time_signature'
    ]
