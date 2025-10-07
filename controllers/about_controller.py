from flask import request, render_template
import pandas as pd
import joblib
from sklearn.metrics.pairwise import euclidean_distances
import os

def load_models_and_data(app):
    """Load optional ML models and dataset; return (model, enc_artist, enc_mood, scaler, df)."""
    model = label_encoder_artist = label_encoder_mood = scaler = None
    df = pd.DataFrame()

    data_path = app.config.get('DATA_PATH')
    model_path = app.config.get('MODEL_PATH')
    artist_enc_path = app.config.get('ARTIST_ENCODER_PATH')
    mood_enc_path = app.config.get('MOOD_ENCODER_PATH')
    scaler_path = app.config.get('SCALER_PATH')

    # Load dataset if available
    try:
        if data_path and os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            df = pd.DataFrame(columns=[
                'name','album','artist','popularity','mood',
                'danceability','acousticness','energy','instrumentalness',
                'liveness','valence','loudness','speechiness','tempo','key','time_signature'
            ])
    except Exception:
        df = pd.DataFrame(columns=[
            'name','album','artist','popularity','mood',
            'danceability','acousticness','energy','instrumentalness',
            'liveness','valence','loudness','speechiness','tempo','key','time_signature'
        ])

    # Load model / encoders / scaler if present
    try:
        if model_path and os.path.exists(model_path):
            model = joblib.load(model_path)
    except Exception:
        model = None

    try:
        if artist_enc_path and os.path.exists(artist_enc_path):
            label_encoder_artist = joblib.load(artist_enc_path)
    except Exception:
        label_encoder_artist = None

    try:
        if mood_enc_path and os.path.exists(mood_enc_path):
            label_encoder_mood = joblib.load(mood_enc_path)
    except Exception:
        label_encoder_mood = None

    try:
        if scaler_path and os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
    except Exception:
        scaler = None

    return model, label_encoder_artist, label_encoder_mood, scaler, df

def get_songs_by_artist(artist_input, df):
    """Return songs for a given artist (safe if df empty)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=['name','album','artist','popularity','YouTube'])
    artist_songs = df[df['artist'] == artist_input].copy()
    artist_songs['YouTube'] = artist_songs['name'].apply(
        lambda song: f"<button onclick=\"window.open('https://www.youtube.com/results?search_query={song}', '_blank')\">YouTube</button>"
    )
    return artist_songs[['name','album','artist','popularity','YouTube']]

def recommend_songs_based_on_model(artist_input, mood_input, model, label_encoder_artist, label_encoder_mood, scaler, df):
    """Recommend songs using model if available; fallback: mood/popularity heuristics."""
    if df is None or df.empty:
        return pd.DataFrame(columns=['name','album','artist','popularity','YouTube'])

    try:
        if model is not None and label_encoder_artist is not None and label_encoder_mood is not None and scaler is not None:
            artist_encoded = label_encoder_artist.transform([artist_input])[0]
            mood_encoded = label_encoder_mood.transform([mood_input])[0]
            input_features = [[artist_encoded, mood_encoded]]
            predicted_features = model.predict(input_features)[0]
            predicted_features = scaler.inverse_transform([predicted_features])[0]
            predicted_df = pd.DataFrame([predicted_features], columns=[
                'danceability','acousticness','energy','instrumentalness','liveness','valence',
                'loudness','speechiness','tempo','key','time_signature'
            ])
            distances = euclidean_distances(
                df[['danceability','acousticness','energy','instrumentalness','liveness','valence',
                    'loudness','speechiness','tempo','key','time_signature']].fillna(0),
                predicted_df.fillna(0)
            ).flatten().tolist()
            df_copy = df.copy()
            df_copy['distance'] = distances
            sorted_df = df_copy.sort_values(by='distance').head(5)
        else:
            # Fallback: filter by mood then sort by popularity
            candidates = df.copy()
            if 'mood' in candidates.columns and mood_input:
                candidates = candidates[candidates['mood'] == mood_input]
            if candidates.empty:
                candidates = df.copy()
            if 'popularity' in candidates.columns:
                sorted_df = candidates.sort_values(by='popularity', ascending=False).head(5)
            else:
                sorted_df = candidates.head(5)
    except Exception:
        sorted_df = df.head(5) if not df.empty else pd.DataFrame(columns=['name','album','artist','popularity','YouTube'])

    sorted_df = sorted_df.copy()
    sorted_df['YouTube'] = sorted_df['name'].apply(
        lambda song: f"<button onclick=\"window.open('https://www.youtube.com/results?search_query={song}', '_blank')\">YouTube</button>"
    )
    return sorted_df[['name','album','artist','popularity','YouTube']]

def index_logic(request, app, emotion=None):
    """Handle the recommendation/search page; accepts emotion (optional)."""
    model, label_encoder_artist, label_encoder_mood, scaler, df = load_models_and_data(app)

    if df is None or df.empty:
        artists_list = []
        moods_list = []
    else:
        artists_list = sorted(df['artist'].dropna().unique().astype(str).tolist())
        moods_list = sorted(df['mood'].dropna().unique().astype(str).tolist()) if 'mood' in df.columns else []

    artist_choice = None
    mood_choice = emotion if emotion in moods_list else None
    artist_songs = None
    recommended_songs = None

    if request.method == 'POST':
        artist_choice = request.form.get('artist') or None
        mood_choice = request.form.get('mood') or mood_choice

    if artist_choice:
        artist_songs = get_songs_by_artist(artist_choice, df)

    if artist_choice and mood_choice:
        recommended_songs = recommend_songs_based_on_model(
            artist_choice, mood_choice, model, label_encoder_artist, label_encoder_mood, scaler, df
        )

    artist_songs_html = artist_songs.to_html(classes='table table-striped', index=False, escape=False) if artist_songs is not None and not artist_songs.empty else None
    recommended_songs_html = recommended_songs.to_html(classes='table table-striped', index=False, escape=False) if recommended_songs is not None and not recommended_songs.empty else None

    return render_template('module1.html',
                           artists=artists_list,
                           moods=moods_list,
                           artist_choice=artist_choice,
                           mood_choice=mood_choice,
                           artist_songs_html=artist_songs_html,
                           recommended_songs_html=recommended_songs_html,
                           emotion=emotion)

def about_page_logic(app, songs=None):
    """Render landing page with artists/moods list; safe if models/data missing."""
    _, _, _, _, df = load_models_and_data(app)
    if df is None or df.empty:
        artists_list = []
        moods_list = []
    else:
        artists_list = sorted(df['artist'].dropna().unique().astype(str).tolist())
        moods_list = sorted(df['mood'].dropna().unique().astype(str).tolist()) if 'mood' in df.columns else []

    return render_template('landing.html',
                           artists=artists_list,
                           moods=moods_list,
                           songs=songs or [])
