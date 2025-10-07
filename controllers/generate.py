from flask import render_template, request, jsonify, send_file, current_app
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import os
import tempfile
import base64
import io
import numpy as np
from datetime import datetime
import threading
import time
import uuid

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'generated_music')
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
MAX_DURATION = 30  # Maximum duration in seconds

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model
model = None
processor = None
is_loading = False
loading_status = "Initializing..."
generation_status = {}


def load_model():
    """Load the MusicGen model and processor"""
    global model, processor, is_loading, loading_status

    try:
        is_loading = True
        loading_status = "Loading MusicGen model... This may take several minutes on first run."
        print("Loading MusicGen model...")

        # Use the small model for faster loading and lower memory requirements
        model_name = "facebook/musicgen-small"

        # Load processor
        loading_status = "Loading audio processor..."
        processor = AutoProcessor.from_pretrained(model_name)

        # Load model
        loading_status = "Loading generation model..."
        model = MusicgenForConditionalGeneration.from_pretrained(model_name)

        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            loading_status = "Model loaded successfully on GPU!"
            print("Model loaded on GPU")
        else:
            loading_status = "Model loaded successfully on CPU!"
            print("Model loaded on CPU")

        is_loading = False
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        loading_status = f"Error loading model: {str(e)}"
        is_loading = False
        return False


def generate_audio(prompt, duration, guidance_scale, temperature, do_sample, generation_id):
    """Generate audio in a background thread"""
    global generation_status
    
    try:
        generation_status[generation_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting generation...'
        }
        
        # Process text input
        generation_status[generation_id]['progress'] = 10
        generation_status[generation_id]['message'] = 'Processing text input...'
        
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        )

        # Move inputs to GPU if model is on GPU
        if next(model.parameters()).is_cuda:
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Calculate max_new_tokens based on duration
        # MusicGen generates at ~50 tokens per second
        max_new_tokens = int(duration * 50)
        
        generation_status[generation_id]['progress'] = 30
        generation_status[generation_id]['message'] = 'Generating audio...'

        # Generate audio
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                guidance_scale=guidance_scale,
                temperature=temperature if do_sample else 1.0
            )
        
        generation_status[generation_id]['progress'] = 70
        generation_status[generation_id]['message'] = 'Processing audio...'

        # Get sampling rate
        sampling_rate = model.config.audio_encoder.sampling_rate

        # Convert to numpy array and ensure it's the right shape
        audio_array = audio_values[0, 0].cpu().numpy()

        # Normalize audio to prevent clipping
        audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Save to file
        generation_status[generation_id]['progress'] = 85
        generation_status[generation_id]['message'] = 'Saving audio file...'
        
        # Save audio file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{generation_id}.wav"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Ensure audio is in the right format for scipy
        audio_int16 = (audio_array * 32767).astype(np.int16)
        scipy.io.wavfile.write(filepath, sampling_rate, audio_int16)
        
        # Convert audio to base64 for web playback
        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, sampling_rate, audio_int16)
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode()
        
        generation_status[generation_id] = {
            'status': 'completed',
            'progress': 100,
            'message': 'Music generated successfully!',
            'filename': filename,
            'audio_base64': audio_base64,
            'duration': len(audio_array) / sampling_rate,
            'sampling_rate': sampling_rate
        }
        
    except Exception as e:
        print(f"Error generating music: {e}")
        generation_status[generation_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}'
        }


def generate_music():
    """API endpoint for music generation"""
    try:
        global model, processor, is_loading
        
        data = request.get_json()
        
        # Check if model is loaded
        if model is None or processor is None:
            if is_loading:
                return jsonify({'error': 'Model is still loading. Please wait...'}), 503
            else:
                return jsonify({'error': 'Model failed to load. Please refresh the page.'}), 500
                
        prompt = data.get('prompt', '').strip()
        duration = min(float(data.get('duration', 15)), MAX_DURATION)
        guidance_scale = float(data.get('guidance_scale', 3.0))
        temperature = float(data.get('temperature', 0.8))
        do_sample = bool(data.get('do_sample', True))
        
        if not prompt:
            return jsonify({'error': 'Please provide a valid prompt'}), 400
            
        # Generate unique ID for this generation
        generation_id = str(uuid.uuid4())
        
        # Start generation in background thread
        generation_thread = threading.Thread(
            target=generate_audio,
            args=(prompt, duration, guidance_scale, temperature, do_sample, generation_id)
        )
        generation_thread.daemon = True
        generation_thread.start()
        
        return jsonify({'generation_id': generation_id})
        
    except Exception as e:
        print(f"Error in generate endpoint: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


def get_status(generation_id):
    """Get status of music generation"""
    status_data = generation_status.get(generation_id, {
        'status': 'not_found',
        'progress': 0,
        'message': 'Generation not found'
    })
    return jsonify(status_data)


def download_file(filename):
    """Download generated audio file"""
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def model_status():
    """Check if model is loaded"""
    return jsonify({
        'loaded': model is not None and processor is not None,
        'is_loading': is_loading,
        'status': loading_status
    })


# Start loading model in background thread when module is imported
loading_thread = threading.Thread(target=load_model, daemon=True)
loading_thread.start()