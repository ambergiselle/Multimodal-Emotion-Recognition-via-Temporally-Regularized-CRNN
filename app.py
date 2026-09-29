"""Flask web app for multimodal (text + speech) emotion recognition."""

import os
import pickle
import tempfile

import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from emotion import EMOTION_LABELS
from emotion.speech_model import load_speech_model, predict_speech
from emotion.text_model import CUSTOM_OBJECTS, get_emotion_vector, preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MAX_LEN = 100
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# ------------------------------------------------------------------
# Load models once at start-up
# ------------------------------------------------------------------
text_model = load_model(
    os.path.join(MODEL_DIR, "combined_final.keras"),
    custom_objects=CUSTOM_OBJECTS,
)

with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

with open(os.path.join(MODEL_DIR, "emoji_rules.pkl"), "rb") as f:
    emoji_rules = pickle.load(f)

speech_model = load_speech_model(os.path.join(MODEL_DIR, "TR_CRNN.pth"))


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    data = request.get_json(silent=True) or {}
    raw_text = (data.get("text") or "").strip()
    if not raw_text:
        return jsonify({"error": "Please enter some text."}), 400

    # Stage 1: rule-based emoji detection takes priority
    for char in raw_text:
        if char in emoji_rules:
            emoji_count = sum(1 for c in raw_text if c in emoji_rules)
            confidence = min(60 + emoji_count * 10, 90)
            return jsonify({"emotion": emoji_rules[char], "confidence": confidence})

    # Stage 2: Relative EA-LSTM on the cleaned text
    text = preprocess(raw_text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="pre", truncating="post")
    emotion_vec = get_emotion_vector(text).reshape(1, 7)

    preds = text_model.predict([padded, emotion_vec], verbose=0)[0]
    emotion_id = int(np.argmax(preds))

    return jsonify({
        "emotion": EMOTION_LABELS[emotion_id],
        "confidence": round(float(preds[emotion_id]) * 100, 2),
    })


@app.route("/analyze_speech", methods=["POST"])
def analyze_speech():
    audio_file = request.files.get("audio")
    if audio_file is None or audio_file.filename == "":
        return jsonify({"error": "Please upload or record an audio file."}), 400

    ext = os.path.splitext(audio_file.filename)[1].lower() or ".wav"
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        return jsonify({"error": f"Unsupported audio format: {ext}"}), 400

    # Save to a temporary file and always delete it afterwards
    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    try:
        audio_file.save(tmp_path)
        emotion_id, confidence = predict_speech(speech_model, tmp_path)
    except Exception:
        app.logger.exception("Speech analysis failed")
        return jsonify({"error": "Could not process this audio file."}), 400
    finally:
        os.remove(tmp_path)

    return jsonify({
        "emotion": EMOTION_LABELS[emotion_id],
        "confidence": round(confidence * 100, 2),
    })


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug)
