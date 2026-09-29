"""Speech emotion recognition: CRNN trained with a temporal regularisation loss.

The CRNN produces an emotion prediction for every time step (frame) of a
log-Mel spectrogram and averages them into one utterance-level prediction.
During training, `temporal_regularization_loss` penalises large jumps between
adjacent frame predictions:

    L_total = L_cross_entropy + lambda * L_temp      (lambda = 0.1)
"""

import librosa
import numpy as np
import torch
import torch.nn as nn

# Feature-extraction settings (must match training)
SAMPLE_RATE = 22050
DURATION = 2.5      # seconds of audio analysed
OFFSET = 0.6        # skip leading silence
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_FRAMES = 108      # fixed time dimension fed to the CNN


def temporal_regularization_loss(time_step_logits):
    """Mean squared difference between predictions at adjacent time steps.

    Args:
        time_step_logits: tensor of shape (batch, time, classes).
    """
    diff = time_step_logits[:, 1:, :] - time_step_logits[:, :-1, :]
    return torch.mean(diff ** 2)


class CRNN(nn.Module):
    """CNN feature extractor -> LSTM over time -> frame-wise classifier."""

    def __init__(self, num_classes=7):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
        )

        self.time_steps = 25
        self.base_feat_dim = 64 * 30

        self.lstm = nn.LSTM(
            input_size=self.base_feat_dim,
            hidden_size=128,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, 1, 128 mel bins, 108 frames)
        x = self.cnn(x)                               # (B, 64, 30, 25)

        # Treat the CNN width axis as time for the LSTM
        x = x.permute(0, 3, 1, 2)                     # (B, 25, 64, 30)
        x = x.reshape(x.size(0), x.size(1), -1)       # (B, 25, 1920)

        lstm_out, _ = self.lstm(x)

        time_logits = self.classifier(lstm_out)       # frame-level predictions
        utterance_logits = time_logits.mean(dim=1)    # utterance-level prediction

        return time_logits, utterance_logits


def extract_log_mel(audio_path):
    """Load an audio file and return a normalised (1, 1, 128, 108) tensor."""
    y, sr = librosa.load(
        audio_path, sr=SAMPLE_RATE, duration=DURATION, offset=OFFSET, mono=True
    )

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)

    # Pad or trim to a fixed number of frames
    if log_mel.shape[1] < N_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, N_FRAMES - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :N_FRAMES]

    return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def load_speech_model(weights_path):
    model = CRNN()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def predict_speech(model, audio_path):
    """Return (class_index, confidence) for an audio file."""
    x = extract_log_mel(audio_path)
    with torch.no_grad():
        _, logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    emotion_id = int(np.argmax(probs))
    return emotion_id, float(probs[emotion_id])
