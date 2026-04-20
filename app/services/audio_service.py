import os
import io
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import soundfile as sf

class AudioAnalyzerService:
    def __init__(self):
        # Setup Path
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.base_dir, 'model')
        self.model_path = os.path.join(self.model_dir, 'model_scripted_resnet18_cpu.pt')

        self.CLASSES =[
            "mali-control", "mali-dysarthria", "no-control", "no-dysarthria",
            "salva-control", "salva-dysarthria", "sei-control", "sei-dysarthria",
            "si-control", "si-dysarthria", "stop-control", "stop-dysarthria",
            "uno-control", "uno-dysarthria", "zero-control", "zero-dysarthria"
        ]
        self.SAMPLE_RATE = 16000
        self.MAX_TIME_STEPS = 192

        # Inisialisasi Transforms
        self.mel_spectrogram = T.MelSpectrogram(sample_rate=self.SAMPLE_RATE, n_fft=512, hop_length=160, n_mels=80)
        self.amplitude_to_db = T.AmplitudeToDB()

        self._load_model()

    def _load_model(self):
        """Memuat model TorchScript"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model Audio tidak ditemukan di {self.model_path}")
        
        try:
            self.model = torch.jit.load(self.model_path)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Gagal memuat model audio: {e}")

    def process_audio(self, file_bytes: bytes) -> torch.Tensor:
        """Helper untuk preprocessing audio ke Mel-Spectrogram"""
        audio_data, sr = sf.read(io.BytesIO(file_bytes))
        waveform = torch.tensor(audio_data, dtype=torch.float32)
        
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0) 
        else:
            waveform = waveform.transpose(0, 1) 
            waveform = torch.mean(waveform, dim=0, keepdim=True) 

        if sr != self.SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sr, new_freq=self.SAMPLE_RATE)
            waveform = resampler(waveform)

        if waveform.shape[1] < self.mel_spectrogram.n_fft:
            pad_amount = self.mel_spectrogram.n_fft - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_amount), mode='constant', value=0.0)

        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

        time_steps = mel_spec.shape[2]
        if time_steps < self.MAX_TIME_STEPS:
            pad_amount = self.MAX_TIME_STEPS - time_steps
            mel_spec = F.pad(mel_spec, (0, pad_amount), value=mel_spec.min().item())
        else:
            mel_spec = mel_spec[:, :, :self.MAX_TIME_STEPS]

        return mel_spec.unsqueeze(0)

    def predict_audio(self, file_bytes: bytes):
        """Method utama untuk prediksi"""
        input_tensor = self.process_audio(file_bytes)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
            predicted_class = self.CLASSES[predicted_idx]
            
            all_probs = {self.CLASSES[i]: round(float(probabilities[i]) * 100, 2) for i in range(len(self.CLASSES))}
            word, speaker_type = predicted_class.split('-')

            return {
                "class": predicted_class,
                "word": word,
                "speaker_type": speaker_type,
                "confidence_percentage": round(confidence * 100, 2),
                "all_probabilities": all_probs
            }