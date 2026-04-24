import os
import joblib
import pandas as pd

class StrokePredictorService:
    def __init__(self):
        # Setup Path
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.base_dir, 'model')
        
        # Pastikan file .joblib Anda masukkan ke dalam folder 'model'
        self.model_path = os.path.join(self.model_dir, 'random_forest_model.joblib')
        self.scaler_path = os.path.join(self.model_dir, 'scaler.joblib')

        # Load model ke memory
        self._load_models()

    def _load_models(self):
        """Memuat model dan scaler"""
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError("Model Random Forest atau Scaler tidak ditemukan di folder model/")
        
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

    def predict_stroke(self, data_dict: dict):
        """Method untuk melakukan prediksi"""
        
        # 1. Validasi gender (Hanya memastikan inputnya 0 atau 1)
        if data_dict['gender'] not in [0, 1]:
            raise ValueError("Invalid gender. Please use 0 or 1.")

        # 2. Convert dictionary ke DataFrame
        input_df = pd.DataFrame([data_dict])

        # 3. Pastikan urutan kolom sesuai dengan model training Anda
        expected_columns = [
            'gender', 
            'age', 
            'hypertension', 
            'heart_disease', 
            'ever_married', 
            'avg_glucose_level', 
            'bmi', 
            'smoking_status'
        ]
        
        try:
            input_df = input_df[expected_columns]
        except KeyError as e:
            raise ValueError(f"Missing required features for prediction: {e}")

        # 4. Scaling data
        input_df[['age', 'avg_glucose_level', 'bmi']] = self.scaler.transform(
            input_df[['age', 'avg_glucose_level', 'bmi']]
        )

        # 5. Prediksi
        prediction = self.model.predict(input_df)[0] # Hasil class (0 atau 1)
        prediction_proba = self.model.predict_proba(input_df)[:, 1] # Ambil peluang kelas 1 (Stroke)
        
        # Jadikan persentase (0.0 - 100.0)
        prob_value = float(prediction_proba[0])
        risk_percentage = round(prob_value * 100, 2)

        severity_score = int(round(risk_percentage))

        if severity_score > 60:
            status_label = "Risiko Tinggi Stroke"
        elif severity_score > 30:
            status_label = "Risiko Sedang"
        else:
            status_label = "Risiko Rendah"

        # 7. Kembalikan Response yang Lebih Informatif
        return {
            "severity_score": severity_score,
            "status_label": status_label,
            "metrics": {
                "predicted_class": int(prediction),
                "risk_percentage": risk_percentage,
                # "raw_probability": round(prob_value, 4)
            }
        }