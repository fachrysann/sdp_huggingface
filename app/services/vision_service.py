import cv2
import mediapipe as mp
import numpy as np
import math
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceAnalyzerService:
    def __init__(self):
        # 1. Setup Path saat Class dipanggil
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.base_dir, 'model')
        self.model_path = os.path.join(self.model_dir, 'face_landmarker.task')

        # 2. Pastikan model ada, lalu load ke memory
        self._ensure_model_exists()
        self.landmarker = self._initialize_model()

    def _ensure_model_exists(self):
        """Method private untuk download model jika belum ada"""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        if not os.path.exists(self.model_path):
            print("Downloading MediaPipe model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)

    def _initialize_model(self):
        """Method private untuk inisialisasi MediaPipe"""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE,

            min_face_detection_confidence=0.2, 
            min_face_presence_confidence=0.2,
            min_tracking_confidence=0.2
        )
        return vision.FaceLandmarker.create_from_options(options)

    # ==========================================
    # ENDPOINT 1: FACIAL PALSY 
    # ==========================================
    def analyze_facial_palsy(self, image: np.ndarray):
        """Method Public yang akan dipanggil oleh API (main.py/routes.py)"""
        # Convert ke MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        # Deteksi Wajah
        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return None, None # Kembalikan None jika tidak ada wajah

        # Jalankan logika perhitungan & gambar
        landmarks = detection_result.face_landmarks[0]
        results = self._calculate_and_draw(landmarks, image)
        
        return results, image

    def _calculate_and_draw(self, landmarks, image):
        h, w, _ = image.shape
        
        def get_pt(i):
            return int(landmarks[i].x * w), int(landmarks[i].y * h)

        # 1. MATHEMATICAL LOGIC
        # Points: B=6, T=1, R=61, L=291
        B, T, R, L = get_pt(6), get_pt(1), get_pt(61), get_pt(291)

        # Face Scale
        p33, p263 = landmarks[33], landmarks[263]
        face_scale = math.sqrt((p33.x - p263.x)**2 + (p33.y - p263.y)**2)

        def calc_angle(p_target, p_vertex, p_base):
            v1 = np.array([p_target[0] - p_vertex[0], p_target[1] - p_vertex[1]])
            v2 = np.array([p_base[0] - p_vertex[0], p_base[1] - p_vertex[1]])
            dot = np.dot(v1, v2)
            mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if mag1 == 0 or mag2 == 0: return 0
            return math.degrees(math.acos(np.clip(dot / (mag1 * mag2), -1.0, 1.0)))

        # Perhitungan Asimetri
        angle_right = calc_angle(B, R, T)
        angle_left = calc_angle(B, L, T)
        mouth_diff = abs(angle_right - angle_left)

        left_eye_open = landmarks[159].y - landmarks[145].y
        right_eye_open = landmarks[386].y - landmarks[374].y
        eye_asym = abs(left_eye_open - right_eye_open) / face_scale * 10

        # Scoring
        m_pct = (mouth_diff / 10.0) * 100 if mouth_diff > 2.5 else 0
        e_pct = (eye_asym / 0.5) * 100 if eye_asym > 0.08 else 0
        final_pct = round(min(100, max(m_pct, e_pct)))

        # Severity Logic
        if final_pct > 45:
            severity, desc, color = "High Severity Detected", "Significant asymmetry. Consult doctor.", (0, 0, 255)
        elif final_pct > 15:
            severity, desc, color = "Mild Asymmetry", "Slight deviation. Monitor.", (0, 165, 255)
        else:
            severity, desc, color = "Within Normal Limits", "No significant symptoms.", (0, 255, 0)

        # --- VISUALIZATION ---
        
        # A. Garis Mulut (Merah)
        cv2.polylines(image, [np.array([B, R, T])], False, (0, 0, 255), 2)
        cv2.polylines(image, [np.array([B, L, T])], False, (0, 0, 255), 2)

        # B. Gambar Mata (Sian & Magenta)
        def draw_eye(indices):
            pts = np.array([get_pt(i) for i in indices], np.int32)
            cv2.polylines(image, [pts], False, (255, 255, 0), 2) # Cyan
            for p in pts:
                cv2.circle(image, p, 3, (255, 0, 255), -1) # Magenta

        draw_eye([130, 161, 160, 159, 158, 157, 133]) # Kiri
        draw_eye([362, 384, 385, 386, 387, 388, 263]) # Kanan

        # C. Label B, T, R, L
        for pt, label in [(B, "B"), (T, "T"), (R, "R"), (L, "L")]:
            cv2.circle(image, pt, 6, (0, 255, 255), -1) # Kuning
            cv2.circle(image, pt, 6, (0, 0, 0), 2)       # Border hitam
            cv2.putText(image, label, (pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # D. Dashboard
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (400, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        # E. Teks Dashboard
        cv2.putText(image, f"Score: {final_pct}%", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(image, severity, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(image, f"Mouth Diff: {mouth_diff:.1f} deg", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, f"Eye Asym : {eye_asym:.3f}", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, desc, (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return {
            "percentage": final_pct,
            "severity": severity,
            "is_stroke_detected": final_pct > 35,
            "mouth_diff": round(mouth_diff, 2),
            "eye_asymmetry": round(eye_asym, 4)
        }
    
    # ==========================================
    # ENDPOINT 2: EYE SYMMETRY
    # ==========================================
    def get_iris_ratio(self, iris_pt, corner_left, corner_right, w, h):
        """Helper untuk menghitung rasio posisi iris dengan akurat (menggunakan pixel)"""
        ix, iy = iris_pt.x * w, iris_pt.y * h
        clx, cly = corner_left.x * w, corner_left.y * h
        crx, cry = corner_right.x * w, corner_right.y * h
        
        dist_total = math.sqrt((crx - clx)**2 + (cry - cly)**2)
        dist_iris = math.sqrt((ix - clx)**2 + (iy - cly)**2)
        if dist_total == 0: return 0.5
        return dist_iris / dist_total

    def analyze_eye_symmetry(self, image: np.ndarray):
        """Method Public untuk Endpoint /analyze/eye-symmetry"""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return None, None

        landmarks = detection_result.face_landmarks[0]
        results = self._calculate_eye_symmetry(landmarks, image)
        return results, image

    def _calculate_eye_symmetry(self, landmarks, image):
        """Logika kalkulasi dan visualisasi Lirikan mata"""
        h, w, _ = image.shape
        
        # Titik Landmark Mata (Sesuai kode asli Anda)
        l_iris, l_c_l, l_c_r = landmarks[468], landmarks[33], landmarks[133]
        r_iris, r_c_l, r_c_r = landmarks[473], landmarks[362], landmarks[263]

        # Hitung Rasio
        l_ratio = self.get_iris_ratio(l_iris, l_c_l, l_c_r, w, h)
        r_ratio = self.get_iris_ratio(r_iris, r_c_l, r_c_r, w, h)
        
        # Analisis Asimetri
        ratio_diff = abs(l_ratio - r_ratio)
        is_symmetrical = ratio_diff <= 0.10

        # Scoring Logic (0 diff = 100%, 0.15 diff = 50%, >0.30 diff = 0%)
        # Rumus ini diadaptasi agar menghasilkan nilai persen yang masuk akal
        score = max(0, min(100, int(100 - (ratio_diff / 0.15 * 50))))
        
        status_text = "Arah Lirikan Sama (Simetris)" if is_symmetrical else "Arah Lirikan Berbeda (Asimetris)"
        color = (0, 255, 0) if is_symmetrical else (0, 0, 255)

        # --- VISUALISASI MATA ---
        # Gambar titik kuning di pupil
        for pt in [l_iris, r_iris]:
            cv2.circle(image, (int(pt.x * w), int(pt.y * h)), 4, (0, 255, 255), -1)

        # Dashboard UI
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (max(400, int(w*0.6)), 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        cv2.putText(image, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(image, f"Symmetry Score: {score}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Diff: {ratio_diff:.3f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Visual Slider Koordinasi (Bar progres)
        bar_x, bar_y, bar_w = 20, 130, 200
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (100, 100, 100), -1)
        
        l_pos = int(bar_x + (l_ratio * bar_w))
        r_pos = int(bar_x + (r_ratio * bar_w))
        
        cv2.circle(image, (l_pos, bar_y + 5), 6, (255, 0, 0), -1) # Biru (Mata Kiri)
        cv2.circle(image, (r_pos, bar_y + 5), 6, (0, 0, 255), -1) # Merah (Mata Kanan)

        return {
            "symmetry_score": score,
            "is_symmetrical": bool(is_symmetrical),
            "left_eye_ratio": round(l_ratio, 3),
            "right_eye_ratio": round(r_ratio, 3),
            "ratio_difference": round(ratio_diff, 3),
            "status": status_text
        }
