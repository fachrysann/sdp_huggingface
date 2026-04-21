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
    def _calculate_eye_symmetry(self, landmarks, image):
        """Logika kalkulasi dan visualisasi Lirikan mata menggunakan MediaPipe Iris"""
        h, w, _ = image.shape

        # Titik Sudut Mata (Corners)
        l_c_l, l_c_r = landmarks[33], landmarks[133]
        r_c_l, r_c_r = landmarks[362], landmarks[263]

        # Titik Kontur Tepi MediaPipe Iris
        l_iris_contour = [469, 470, 471, 472]
        r_iris_contour = [474, 475, 476, 477]

        # Dapatkan center dan radius yang presisi dari kontur
        l_iris_center, l_iris_radius = self.get_iris_center_and_radius(landmarks, l_iris_contour, w, h)
        r_iris_center, r_iris_radius = self.get_iris_center_and_radius(landmarks, r_iris_contour, w, h)

        # Hitung Rasio Lirikan
        l_ratio = self.get_iris_ratio(l_iris_center, l_c_l, l_c_r, w, h)
        r_ratio = self.get_iris_ratio(r_iris_center, r_c_l, r_c_r, w, h)

        # -------------------------------------------------------------------------
        # UPGRADE: HEAD POSE COMPENSATION via facial midline
        # Nose tip (1) dan midpoint antara kedua eye corners -> estimasi yaw bias
        # Jika kepala miring ke kanan, l_ratio naik dan r_ratio turun secara bersamaan.
        # Kita koreksi dengan menghitung bias rata-rata lalu shift kedua rasio.
        # -------------------------------------------------------------------------
        nose_tip = landmarks[1]
        nose_x = nose_tip.x * w

        # Midpoint horizontal antara inner corners kedua mata (landmark 133 & 362)
        inner_l_x = landmarks[133].x * w
        inner_r_x = landmarks[362].x * w
        eye_midpoint_x = (inner_l_x + inner_r_x) / 2.0

        # Lebar wajah estimasi (jarak outer corners)
        face_width = abs(landmarks[263].x * w - landmarks[33].x * w)
        if face_width > 0:
            # Seberapa jauh hidung bergeser dari garis tengah mata (normalized -0.5..0.5)
            yaw_bias = (nose_x - eye_midpoint_x) / face_width
        else:
            yaw_bias = 0.0

        # Kompensasi: geser kedua rasio berlawanan arah bias yaw
        # Koefisien 0.5 didapat dari observasi bahwa 1 unit bias ~ 0.5 unit ratio shift
        COMPENSATION_COEFF = 0.5
        l_ratio_compensated = np.clip(l_ratio - yaw_bias * COMPENSATION_COEFF, 0.0, 1.0)
        r_ratio_compensated = np.clip(r_ratio + yaw_bias * COMPENSATION_COEFF, 0.0, 1.0)

        # -------------------------------------------------------------------------
        # UPGRADE: GAZE CONVERGENCE CHECK
        # Untuk pandangan lurus/simetris, kedua rasio harus mendekati 0.5.
        # Untuk pandangan menyerong tapi simetris (keduanya lirik kiri misal),
        # selisih tetap kecil tapi keduanya sama-sama jauh dari 0.5.
        # Kita pakai dua komponen:
        #   1. ratio_diff         -> apakah kedua mata bergerak BERSAMA (konsistensi)
        #   2. convergence_error  -> apakah arah lirikan MASUK AKAL secara binokuler
        #      (mata kiri & kanan seharusnya mirror satu sama lain di sumbu wajah)
        # -------------------------------------------------------------------------
        ratio_diff = abs(l_ratio_compensated - r_ratio_compensated)

        # Mirror check: untuk simetri sejati, (l_ratio) + (1 - r_ratio) harus ~ 1.0
        # Contoh: keduanya lirik kiri -> l=0.3, r=0.3
        #   mirror_sum = 0.3 + (1-0.3) = 1.0  ✓ simetris
        # Keduanya lirik berlainan -> l=0.3, r=0.7
        #   mirror_sum = 0.3 + (1-0.7) = 0.6  ✗ asimetris
        mirror_sum = l_ratio_compensated + (1.0 - r_ratio_compensated)
        convergence_error = abs(mirror_sum - 1.0)  # 0.0 = perfect, makin besar makin asimetris

        # -------------------------------------------------------------------------
        # UPGRADE: WEIGHTED COMPOSITE SCORE
        # ratio_diff      -> bobot 60% (perbedaan langsung antar mata)
        # convergence_err -> bobot 40% (validasi arah binokuler)
        # Kedua komponen dinormalisasi ke skala yang sama (max_error = 0.5)
        # -------------------------------------------------------------------------
        MAX_ERROR = 0.5  # nilai maksimal teoritis untuk normalisasi
        normalized_diff = min(ratio_diff / MAX_ERROR, 1.0)
        normalized_conv = min(convergence_error / MAX_ERROR, 1.0)

        composite_error = (0.6 * normalized_diff) + (0.4 * normalized_conv)
        score = int(round(max(0.0, min(100.0, (1.0 - composite_error) * 100))))

        # -------------------------------------------------------------------------
        # UPGRADE: ADAPTIVE THRESHOLD untuk is_symmetrical
        # Threshold tidak lagi fixed 0.10 — melainkan proporsional terhadap
        # seberapa ekstrem lirikannya. Lirikan ekstrem (jauh dari center)
        # secara alami lebih sulit presisi, jadi toleransinya sedikit lebih longgar.
        # -------------------------------------------------------------------------
        avg_deviation = (abs(l_ratio_compensated - 0.5) + abs(r_ratio_compensated - 0.5)) / 2.0
        adaptive_threshold = 0.06 + (avg_deviation * 0.10)  # 0.06 (center) ~ 0.11 (extreme)
        is_symmetrical = (ratio_diff <= adaptive_threshold) and (convergence_error <= adaptive_threshold * 1.5)

        status_text = "Arah Lirikan Sama (Simetris)" if is_symmetrical else "Arah Lirikan Berbeda (Asimetris)"
        color = (0, 255, 0) if is_symmetrical else (0, 0, 255)

        # --- VISUALISASI MATA --- (tidak berubah)
        cv2.circle(image, (int(l_iris_center[0]), int(l_iris_center[1])), int(l_iris_radius), (0, 255, 255), 1)
        cv2.circle(image, (int(r_iris_center[0]), int(r_iris_center[1])), int(r_iris_radius), (0, 255, 255), 1)
        cv2.circle(image, (int(l_iris_center[0]), int(l_iris_center[1])), 2, (0, 0, 255), -1)
        cv2.circle(image, (int(r_iris_center[0]), int(r_iris_center[1])), 2, (0, 0, 255), -1)

        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (max(400, int(w * 0.6)), 175), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        cv2.putText(image, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(image, f"Symmetry Score: {score}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Diff: {ratio_diff:.3f}  Conv Err: {convergence_error:.3f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(image, f"Yaw Bias: {yaw_bias:+.3f}  Threshold: {adaptive_threshold:.3f}", (20, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        bar_x, bar_y, bar_w = 20, 150, 200
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (100, 100, 100), -1)
        l_pos = int(bar_x + (l_ratio_compensated * bar_w))
        r_pos = int(bar_x + (r_ratio_compensated * bar_w))
        cv2.circle(image, (l_pos, bar_y + 5), 6, (255, 0, 0), -1)
        cv2.circle(image, (r_pos, bar_y + 5), 6, (0, 0, 255), -1)

        return {
            "symmetry_score": score,
            "is_symmetrical": bool(is_symmetrical),
            "left_eye_ratio": round(l_ratio_compensated, 3),
            "right_eye_ratio": round(r_ratio_compensated, 3),
            "ratio_difference": round(ratio_diff, 3),
            "convergence_error": round(convergence_error, 3),
            "yaw_bias": round(yaw_bias, 3),
            "status": status_text
        }