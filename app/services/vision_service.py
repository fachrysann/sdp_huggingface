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
    # ENDPOINT 2: EYE SYMMETRY (IMPROVED)
    # ==========================================
    def get_gaze_ratio(self, iris_pt, corner_medial, corner_lateral, w, h):
        """
        Hitung posisi iris relatif dari sudut MEDIAL ke LATERAL.
        Return: 0.0 = iris di sudut medial (melirik kearah luar/temporal),
                1.0 = iris di sudut lateral (melirik ke dalam/nasal)
        
        Untuk mata kiri (patient view): medial=33, lateral=133
        Untuk mata kanan (patient view): medial=263, lateral=362
        Dengan cara ini, keduanya menggunakan konvensi yang sama:
        melirik kanan → gaze_L tinggi, gaze_R tinggi (sinkron)
        """
        ix  = iris_pt.x * w
        iy  = iris_pt.y * h
        cmx = corner_medial.x * w
        cmy = corner_medial.y * h
        clx = corner_lateral.x * w
        cly = corner_lateral.y * h

        dist_total = math.sqrt((clx - cmx)**2 + (cly - cmy)**2)
        if dist_total < 1e-6:
            return 0.5

        # Proyeksi iris ke axis medial→lateral (robust terhadap kepala sedikit miring)
        axis_x = (clx - cmx) / dist_total
        axis_y = (cly - cmy) / dist_total
        proj   = (ix - cmx) * axis_x + (iy - cmy) * axis_y

        return max(0.0, min(1.0, proj / dist_total))

    def analyze_eye_symmetry(self, image: np.ndarray):
        """Method Public untuk Endpoint /analyze/eye-symmetry"""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return None, None

        landmarks = detection_result.face_landmarks[0]
        results   = self._calculate_eye_symmetry(landmarks, image)
        return results, image

    def _calculate_eye_symmetry(self, landmarks, image):
        """
        Logika baru:
        - gaze_L: ratio iris kiri dari sudut medial (33) ke lateral (133)
        - gaze_R: ratio iris kanan dari sudut medial (263) ke lateral (362)
        - Keduanya menggunakan arah MEDIAL→LATERAL yang sama sebagai referensi,
        sehingga melirik ke arah yang sama = kedua nilai naik/turun bersamaan.
        - Sinkronisasi: |gaze_L - gaze_R| mendekati 0 = simetris
        """
        h, w, _ = image.shape

        # ---- Landmark ----
        l_iris     = landmarks[468]   # iris kiri
        l_medial   = landmarks[33]    # sudut medial mata kiri   (dekat hidung)
        l_lateral  = landmarks[133]   # sudut lateral mata kiri  (dekat telinga)

        r_iris     = landmarks[473]   # iris kanan
        r_medial   = landmarks[263]   # sudut medial mata kanan  (dekat hidung)
        r_lateral  = landmarks[362]   # sudut lateral mata kanan (dekat telinga)

        # ---- Gaze ratio (skala 0–1, konvensi SAMA untuk kedua mata) ----
        gaze_L = self.get_gaze_ratio(l_iris, l_medial, l_lateral, w, h)
        gaze_R = self.get_gaze_ratio(r_iris, r_medial, r_lateral, w, h)

        # ---- Sinkronisasi ----
        gaze_diff = abs(gaze_L - gaze_R)

        # Threshold kalibrasi:
        #   < 0.07  → normal (variasi natural & mikro-gerak)
        #   0.07–0.18 → mild asymmetry
        #   > 0.18   → significant asymmetry
        THRESH_NORMAL = 0.04   # sebelumnya 0.07 → zona "aman" diperkecil
        THRESH_MILD   = 0.10   # sebelumnya 0.18 → asimetri ringan lebih cepat terdeteksi

        if gaze_diff <= THRESH_NORMAL:
            score  = int(100 - (gaze_diff / THRESH_NORMAL) * 20)  # 80–100
            status = "Simetris"
            is_sym = True
            color  = (0, 220, 80)
        elif gaze_diff <= THRESH_MILD:
            t      = (gaze_diff - THRESH_NORMAL) / (THRESH_MILD - THRESH_NORMAL)
            score  = int(80 - t * 50)                              # 30–80
            status = "Asimetri Ringan"
            is_sym = False
            color  = (0, 165, 255)
        else:
            t      = min(1.0, (gaze_diff - THRESH_MILD) / 0.08)
            score  = int(30 - t * 30)                              # 0–30
            status = "Asimetri Signifikan"
            is_sym = False
            color  = (0, 0, 255)

        # ---- Deteksi arah lirikan ----
        CENTER = 0.5
        DEAD_ZONE = 0.12  # zona tengah = dianggap lurus

        avg_gaze = (gaze_L + gaze_R) / 2
        if avg_gaze > CENTER + DEAD_ZONE:
            gaze_dir = "Melirik Kanan"
        elif avg_gaze < CENTER - DEAD_ZONE:
            gaze_dir = "Melirik Kiri"
        else:
            gaze_dir = "Lurus / Tengah"

        # ==================== VISUALISASI ====================
        def lm_px(idx):
            return int(landmarks[idx].x * w), int(landmarks[idx].y * h)

        # Titik-titik iris & sudut
        for idx in [468, 473]:
            cx, cy = lm_px(idx)
            cv2.circle(image, (cx, cy), 5, (0, 255, 255), -1)
            cv2.circle(image, (cx, cy), 5, (0, 0, 0), 1)

        for idx in [33, 133, 263, 362]:
            cx, cy = lm_px(idx)
            cv2.circle(image, (cx, cy), 4, (255, 200, 0), -1)

        # Garis axis gaze (medial → lateral) tiap mata
        cv2.arrowedLine(image, lm_px(33),  lm_px(133), (180, 180, 0), 1, tipLength=0.15)
        cv2.arrowedLine(image, lm_px(263), lm_px(362), (180, 180, 0), 1, tipLength=0.15)

        # Dashboard
        dash_w = min(w, 460)
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (dash_w, 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        cv2.putText(image, f"Score: {score}%  |  {status}",
                    (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(image, f"Gaze L (med->lat): {gaze_L:.3f}",
                    (15, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, f"Gaze R (med->lat): {gaze_R:.3f}",
                    (15, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, f"Diff: {gaze_diff:.3f}  (threshold normal: <{THRESH_NORMAL})",
                    (15, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

        # Progress bar sinkronisasi
        bar_x, bar_y, bar_w_px = 15, 150, 200
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w_px, bar_y + 12), (80, 80, 80), -1)
        fill = int(score / 100 * bar_w_px)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill, bar_y + 12), color, -1)
        cv2.putText(image, "sync", (bar_x + bar_w_px + 8, bar_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        return {
            "symmetry_score":  score,
            "is_symmetrical":  bool(is_sym),
            "gaze_left":       round(gaze_L, 3),
            "gaze_right":      round(gaze_R, 3),
            "gaze_difference": round(gaze_diff, 3),
            "status":          status,
        }