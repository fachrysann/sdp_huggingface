import cv2
import mediapipe as mp
import numpy as np
import math
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mtcnn import MTCNN

class FaceAnalyzerService:
    def __init__(self):
        # 1. Setup Path saat Class dipanggil
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.base_dir, 'model')
        self.model_path = os.path.join(self.model_dir, 'face_landmarker.task')

        # 2. Pastikan model ada, lalu load ke memory
        self._ensure_model_exists()
        self.landmarker = self._initialize_model()

        # 3. Inisialisasi MTCNN detector
        self.mtcnn_detector = MTCNN()

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

    def _align_face_with_mtcnn(self, image: np.ndarray, padding: float = 0.3):
        """
        Deteksi wajah dengan MTCNN, lalu lakukan face alignment
        berdasarkan posisi kedua mata agar wajah selalu tegak lurus.
 
        Parameters
        ----------
        image   : np.ndarray  – gambar BGR (output kamera / upload)
        padding : float       – margin ekstra di sekitar bounding-box (0.3 = 30%)
 
        Returns
        -------
        aligned   : np.ndarray | None  – crop wajah yang sudah di-align (RGB)
        meta      : dict | None        – info: box, keypoints, angle, scale
        """
        # MTCNN butuh RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.mtcnn_detector.detect_faces(image_rgb)
 
        if not detections:
            return None, None
 
        # Ambil deteksi dengan confidence tertinggi
        best = max(detections, key=lambda d: d['confidence'])
 
        if best['confidence'] < 0.70:
            return None, None
 
        keypoints = best['keypoints']
        left_eye  = np.array(keypoints['left_eye'],  dtype=np.float32)   # (x, y) dari sisi KIRI pasien
        right_eye = np.array(keypoints['right_eye'], dtype=np.float32)   # (x, y) dari sisi KANAN pasien
 
        # ── 1. Hitung sudut rotasi ────────────────────────────────────────────
        # Agar garis antar-mata menjadi horizontal (0°)
        delta = right_eye - left_eye          # vektor dari mata kiri ke kanan
        angle_rad = math.atan2(delta[1], delta[0])
        angle_deg = math.degrees(angle_rad)   # positif = wajah miring CCW
 
        # ── 2. Pusat rotasi = titik tengah antara kedua mata ─────────────────
        eye_center = ((left_eye + right_eye) / 2).astype(np.float32)
 
        # ── 3. Rotasi gambar penuh ───────────────────────────────────────────
        h_img, w_img = image_rgb.shape[:2]
        M_rot = cv2.getRotationMatrix2D(tuple(eye_center), angle_deg, 1.0)
        rotated = cv2.warpAffine(
            image_rgb, M_rot, (w_img, h_img),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
 
        # ── 4. Transformasikan titik-titik keypoints ke koordinat baru ───────
        def rotate_pt(pt):
            p = np.array([pt[0], pt[1], 1.0])
            return M_rot @ p  # shape (2,)
 
        left_eye_rot  = rotate_pt(left_eye)
        right_eye_rot = rotate_pt(right_eye)
 
        # ── 5. Buat bounding-box berbasis jarak antar-mata ───────────────────
        eye_dist     = np.linalg.norm(right_eye_rot - left_eye_rot)
        face_width   = eye_dist / 0.45        # rasio empiris: jarak mata ≈ 45% lebar wajah
        face_height  = face_width * 1.3       # wajah sedikit lebih tinggi dari lebarnya
 
        cx = (left_eye_rot[0] + right_eye_rot[0]) / 2
        # Geser center sedikit ke bawah agar dahi & dagu seimbang
        cy = (left_eye_rot[1] + right_eye_rot[1]) / 2 + face_height * 0.1
 
        pad_w = face_width  * (1 + padding) / 2
        pad_h = face_height * (1 + padding) / 2
 
        x1 = int(max(0, cx - pad_w))
        y1 = int(max(0, cy - pad_h))
        x2 = int(min(w_img, cx + pad_w))
        y2 = int(min(h_img, cy + pad_h))
 
        aligned = rotated[y1:y2, x1:x2]
 
        if aligned.size == 0:
            return None, None

        aligned = np.ascontiguousarray(aligned)
 
        meta = {
            "mtcnn_box"   : best['box'],           # [x, y, w, h] pada gambar asli
            "keypoints"   : keypoints,
            "confidence"  : round(best['confidence'], 3),
            "rotation_angle_deg": round(angle_deg, 2),
            "eye_distance_px"   : round(float(eye_dist), 1),
            "crop_box"    : [x1, y1, x2, y2],     # pada gambar yang sudah dirotasi
        }
 
        return aligned, meta
 
    # ==========================================
    # ENDPOINT 1: FACIAL PALSY
    # ==========================================
 
    def analyze_facial_palsy(self, image: np.ndarray):
        """
        Pipeline lengkap:
          1. MTCNN alignment  → wajah lurus & terpotong
          2. MediaPipe        → hitung asimetri
          3. Gabungkan output → kembalikan hasil + gambar terannotasi
        """
        # ── Step 1: MTCNN alignment ──────────────────────────────────────────
        aligned_rgb, align_meta = self._align_face_with_mtcnn(image)
 
        if aligned_rgb is None:
            # Fallback: coba langsung tanpa alignment
            print("[WARN] MTCNN gagal mendeteksi wajah – mencoba tanpa alignment.")
            aligned_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            align_meta  = None
 
        # Ubah ke BGR untuk proses OpenCV selanjutnya
        aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)

        cv2.imwrite("debug_mtcnn_crop.jpg", aligned_bgr)
 
        # ── Step 2: MediaPipe pada wajah yang sudah di-align ─────────────────
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=aligned_rgb)
        detection_result = self.landmarker.detect(mp_image)
 
        if not detection_result.face_landmarks:
            return None, None
 
        landmarks = detection_result.face_landmarks[0]
        results   = self._calculate_and_draw(landmarks, aligned_bgr)
 
        # ── Step 3: Tambahkan metadata alignment ke hasil ────────────────────
        if align_meta:
            results["alignment"] = {
                "rotation_angle_deg": align_meta["rotation_angle_deg"],
                "eye_distance_px"   : align_meta["eye_distance_px"],
                "mtcnn_confidence"  : align_meta["confidence"],
            }
            # Tampilkan info alignment di gambar
            self._draw_alignment_info(aligned_bgr, align_meta)
 
        return results, aligned_bgr
 
    def _draw_alignment_info(self, image: np.ndarray, meta: dict):
        """Tambahkan overlay kecil di sudut kanan atas menampilkan info alignment."""
        h, w = image.shape[:2]
        angle = meta["rotation_angle_deg"]
        conf  = meta["confidence"]
 
        text_angle = f"Rot: {angle:+.1f} deg"
        text_conf  = f"MTCNN: {conf:.2f}"
 
        x_start = w - 220
        for i, txt in enumerate([text_angle, text_conf]):
            cv2.putText(
                image, txt,
                (x_start, 25 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 255, 180), 1, cv2.LINE_AA
            )
 
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
            if mag1 == 0 or mag2 == 0:
                return 0
            return math.degrees(math.acos(np.clip(dot / (mag1 * mag2), -1.0, 1.0)))
 
        # Perhitungan Asimetri
        angle_right = calc_angle(B, R, T)
        angle_left  = calc_angle(B, L, T)
        mouth_diff  = abs(angle_right - angle_left)
 
        left_eye_open  = landmarks[159].y - landmarks[145].y
        right_eye_open = landmarks[386].y - landmarks[374].y
        eye_asym       = abs(left_eye_open - right_eye_open) / face_scale * 10
 
        # Scoring
        m_pct = (mouth_diff / 10.0) * 100 if mouth_diff > 2.5 else 0
        e_pct = (eye_asym / 0.5)    * 100 if eye_asym > 0.08  else 0
        final_pct = round(min(100, max(m_pct, e_pct)))
 
        # Severity Logic
        if final_pct > 45:
            severity, desc, color = "Asimetri Parah",     "Deviasi signifikan",            (0, 0, 255)
        elif final_pct > 30:
            severity, desc, color = "Asimetri Ringan",    "Deviasi ringan",                 (0, 165, 255)
        else:
            severity, desc, color = "Dalam Batas Normal", "Tidak ada gejala signifikan.",   (0, 255, 0)
 
        # --- VISUALIZATION ---
 
        # A. Garis Mulut (Merah)
        cv2.polylines(image, [np.array([B, R, T])], False, (0, 0, 255), 2)
        cv2.polylines(image, [np.array([B, L, T])], False, (0, 0, 255), 2)
 
        # B. Gambar Mata (Cyan & Magenta)
        def draw_eye(indices):
            pts = np.array([get_pt(i) for i in indices], np.int32)
            cv2.polylines(image, [pts], False, (255, 255, 0), 2)    # Cyan
            for p in pts:
                cv2.circle(image, p, 3, (255, 0, 255), -1)           # Magenta
 
        draw_eye([130, 161, 160, 159, 158, 157, 133])  # Kiri
        draw_eye([362, 384, 385, 386, 387, 388, 263])  # Kanan
 
        # C. Label B, T, R, L
        for pt, label in [(B, "B"), (T, "T"), (R, "R"), (L, "L")]:
            cv2.circle(image, pt, 6, (0, 255, 255), -1)
            cv2.circle(image, pt, 6, (0, 0, 0), 2)
            cv2.putText(image, label, (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
 
        # D. Dashboard
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (400, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
 
        # E. Teks Dashboard
        cv2.putText(image, f"Score: {final_pct}%",              (15,  30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(image, severity,                             (15,  60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(image, f"Mouth Diff: {mouth_diff:.1f} deg", (15,  90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, f"Eye Asym : {eye_asym:.3f}",        (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, desc,                                 (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
 
        return {
            "severity_score": final_pct,
            "status_label"  : severity,
            "metrics"       : {
                "raw_severity_pct"     : final_pct,
                "mouth_diff_deg"       : round(mouth_diff, 2),
                "eye_asymmetry_ratio"  : round(eye_asym, 4),
            }
        }
    
    # ==========================================
    # ENDPOINT 2: EYE SYMMETRY (IMPROVED)
    # ==========================================
    def get_gaze_ratio(self, iris_pt, corner_medial, corner_lateral, w, h):
        """Hitung posisi iris relatif dari sudut MEDIAL ke LATERAL."""
        ix  = iris_pt.x * w
        iy  = iris_pt.y * h
        cmx = corner_medial.x * w
        cmy = corner_medial.y * h
        clx = corner_lateral.x * w
        cly = corner_lateral.y * h

        dist_total = math.sqrt((clx - cmx)**2 + (cly - cmy)**2)
        if dist_total < 1e-6:
            return 0.5

        # Proyeksi iris ke axis medial→lateral
        axis_x = (clx - cmx) / dist_total
        axis_y = (cly - cmy) / dist_total
        proj   = (ix - cmx) * axis_x + (iy - cmy) * axis_y

        return max(0.0, min(1.0, proj / dist_total))

    def analyze_eye_symmetry(self, image: np.ndarray):
        """Method Public untuk Endpoint /analyze/eye-symmetry"""
        # [PENTING]: MediaPipe butuh RGB, sedangkan input dari OpenCV adalah BGR.
        image_rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return None, None

        landmarks = detection_result.face_landmarks[0]
        
        results, cropped_eye_image = self._calculate_eye_symmetry(landmarks, image)
        
        return results, cropped_eye_image

    def _calculate_eye_symmetry(self, landmarks, image):
        h, w, _ = image.shape

        # ---- Landmark ----
        l_iris     = landmarks[468]   
        l_medial   = landmarks[33]    
        l_lateral  = landmarks[133]   

        r_iris     = landmarks[473]   
        r_medial   = landmarks[263]   
        r_lateral  = landmarks[362]   

        # ---- Gaze ratio ----
        gaze_L = self.get_gaze_ratio(l_iris, l_medial, l_lateral, w, h)
        gaze_R = self.get_gaze_ratio(r_iris, r_medial, r_lateral, w, h)

        # ---- Sinkronisasi ----
        gaze_diff = abs(gaze_L - gaze_R)

        THRESH_NORMAL = 0.04   
        THRESH_MILD   = 0.10   

        if gaze_diff <= THRESH_NORMAL:
            score  = int((gaze_diff / THRESH_NORMAL) * 20)  
            status = "Normal / Simetris"
            color  = (0, 255, 0) # Hijau
        elif gaze_diff <= THRESH_MILD:
            t      = (gaze_diff - THRESH_NORMAL) / (THRESH_MILD - THRESH_NORMAL)
            score  = int(20 + (t * 30))                     
            status = "Asimetri Ringan"
            color  = (0, 165, 255) # Orange
        else:
            t      = min(1.0, (gaze_diff - THRESH_MILD) / 0.08)
            score  = int(50 + (t * 50))                     
            status = "Asimetri Parah"
            color  = (0, 0, 255) # Merah

        # --- LOGIKA TAMBAHAN UNTUK MENCEGAH ERROR PYDANTIC ---
        is_anomaly = bool(score > 35)
        
        CENTER = 0.5
        DEAD_ZONE = 0.12
        avg_gaze = (gaze_L + gaze_R) / 2
        
        if avg_gaze > CENTER + DEAD_ZONE:
            gaze_dir = "Melirik Kanan"
        elif avg_gaze < CENTER - DEAD_ZONE:
            gaze_dir = "Melirik Kiri"
        else:
            gaze_dir = "Lurus / Tengah"

        # ==================== 1. VISUALISASI KE GAMBAR UTUH ====================
        def lm_px(idx):
            return int(landmarks[idx].x * w), int(landmarks[idx].y * h)

        for idx in [468, 473]:
            cx, cy = lm_px(idx)
            cv2.circle(image, (cx, cy), 5, (0, 255, 255), -1)
            cv2.circle(image, (cx, cy), 5, (0, 0, 0), 1)

        for idx in [33, 133, 263, 362]:
            cx, cy = lm_px(idx)
            cv2.circle(image, (cx, cy), 4, (255, 200, 0), -1)

        cv2.arrowedLine(image, lm_px(33),  lm_px(133), (180, 180, 0), 1, tipLength=0.15)
        cv2.arrowedLine(image, lm_px(263), lm_px(362), (180, 180, 0), 1, tipLength=0.15)

        # ==================== 2. MENCARI BOUNDING BOX AREA MATA ====================
        eye_points =[133, 362, 33, 263, 159, 386, 145, 374, 70, 300] 
        px_pts =[lm_px(idx) for idx in eye_points]

        xs = [p[0] for p in px_pts]
        ys = [p[1] for p in px_pts]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        box_w = max_x - min_x
        box_h = max_y - min_y

        pad_x = int(box_w * 0.25)
        pad_y = int(box_h * 1.0) 

        x1 = max(0, min_x - pad_x)
        y1 = max(0, min_y - pad_y)
        x2 = min(w, max_x + pad_x)
        y2 = min(h, max_y + pad_y)

        # ==================== 3. CROP GAMBAR (DENGAN PENGAMAN MEMORI) ====================
        eye_crop = image[y1:y2, x1:x2].copy()
        
        # [PENTING]: Mencegah Error 500 saat API mengubah gambar ke format Base64
        eye_crop = np.ascontiguousarray(eye_crop)
        
        if eye_crop.size == 0 or eye_crop.shape[0] == 0 or eye_crop.shape[1] == 0:
            eye_crop = np.ascontiguousarray(image.copy()) # Fallback aman ke gambar asli
            
        # ==================== 4. DASHBOARD PADA GAMBAR CROP ====================
        ch, cw = eye_crop.shape[:2]
        
        overlay = eye_crop.copy()
        dash_h = min(ch, 100) 
        cv2.rectangle(overlay, (0, 0), (cw, dash_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, eye_crop, 0.4, 0, eye_crop)

        cv2.putText(eye_crop, f"Severity: {score}% | {status}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(eye_crop, f"Gaze L (med->lat): {gaze_L:.3f}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(eye_crop, f"Gaze R (med->lat): {gaze_R:.3f}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        # cv2.putText(eye_crop, f"Diff: {gaze_diff:.3f}",
        #            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        return {
            "severity_score": score,
            "status_label": status,
            # "is_anomaly_detected": is_anomaly,
            "metrics": {
                "gaze_left": round(float(gaze_L), 3),
                "gaze_right": round(float(gaze_R), 3),
                "gaze_difference": round(float(gaze_diff), 3),
                # "gaze_direction": gaze_dir
            }
        }, eye_crop