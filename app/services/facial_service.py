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
    @staticmethod
    def _get_yaw_pitch(lm):
        """
        Estimate head yaw and pitch from stable landmark pairs.
        Returns (yaw_rad, pitch_rad) in image-space.

        Yaw  : rotation around vertical axis   (left/right turn)
        Pitch: rotation around horizontal axis  (nod up/down)

        We use the eye-to-eye axis for yaw and the nose-tip vs
        midpoint-between-eyes for pitch.  Both are cheap and
        stable without a full 3-D solver.
        """
        # Yaw from horizontal angle of inter-ocular axis
        # lm[33] = right eye medial corner, lm[263] = left eye medial corner
        dx = lm[263].x - lm[33].x
        dy = lm[263].y - lm[33].y
        yaw = math.atan2(dy, dx)          # 0 when perfectly level

        # Pitch from ratio of nose-tip Y vs eye midpoint Y
        eye_mid_y = (lm[33].y + lm[263].y) / 2
        nose_y    = lm[1].y
        face_h    = abs(lm[152].y - lm[10].y) + 1e-9   # chin to top-of-head
        pitch = (nose_y - eye_mid_y) / face_h            # ~0.18 at neutral

        return yaw, pitch

    @staticmethod
    def _rotate_pt(x, y, cx, cy, angle):
        """Rotate (x,y) around (cx,cy) by -angle to cancel head roll."""
        cos_a, sin_a = math.cos(-angle), math.sin(-angle)
        nx = cos_a * (x - cx) - sin_a * (y - cy) + cx
        ny = sin_a * (x - cx) + cos_a * (y - cy) + cy
        return nx, ny

    def _normalised_landmarks(self, lm, w, h):
        """
        Return pixel coordinates corrected for head ROLL only.
        Yaw/pitch corrections require a 3-D model; roll is the
        main source of false asymmetry and is easy to undo in 2-D.
        """
        # Roll angle = tilt of inter-ocular axis from horizontal
        dx = lm[263].x * w - lm[33].x * w
        dy = lm[263].y * h - lm[33].y * h
        roll = math.atan2(dy, dx)

        cx = (lm[33].x + lm[263].x) / 2 * w
        cy = (lm[33].y + lm[263].y) / 2 * h

        def px(i):
            rx, ry = self._rotate_pt(lm[i].x * w, lm[i].y * h, cx, cy, roll)
            return rx, ry

        return px, roll

    @staticmethod
    def _face_scale(lm):
        """Inter-ocular distance (normalised coords). Robust, view-stable."""
        dx = lm[33].x - lm[263].x
        dy = lm[33].y - lm[263].y
        return math.sqrt(dx * dx + dy * dy) + 1e-9

    # ──────────────────────────────────────────────────────────
    # FIVE FEATURE SIGNALS  (all return non-negative floats;
    # larger = more asymmetric)
    # ──────────────────────────────────────────────────────────

    def _signal_mouth_corner(self, px, lm):
        """
        Vertical displacement of mouth corners relative to the
        midline (mean Y of the two corners).
        Normalised by mouth width → scale-invariant.

        Landmarks:
          61  = right mouth corner
          291 = left  mouth corner
          13  = upper lip centre (midpoint reference)
          14  = lower lip centre
        """
        r_x, r_y = px(61)
        l_x, l_y = px(291)

        mouth_width  = abs(l_x - r_x) + 1e-9
        corner_delta = abs(r_y - l_y) / mouth_width   # ratio; ~0 when symmetric
        return corner_delta

    def _signal_lip_curl(self, px, lm):
        """
        Asymmetry of upper-lip curvature.
        We approximate curl as the Y offset of the lip peak
        (37 right-side cupid's bow, 267 left-side) relative to
        the lip centre (13).

        Normalised by inter-corner width.
        """
        _, centre_y = px(13)
        _, r_peak_y = px(37)
        _, l_peak_y = px(267)

        r_x, _ = px(61)
        l_x, _ = px(291)
        width = abs(l_x - r_x) + 1e-9

        r_curl = (centre_y - r_peak_y) / width
        l_curl = (centre_y - l_peak_y) / width
        return abs(r_curl - l_curl)

    def _signal_eye_openness(self, px, lm):
        """
        Vertical aperture of each eye, normalised by eye width
        (not by inter-ocular distance, which adds yaw sensitivity).

        Right eye: 159 upper lid, 145 lower lid, 133/33 lateral/medial corners
        Left  eye: 386 upper lid, 374 lower lid, 362/263 lateral/medial corners
        """
        def aperture(top_i, bot_i, lat_i, med_i):
            _, t_y = px(top_i)
            _, b_y = px(bot_i)
            l_x, _ = px(lat_i)
            m_x, _ = px(med_i)
            eye_w  = abs(l_x - m_x) + 1e-9
            return (b_y - t_y) / eye_w    # positive when eye is open

        r_ap = aperture(159, 145, 133, 33)
        l_ap = aperture(386, 374, 362, 263)
        return abs(r_ap - l_ap)

    def _signal_nasolabial(self, px, lm):
        """
        Depth of the nasolabial fold, approximated as the angle
        at the alar base between the nose wing and the mouth corner.

        Right: 102 (alar), 61 (corner), 49 (cheek reference)
        Left : 331 (alar), 291 (corner), 279 (cheek reference)

        A flattened fold (common in palsy) produces a larger angle.
        """
        def fold_angle(alar_i, corner_i, cheek_i):
            ax, ay = px(alar_i)
            cx, cy = px(corner_i)
            hx, hy = px(cheek_i)
            v1 = np.array([ax - cx, ay - cy])
            v2 = np.array([hx - cx, hy - cy])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                return 0.0
            return math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))

        r_fold = fold_angle(102, 61, 49)
        l_fold = fold_angle(331, 291, 279)
        return abs(r_fold - l_fold) / 90.0    # normalise to ~[0,1]

    def _signal_eyebrow_height(self, px, lm):
        """
        Mean eyebrow height above the upper eyelid, normalised
        by eye-width.  Uses the brow arch peak on each side.

        Right brow peak: 105   Right upper lid centre: 159
        Left  brow peak: 334   Left  upper lid centre: 386
        """
        def brow_gap(brow_i, lid_i, lat_i, med_i):
            _, br_y = px(brow_i)
            _, li_y = px(lid_i)
            l_x, _  = px(lat_i)
            m_x, _  = px(med_i)
            eye_w   = abs(l_x - m_x) + 1e-9
            return (li_y - br_y) / eye_w   # positive; larger = brow higher

        r_gap = brow_gap(105, 159, 133, 33)
        l_gap = brow_gap(334, 386, 362, 263)
        return abs(r_gap - l_gap)

    # ──────────────────────────────────────────────────────────
    # SCORING
    # ──────────────────────────────────────────────────────────

    # Empirical thresholds (value at which signal ≈ "clearly abnormal")
    # Tune these on your validation set if you have labelled data.
    _SIGNAL_SCALES = {
        'mouth_corner':  0.12,   # ratio; ~0.04 natural variation
        'lip_curl':      0.08,
        'eye_openness':  0.20,   # ratio per eye-width
        'nasolabial':    0.25,   # normalised angle
        'eyebrow_height':0.18,
    }

    # Weights sum to 1.0.  Mouth corner + eye are the two most
    # clinically reliable signs; others provide corroborating evidence.
    _WEIGHTS = {
        'mouth_corner':   0.35,
        'eye_openness':   0.30,
        'lip_curl':       0.15,
        'eyebrow_height': 0.10,
        'nasolabial':     0.10,
    }

    @staticmethod
    def _sigmoid_score(raw, k=8.0):
        """
        Map raw weighted signal (0 = perfect symmetry, 1 = threshold)
        to a 0–100 score via sigmoid, centred at 0.5.
        k controls steepness: higher k = sharper transition.
        """
        # Sigmoid: 0→~2, 0.5→50, 1→~98, values beyond 1 approach 100
        return round(100 / (1 + math.exp(-k * (raw - 0.5))))

    def _compute_score(self, signals: dict):
        """
        Normalise each signal by its scale threshold, apply weights,
        squeeze through sigmoid.  Also returns per-signal contributions
        and a confidence flag.
        """
        weighted_sum = 0.0
        contributions = {}
        for name, value in signals.items():
            normalised = value / self._SIGNAL_SCALES[name]
            w = self._WEIGHTS[name]
            contributions[name] = round(normalised, 4)
            weighted_sum += w * normalised

        final_score = self._sigmoid_score(weighted_sum)

        # Confidence: low if the two dominant signals strongly disagree.
        # (e.g. clear mouth asymmetry but symmetric eyes → could be natural)
        dominant = [
            signals['mouth_corner'] / self._SIGNAL_SCALES['mouth_corner'],
            signals['eye_openness'] / self._SIGNAL_SCALES['eye_openness'],
        ]
        low_confidence = abs(dominant[0] - dominant[1]) > 1.2

        return final_score, contributions, low_confidence

    # ──────────────────────────────────────────────────────────
    # PUBLIC ENDPOINT
    # ──────────────────────────────────────────────────────────

    def analyze_facial_palsy(self, image: np.ndarray):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return None, None

        landmarks = detection_result.face_landmarks[0]

        # Reject extreme yaw/pitch early — measurements are unreliable
        yaw, pitch = self._get_yaw_pitch(landmarks)
        if abs(yaw) > math.radians(25) or abs(pitch - 0.18) > 0.12:
            return {"error": "head_pose_too_extreme", "yaw_deg": round(math.degrees(yaw), 1)}, image

        results = self._calculate_and_draw(landmarks, image)
        return results, image

    # ──────────────────────────────────────────────────────────
    # CALCULATION + VISUALISATION
    # ──────────────────────────────────────────────────────────

    def _calculate_and_draw(self, landmarks, image):
        h, w, _ = image.shape
        lm = landmarks

        # Roll-corrected pixel accessor
        px, roll_rad = self._normalised_landmarks(lm, w, h)

        # ── Compute the five signals ──────────────────────────
        signals = {
            'mouth_corner':   self._signal_mouth_corner(px, lm),
            'lip_curl':       self._signal_lip_curl(px, lm),
            'eye_openness':   self._signal_eye_openness(px, lm),
            'nasolabial':     self._signal_nasolabial(px, lm),
            'eyebrow_height': self._signal_eyebrow_height(px, lm),
        }

        final_score, contributions, low_confidence = self._compute_score(signals)

        # ── Severity classification ───────────────────────────
        if final_score > 50:
            severity = "Asimetri Parah"
            desc     = "Deviasi signifikan"
            color    = (0, 0, 255)
        elif final_score > 20:
            severity = "Asimetri Ringan"
            desc     = "Deviasi ringan"
            color    = (0, 165, 255)
        else:
            severity = "Dalam Batas Normal"
            desc     = "Tidak ada gejala signifikan."
            color    = (0, 255, 0)

        if low_confidence:
            desc += " (confidence rendah)"

        # ── Visualisation ─────────────────────────────────────
        def draw_pt(i, c=(0, 255, 255), r=5):
            x_, y_ = px(i)
            cv2.circle(image, (int(x_), int(y_)), r, c, -1)
            cv2.circle(image, (int(x_), int(y_)), r, (0, 0, 0), 1)

        def draw_line(i, j, c=(0, 0, 255), t=2):
            x1, y1 = px(i); x2, y2 = px(j)
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), c, t)

        # Mouth corners + centre
        for idx in [61, 291, 13, 14]:
            draw_pt(idx)
        draw_line(61, 291, (0, 0, 200))

        # Eye landmarks
        for eye_ids in (
            [33, 159, 158, 157, 133, 145, 144, 163],   # right
            [263, 386, 385, 384, 362, 374, 373, 390],   # left
        ):
            for a, b in zip(eye_ids, eye_ids[1:]):
                draw_line(a, b, (255, 220, 0), 2)
            for idx in eye_ids:
                draw_pt(idx, (255, 0, 255), 3)

        # Eyebrow peaks
        for idx in [105, 334]:
            draw_pt(idx, (0, 200, 255), 4)

        # Nasolabial reference points
        for idx in [102, 331, 49, 279]:
            draw_pt(idx, (200, 100, 255), 3)

        # ── Dashboard overlay ─────────────────────────────────
        dash_h = 180
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (420, dash_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)

        cv2.putText(image, f"Score: {final_score}%", (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        cv2.putText(image, severity, (15, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)
        cv2.putText(image, desc, (15, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)

        y_offset = 105
        for name, norm_val in contributions.items():
            bar_len = int(min(norm_val, 2.0) / 2.0 * 180)
            bar_color = (0, 200, 0) if norm_val < 0.7 else \
                        (0, 165, 255) if norm_val < 1.2 else (0, 0, 220)
            cv2.rectangle(image, (110, y_offset - 8), (110 + bar_len, y_offset + 2), bar_color, -1)
            label = name.replace('_', ' ')[:14]
            cv2.putText(image, f"{label}: {norm_val:.2f}",
                        (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (180, 180, 180), 1)
            y_offset += 16

        # Roll correction note
        cv2.putText(image, f"Roll corr: {math.degrees(roll_rad):.1f}deg",
                    (15, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

        return {
            "severity_score":  final_score,
            "status_label":    severity,
            "low_confidence":  low_confidence,
            "metrics": {
                "mouth_corner_norm":   contributions['mouth_corner'],
                "eye_openness_norm":   contributions['eye_openness'],
                "lip_curl_norm":       contributions['lip_curl'],
                "eyebrow_height_norm": contributions['eyebrow_height'],
                "nasolabial_norm":     contributions['nasolabial'],
                "roll_correction_deg": round(math.degrees(roll_rad), 2),
            }
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
            score  = int((gaze_diff / THRESH_NORMAL) * 20)  # Skor: 0 - 20%
            status = "Normal / Simetris"
            color  = (0, 255, 0) # Hijau
        elif gaze_diff <= THRESH_MILD:
            t      = (gaze_diff - THRESH_NORMAL) / (THRESH_MILD - THRESH_NORMAL)
            score  = int(20 + (t * 30))                     # Skor: 20 - 50%
            status = "Asimetri Ringan"
            color  = (0, 165, 255) # Orange
        else:
            t      = min(1.0, (gaze_diff - THRESH_MILD) / 0.08)
            score  = int(50 + (t * 50))                     # Skor: 50 - 100%
            status = "Asimetri Parah"
            color  = (0, 0, 255) # Merah

        # is_anomaly = score > 35

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

        cv2.putText(image, f"Severity: {score}%  |  {status}",
                    (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(image, f"Gaze L (med->lat): {gaze_L:.3f}",
                    (15, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, f"Gaze R (med->lat): {gaze_R:.3f}",
                    (15, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, f"Diff: {gaze_diff:.3f}",
                    (15, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

        # Progress bar sinkronisasi
        bar_x, bar_y, bar_w_px = 15, 150, 200
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w_px, bar_y + 12), (80, 80, 80), -1)
        fill = int(score / 100 * bar_w_px)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill, bar_y + 12), color, -1)
        cv2.putText(image, "Severity Level", (bar_x + bar_w_px + 8, bar_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        return {
            "severity_score": score,
            "status_label": status,
            # "is_anomaly_detected": is_anomaly,
            "metrics": {
                "gaze_left": round(gaze_L, 3),
                "gaze_right": round(gaze_R, 3),
                "gaze_difference": round(gaze_diff, 3),
                # "gaze_direction": gaze_dir
            }
        }