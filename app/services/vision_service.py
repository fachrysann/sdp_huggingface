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
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.base_dir, 'model')
        self.model_path = os.path.join(self.model_dir, 'face_landmarker.task')
        self._ensure_model_exists()
        self.landmarker = self._initialize_model()

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.model_path):
            print("Downloading MediaPipe model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)

    def _initialize_model(self):
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

    # ─────────────────────────────────────────
    # HELPER: Mobile-adaptive scale
    # ─────────────────────────────────────────
    def _get_scale(self, w, h):
        """Semua ukuran UI di-scale relatif terhadap sisi terpendek gambar (base = 1080px)"""
        base_dim = min(w, h)
        s = base_dim / 1080.0
        return {
            "s":        s,
            "font_lg":  s * 1.2,
            "font_md":  s * 0.85,
            "font_sm":  s * 0.60,
            "font_xs":  s * 0.50,
            "thick_lg": max(2, int(s * 3)),
            "thick_md": max(1, int(s * 2)),
            "thick_sm": max(1, int(s * 1.5)),
            "line_h":   int(s * 55),
            "pad_x":    int(s * 25),
            "dot_lg":   max(6,  int(s * 12)),
            "dot_md":   max(4,  int(s * 8)),
            "dot_sm":   max(3,  int(s * 5)),
        }

    # ─────────────────────────────────────────
    # ENDPOINT 1: FACIAL PALSY
    # ─────────────────────────────────────────
    def analyze_facial_palsy(self, image: np.ndarray):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.landmarker.detect(mp_image)
        if not detection_result.face_landmarks:
            return None, None
        landmarks = detection_result.face_landmarks[0]
        results = self._calculate_and_draw(landmarks, image)
        return results, image

    def _calculate_and_draw(self, landmarks, image):
        h, w, _ = image.shape
        sc = self._get_scale(w, h)

        def get_pt(i):
            return int(landmarks[i].x * w), int(landmarks[i].y * h)

        # ── Kalkulasi ──
        B, T, R, L = get_pt(6), get_pt(1), get_pt(61), get_pt(291)

        p33, p263 = landmarks[33], landmarks[263]
        face_scale = math.sqrt((p33.x - p263.x)**2 + (p33.y - p263.y)**2)

        def calc_angle(p_target, p_vertex, p_base):
            v1 = np.array([p_target[0] - p_vertex[0], p_target[1] - p_vertex[1]])
            v2 = np.array([p_base[0]   - p_vertex[0], p_base[1]   - p_vertex[1]])
            dot = np.dot(v1, v2)
            mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if mag1 == 0 or mag2 == 0: return 0
            return math.degrees(math.acos(np.clip(dot / (mag1 * mag2), -1.0, 1.0)))

        angle_right = calc_angle(B, R, T)
        angle_left  = calc_angle(B, L, T)
        mouth_diff  = abs(angle_right - angle_left)

        left_eye_open  = landmarks[159].y - landmarks[145].y
        right_eye_open = landmarks[386].y - landmarks[374].y
        eye_asym = abs(left_eye_open - right_eye_open) / face_scale * 10

        m_pct = (mouth_diff / 10.0) * 100 if mouth_diff > 2.5 else 0
        e_pct = (eye_asym  / 0.5)   * 100 if eye_asym  > 0.08 else 0
        final_pct = round(min(100, max(m_pct, e_pct)))

        if final_pct > 45:
            severity, desc, color = "High Severity", "Significant asymmetry. Consult doctor.", (0, 0, 255)
        elif final_pct > 15:
            severity, desc, color = "Mild Asymmetry", "Slight deviation. Monitor.", (0, 165, 255)
        else:
            severity, desc, color = "Normal", "No significant symptoms.", (0, 220, 80)

        # ── Visualisasi landmark ──
        # Garis sudut mulut
        cv2.polylines(image, [np.array([B, R, T])], False, (0, 80, 255),  sc["thick_md"])
        cv2.polylines(image, [np.array([B, L, T])], False, (0, 80, 255),  sc["thick_md"])

        # Kontur mata
        def draw_eye(indices):
            pts = np.array([get_pt(i) for i in indices], np.int32)
            cv2.polylines(image, [pts], False, (0, 255, 255), sc["thick_md"])
            for p in pts:
                cv2.circle(image, p, sc["dot_sm"], (255, 0, 255), -1)

        draw_eye([130, 161, 160, 159, 158, 157, 133])
        draw_eye([362, 384, 385, 386, 387, 388, 263])

        # Label titik B T R L
        for pt, label in [(B, "B"), (T, "T"), (R, "R"), (L, "L")]:
            cv2.circle(image, pt, sc["dot_lg"], (0, 255, 255), -1)
            cv2.circle(image, pt, sc["dot_lg"], (0, 0, 0), sc["thick_md"])
            cv2.putText(image, label,
                        (pt[0] + sc["dot_lg"] + 4, pt[1] - sc["dot_lg"]),
                        cv2.FONT_HERSHEY_SIMPLEX, sc["font_md"],
                        (255, 255, 255), sc["thick_md"])

        # ── Dashboard (mobile-adaptive) ──
        line_h  = sc["line_h"]
        pad_x   = sc["pad_x"]
        dash_h  = line_h * 6 + int(sc["s"] * 20)

        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, dash_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)
        cv2.line(image, (0, dash_h), (w, dash_h), (60, 60, 60), sc["thick_sm"])

        # Baris 1 – Score
        y1 = int(line_h * 0.9)
        cv2.putText(image, f"Score: {final_pct}%",
                    (pad_x, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, sc["font_lg"], color, sc["thick_lg"])

        # Baris 2 – Severity
        y2 = y1 + line_h
        cv2.putText(image, severity,
                    (pad_x, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, sc["font_md"], color, sc["thick_md"])

        # Baris 3 – Mouth diff & Eye asym (dua kolom)
        y3 = y2 + line_h
        cv2.putText(image, f"Mouth Diff: {mouth_diff:.1f} deg",
                    (pad_x, y3),
                    cv2.FONT_HERSHEY_SIMPLEX, sc["font_sm"], (200, 200, 200), sc["thick_sm"])
        cv2.putText(image, f"Eye Asym: {eye_asym:.3f}",
                    (pad_x + w // 2, y3),
                    cv2.FONT_HERSHEY_SIMPLEX, sc["font_sm"], (200, 200, 200), sc["thick_sm"])

        # Baris 4 – Deskripsi
        y4 = y3 + int(line_h * 0.85)
        cv2.putText(image, desc,
                    (pad_x, y4),
                    cv2.FONT_HERSHEY_SIMPLEX, sc["font_sm"] * 0.85, (160, 160, 160), sc["thick_sm"])

        # Baris 5 – Progress bar
        bar_y    = y4 + int(line_h * 0.75)
        bar_w_px = int(w * 0.60)
        bar_h_px = max(8, int(sc["s"] * 18))
        radius   = bar_h_px // 2

        cv2.rectangle(image, (pad_x + radius, bar_y),
                      (pad_x + bar_w_px - radius, bar_y + bar_h_px), (70, 70, 70), -1)
        cv2.circle(image, (pad_x + radius,             bar_y + radius), radius, (70, 70, 70), -1)
        cv2.circle(image, (pad_x + bar_w_px - radius,  bar_y + radius), radius, (70, 70, 70), -1)

        fill_w = max(radius * 2, int(final_pct / 100 * bar_w_px))
        cv2.rectangle(image, (pad_x + radius, bar_y),
                      (pad_x + fill_w - radius, bar_y + bar_h_px), color, -1)
        cv2.circle(image, (pad_x + radius,          bar_y + radius), radius, color, -1)
        cv2.circle(image, (pad_x + fill_w - radius, bar_y + radius), radius, color, -1)

        cv2.putText(image, "SEVERITY",
                    (pad_x + bar_w_px + int(sc["s"] * 18), bar_y + bar_h_px - int(sc["s"] * 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, sc["font_sm"], (160, 160, 160), sc["thick_sm"])

        return {
            "percentage":        final_pct,
            "severity":          severity,
            "is_stroke_detected": final_pct > 35,
            "mouth_diff":        round(mouth_diff, 2),
            "eye_asymmetry":     round(eye_asym, 4)
        }

    # ─────────────────────────────────────────
    # ENDPOINT 2: EYE SYMMETRY
    # ─────────────────────────────────────────
    def get_gaze_ratio(self, iris_pt, corner_medial, corner_lateral, w, h):
        ix  = iris_pt.x * w;  iy  = iris_pt.y * h
        cmx = corner_medial.x * w;  cmy = corner_medial.y * h
        clx = corner_lateral.x * w; cly = corner_lateral.y * h

        dist_total = math.sqrt((clx - cmx)**2 + (cly - cmy)**2)
        if dist_total < 1e-6: return 0.5

        axis_x = (clx - cmx) / dist_total
        axis_y = (cly - cmy) / dist_total
        proj   = (ix - cmx) * axis_x + (iy - cmy) * axis_y
        return max(0.0, min(1.0, proj / dist_total))

    def analyze_eye_symmetry(self, image: np.ndarray):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.landmarker.detect(mp_image)
        if not detection_result.face_landmarks:
            return None, None
        landmarks = detection_result.face_landmarks[0]
        results   = self._calculate_eye_symmetry(landmarks, image)
        return results, image

    def _calculate_eye_symmetry(self, landmarks, image):
        h, w, _ = image.shape
        sc = self._get_scale(w, h)

        # ── Landmark ──
        l_iris    = landmarks[468]; l_medial = landmarks[33];  l_lateral = landmarks[133]
        r_iris    = landmarks[473]; r_medial = landmarks[263]; r_lateral = landmarks[362]

        # ── Kalkulasi ──
        gaze_L    = self.get_gaze_ratio(l_iris, l_medial, l_lateral, w, h)
        gaze_R    = self.get_gaze_ratio(r_iris, r_medial, r_lateral, w, h)
        gaze_diff = abs(gaze_L - gaze_R)

        THRESH_NORMAL = 0.04
        THRESH_MILD   = 0.10

        if gaze_diff <= THRESH_NORMAL:
            score  = int(100 - (gaze_diff / THRESH_NORMAL) * 20)
            status = "Simetris"
            is_sym = True
            color  = (0, 220, 80)
        elif gaze_diff <= THRESH_MILD:
            t      = (gaze_diff - THRESH_NORMAL) / (THRESH_MILD - THRESH_NORMAL)
            score  = int(80 - t * 50)
            status = "Asimetri Ringan"
            is_sym = False
            color  = (0, 165, 255)
        else:
            t      = min(1.0, (gaze_diff - THRESH_MILD) / 0.08)
            score  = int(30 - t * 30)
            status = "Asimetri Signifikan"
            is_sym = False
            color  = (0, 0, 255)

        avg_gaze = (gaze_L + gaze_R) / 2
        if avg_gaze > 0.62:
            gaze_dir = "Melirik Kanan"
        elif avg_gaze < 0.38:
            gaze_dir = "Melirik Kiri"
        else:
            gaze_dir = "Lurus / Tengah"

        # ── Visualisasi landmark ──
        def lm_px(idx):
            return int(landmarks[idx].x * w), int(landmarks[idx].y * h)

        # Kontur kelopak mata kiri & kanan
        left_contour  = [33, 246, 161, 160, 159, 158, 157, 173, 133,
                         155, 154, 153, 145, 144, 163, 7]
        right_contour = [263, 466, 388, 387, 386, 385, 384, 398, 362,
                         382, 381, 380, 374, 373, 390, 249]

        for contour, eye_color in [(left_contour, (255, 200, 0)), (right_contour, (255, 200, 0))]:
            pts = np.array([lm_px(i) for i in contour], np.int32)
            cv2.polylines(image, [pts], True, eye_color, sc["thick_md"])

        # Iris circle (estimasi radius dari lebar iris)
        def draw_iris_circle(iris_lm, left_lm, right_lm, dot_color):
            cx = int(iris_lm.x * w)
            cy = int(iris_lm.y * h)
            lx = int(left_lm.x * w)
            rx = int(right_lm.x * w)
            # radius ≈ setengah lebar iris (landmark 469-471 kiri, 474-476 kanan)
            iris_r = max(sc["dot_lg"], int(abs(rx - lx) * 0.5))
            cv2.circle(image, (cx, cy), iris_r, dot_color, sc["thick_md"])
            cv2.circle(image, (cx, cy), max(3, iris_r // 4), dot_color, -1)

        # Iris kiri: landmark 469(kiri)–471(kanan), iris kanan: 474(kiri)–476(kanan)
        draw_iris_circle(landmarks[468], landmarks[469], landmarks[471], (0, 255, 255))
        draw_iris_circle(landmarks[473], landmarks[474], landmarks[476], (0, 255, 255))

        # Garis axis gaze (medial → lateral)
        cv2.arrowedLine(image, lm_px(33),  lm_px(133), (180, 180, 0), sc["thick_sm"], tipLength=0.08)
        cv2.arrowedLine(image, lm_px(263), lm_px(362), (180, 180, 0), sc["thick_sm"], tipLength=0.08)

        # Garis vertikal tengah (referensi simetri wajah)
        nose_top = lm_px(6)
        nose_bot = lm_px(2)
        mid_x    = (nose_top[0] + nose_bot[0]) // 2
        cv2.line(image,
                 (mid_x, int(h * 0.05)),
                 (mid_x, int(h * 0.55)),
                 (120, 120, 120), sc["thick_sm"])

        # ── Dashboard ──
        line_h = sc["line_h"]
        pad_x  = sc["pad_x"]
        dash_h = line_h * 6 + int(sc["s"] * 20)

        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, dash_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)
        cv2.line(image, (0, dash_h), (w, dash_h), (60, 60, 60), sc["thick_sm"])

        # Baris 1 – Score
        y1 = int(line_h * 0.9)
        cv2.putText(image, f"Score: {score}%  |  {status}",
                    (pad_x, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, sc["font_lg"], color, sc["thick_lg"])

        # Baris 2 – Arah lirikan
        y2 = y1 + line_h
        cv2.putText(image, f"Arah: {gaze_dir}",
                    (pad_x, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, sc["font_md"], (255, 255, 255), sc["thick_md"])

        # Baris 3 – Gaze L & R (dua kolom)
        y3 = y2 + line_h
        cv2.putText(image, f"Gaze L: {gaze_L:.3f}",
                    (pad_x, y3),
                    cv2.FONT_HERSHEY_SIMPLEX, sc["font_sm"], (200, 200, 200), sc["thick_sm"])
        cv2.putText(image, f"Gaze R: {gaze_R:.3f}",
                    (pad_x + w // 2, y3),
                    cv2.FONT_HERSHEY_SIMPLEX, sc["font_sm"], (200, 200, 200), sc["thick_sm"])

        # Baris 4 – Diff & threshold
        y4 = y3 + int(line_h * 0.85)
        cv2.putText(image, f"Diff: {gaze_diff:.3f}   threshold normal < {THRESH_NORMAL}",
                    (pad_x, y4),
                    cv2.FONT_HERSHEY_SIMPLEX, sc["font_sm"] * 0.85, (140, 140, 140), sc["thick_sm"])

        # Baris 5 – Progress bar (rounded)
        bar_y    = y4 + int(line_h * 0.75)
        bar_w_px = int(w * 0.60)
        bar_h_px = max(8, int(sc["s"] * 18))
        radius   = bar_h_px // 2

        cv2.rectangle(image, (pad_x + radius, bar_y),
                      (pad_x + bar_w_px - radius, bar_y + bar_h_px), (70, 70, 70), -1)
        cv2.circle(image, (pad_x + radius,            bar_y + radius), radius, (70, 70, 70), -1)
        cv2.circle(image, (pad_x + bar_w_px - radius, bar_y + radius), radius, (70, 70, 70), -1)

        fill_w = max(radius * 2, int(score / 100 * bar_w_px))
        cv2.rectangle(image, (pad_x + radius, bar_y),
                      (pad_x + fill_w - radius, bar_y + bar_h_px), color, -1)
        cv2.circle(image, (pad_x + radius,          bar_y + radius), radius, color, -1)
        cv2.circle(image, (pad_x + fill_w - radius, bar_y + radius), radius, color, -1)

        cv2.putText(image, "SYNC",
                    (pad_x + bar_w_px + int(sc["s"] * 18), bar_y + bar_h_px - int(sc["s"] * 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, sc["font_sm"], (160, 160, 160), sc["thick_sm"])

        return {
            "symmetry_score":  score,
            "is_symmetrical":  bool(is_sym),
            "gaze_direction":  gaze_dir,
            "gaze_left":       round(gaze_L, 3),
            "gaze_right":      round(gaze_R, 3),
            "gaze_difference": round(gaze_diff, 3),
            "status":          status,
        }