import cv2
import mediapipe as mp
import numpy as np
import math
import os
import urllib.request
import tempfile

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# 1. ONE EURO FILTER CLASSES & HELPERS
# ==========================================
class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.last_raw_value = None

    def apply_with_alpha(self, value, alpha):
        if self.last_raw_value is None:
            self.last_raw_value = value
        else:
            self.last_raw_value = alpha * value + (1 - alpha) * self.last_raw_value
        return self.last_raw_value

    def apply(self, value):
        return self.apply_with_alpha(value, self.alpha)


class OneEuroFilter:
    def __init__(self, frequency, min_cutoff=1.0, beta=0.0, derivate_cutoff=1.0, to_print=False):
        if frequency <= 0:
            raise ValueError("Frequency should be > 0")
        if min_cutoff <= 0:
            raise ValueError("Min cutoff should be > 0")
        if derivate_cutoff <= 0:
            raise ValueError("Derivate cutoff should be > 0")

        self.frequency = frequency
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivate_cutoff = derivate_cutoff

        self.x = LowPassFilter(self.alpha(self.min_cutoff))
        self.dx = LowPassFilter(self.alpha(self.derivate_cutoff))
        self.last_time = np.iinfo(np.int64).min
        self.to_print = to_print

    def alpha(self, cutoff):
        te = 1.0 / self.frequency
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def apply(self, value, timestamp, value_scale=1.0):
        new_timestamp = timestamp

        if self.last_time >= new_timestamp:
            return value

        if self.last_time != np.iinfo(np.int64).min and new_timestamp != 0:
            # Menggunakan selisih waktu dalam detik untuk akurasi frekuensi (Hz)
            dt = (new_timestamp - self.last_time) / 1000.0
            if dt > 0:
                self.frequency = 1.0 / dt
                
        self.last_time = new_timestamp

        dvalue = self.x.last_raw_value if self.x.last_raw_value is not None else 0.0
        dvalue = (value - dvalue) * value_scale * self.frequency

        edvalue = self.dx.apply_with_alpha(dvalue, self.alpha(self.derivate_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(edvalue)
        result = self.x.apply_with_alpha(value, self.alpha(cutoff))

        if self.to_print:
            print(f"original: {value}, new: {result}")

        return result


class FilteredLandmark:
    """Kelas pembungkus sederhana untuk meniru struktur Landmark asli MediaPipe"""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

# ==========================================
# 2. MAIN ANALYZER SERVICE
# ==========================================
class ArmAnalyzerService:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.base_dir, 'model')
        self.model_path = os.path.join(self.model_dir, 'pose_landmarker.task')

        self._ensure_model_exists()
        
        with open(self.model_path, "rb") as f:
            self.model_bytes = f.read()

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        if not os.path.exists(self.model_path):
            print("Downloading MediaPipe Pose model...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
            urllib.request.urlretrieve(url, self.model_path)

    def _initialize_model(self):
        base_options = python.BaseOptions(model_asset_buffer=self.model_bytes)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
            running_mode=vision.RunningMode.VIDEO
        )
        return vision.PoseLandmarker.create_from_options(options)
    
    def _get_object_scale(self, landmarks):
        xs = [landmark.x for landmark in landmarks]
        ys = [landmark.y for landmark in landmarks]
        object_width = max(xs) - min(xs)
        object_height = max(ys) - min(ys)
        return (object_width + object_height) / 2.0

    def analyze_arm_weakness(self, input_video_path: str, output_video_path: str):
        
        def draw_ui_box(img, text, x, y, bg_color=(30, 30, 30), text_color=(255, 255, 255), font_scale=0.7, thickness=2):
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            pad_x, pad_y = 15, 10
            overlay = img.copy()
            cv2.rectangle(overlay, (x, y), (x + tw + pad_x*2, y + th + pad_y*2), bg_color, -1)
            cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
            cv2.putText(img, text, (x + pad_x, y + th + pad_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        landmarker = self._initialize_model()

        # --- INISIALISASI ONE EURO FILTER ---
        # Membuat array filter 3D (33 landmarks x 3 koordinat) yang baru untuk proses video ini
        min_cutoff = 0.05
        beta = 80.0
        derivate_cutoff = 1.0
        num_landmarks = 33
        num_coordinates = 3  # x, y, z

        filters = np.array([[
            OneEuroFilter(frequency=30, min_cutoff=min_cutoff, beta=beta, derivate_cutoff=derivate_cutoff, to_print=False)
            for _ in range(num_coordinates)]
            for _ in range(num_landmarks)
        ])
        
        current_sec = 0.0
        max_drift = 0
        max_asymmetry = 0
        drift_threshold_ref = 1
        final_result_label = "Normal / Kekuatan Penuh"

        try:
            cap = cv2.VideoCapture(input_video_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or math.isnan(fps) or fps > 120: 
                fps = 30
                
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, int(fps), (w, h))

            test_active = False
            test_start_sec = 0.0
            baseline_ly, baseline_ry = 0, 0
            violation_start_sec = None
            current_violation = None
            
            frame_idx = 0
            last_timestamp_ms = -1 

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                    
                calculated_timestamp_ms = int((frame_idx / fps) * 1000)
                
                if calculated_timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                else:
                    timestamp_ms = calculated_timestamp_ms
                    
                last_timestamp_ms = timestamp_ms
                current_sec = timestamp_ms / 1000.0
                frame_idx += 1

                # Konversi BGR ke RGB untuk deteksi MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks:
                    raw_landmarks = result.pose_landmarks[0]
                    object_scale = self._get_object_scale(raw_landmarks)

                    # --- PROSES FILTERING KOORDINAT ---
                    # Terapkan One Euro Filter ke seluruh landmark koordinat x, y, z
                    landmarks = []
                    for i, lm in enumerate(raw_landmarks):
                        fx = filters[i][0].apply(lm.x, timestamp_ms, object_scale)
                        fy = filters[i][1].apply(lm.y, timestamp_ms, object_scale)
                        fz = filters[i][2].apply(lm.z, timestamp_ms, object_scale)
                        landmarks.append(FilteredLandmark(fx, fy, fz))

                    # Fungsi penolong get_pt dan variabel sisa logika di bawah otomatis menggunakan data terfilter
                    def get_pt(idx): return int(landmarks[idx].x * w), int(landmarks[idx].y * h)

                    ls, rs = landmarks[11], landmarks[12] 
                    lw, rw = landmarks[15], landmarks[16] 
                    
                    pt_ls, pt_rs = get_pt(11), get_pt(12)
                    pt_lw, pt_rw = get_pt(15), get_pt(16)

                    shoulder_width = math.sqrt((ls.x - rs.x)**2 + (ls.y - rs.y)**2)
                    
                    if shoulder_width > 0:
                        DRIFT_THRESHOLD = shoulder_width * 0.35 
                        ASYM_THRESHOLD = shoulder_width * 0.20
                        drift_threshold_ref = DRIFT_THRESHOLD
                        
                        arms_raised = (lw.y < ls.y + 0.25) and (rw.y < rs.y + 0.25)

                        color_bone = (255, 180, 50)  
                        
                        cv2.line(frame, pt_ls, pt_lw, color_bone, 3, cv2.LINE_AA)
                        cv2.line(frame, pt_rs, pt_rw, color_bone, 3, cv2.LINE_AA)
                        
                        cv2.circle(frame, pt_lw, 9, color_bone, 2, cv2.LINE_AA)
                        cv2.circle(frame, pt_lw, 4, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, pt_rw, 9, color_bone, 2, cv2.LINE_AA)
                        cv2.circle(frame, pt_rw, 4, (255, 255, 255), -1, cv2.LINE_AA)

                        if not test_active and arms_raised:
                            test_active = True
                            test_start_sec = current_sec
                            baseline_ly = lw.y
                            baseline_ry = rw.y

                        elif test_active:
                            elapsed = current_sec - test_start_sec
                            
                            if elapsed <= 10.0:
                                y_l = int(baseline_ly * h)
                                y_r = int(baseline_ry * h)
                                color_baseline = (100, 200, 100) 
                                
                                cv2.line(frame, (0, y_l), (w, y_l), color_baseline, 2, cv2.LINE_AA)
                                cv2.putText(frame, "REF L", (10, y_l - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_baseline, 1, cv2.LINE_AA)
                                
                                cv2.line(frame, (0, y_r), (w, y_r), color_baseline, 2, cv2.LINE_AA)
                                cv2.putText(frame, "REF R", (w - 60, y_r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_baseline, 1, cv2.LINE_AA)

                                drift_left = lw.y - baseline_ly
                                drift_right = rw.y - baseline_ry
                                asymmetry = abs(drift_left - drift_right)

                                max_drift = max(max_drift, max(drift_left, drift_right))
                                max_asymmetry = max(max_asymmetry, asymmetry)

                                current_violation = None
                                if drift_left > DRIFT_THRESHOLD and drift_right > DRIFT_THRESHOLD:
                                    current_violation = "Kelemahan Kedua Lengan"
                                elif drift_left > DRIFT_THRESHOLD:
                                    current_violation = "Kelemahan Lengan Kiri"
                                elif drift_right > DRIFT_THRESHOLD:
                                    current_violation = "Kelemahan Lengan Kanan"
                                elif asymmetry > ASYM_THRESHOLD:
                                    current_violation = "Asimetri Lengan Terdeteksi"

                                if current_violation:
                                    if violation_start_sec is None:
                                        violation_start_sec = current_sec
                                    
                                    violation_duration = current_sec - violation_start_sec
                                    
                                    draw_ui_box(frame, f"WARNING: {current_violation}!", 30, 140, bg_color=(0, 0, 180), font_scale=0.8)
                                    
                                    if violation_duration > 2.0:
                                        final_result_label = current_violation
                                else:
                                    violation_start_sec = None

                                remaining = max(0, 10.0 - elapsed)
                                draw_ui_box(frame, f"EVALUATION TIME: {remaining:.1f}s", 30, 70, bg_color=(180, 100, 0), font_scale=0.8)
                            
                            else:
                                bg_color_result = (0, 140, 0) if "Normal" in final_result_label else (0, 0, 180)
                                draw_ui_box(frame, "STATUS: ANALYSIS COMPLETE", 30, 70, bg_color=(40, 40, 40), font_scale=0.8)
                                draw_ui_box(frame, f"RESULT: {final_result_label}", 30, 140, bg_color=bg_color_result, font_scale=0.9)

                else:
                    draw_ui_box(frame, "NO SUBJECT DETECTED", 30, 70, bg_color=(0, 0, 180), font_scale=0.8)

                out.write(frame)

            cap.release()
            out.release()
        
        finally:
            landmarker.close()

        severity_ratio = (max_drift / drift_threshold_ref) if drift_threshold_ref > 0 else 0
        severity_score = int(min(100, max(0, severity_ratio * 50)))

        if final_result_label == "Normal / Kekuatan Penuh" and max_drift > 0:
            if severity_score > 25:
                final_result_label = "Kelemahan Sangat Ringan"

        metrics = {
            "max_arm_drift_ratio": round(severity_ratio, 3),
            "max_asymmetry_ratio": round(max_asymmetry / drift_threshold_ref, 3) if drift_threshold_ref > 0 else 0,
            "test_duration_analyzed_sec": round(current_sec, 2)
        }

        return {
            "severity_score": severity_score,
            "status_label": final_result_label,
            "metrics": metrics
        }