import cv2
import mediapipe as mp
import numpy as np
import math
import os
import urllib.request
import tempfile

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ArmAnalyzerService:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.base_dir, 'model')
        self.model_path = os.path.join(self.model_dir, 'pose_landmarker.task')

        self._ensure_model_exists()
        
        # PERBAIKAN: Baca model bytes satu kali ke memori agar lebih cepat & thread-safe
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
        # Gunakan buffer dari memori alih-alih path file
        base_options = python.BaseOptions(model_asset_buffer=self.model_bytes)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
            running_mode=vision.RunningMode.VIDEO
        )
        return vision.PoseLandmarker.create_from_options(options)

    def analyze_arm_weakness(self, input_video_path: str, output_video_path: str):
        
        # --- ENTERPRISE UI HELPER ---
        def draw_ui_box(img, text, x, y, bg_color=(30, 30, 30), text_color=(255, 255, 255), font_scale=0.7, thickness=2):
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            pad_x, pad_y = 15, 10
            overlay = img.copy()
            # Gambar box dengan sudut tegas khas UI Enterprise
            cv2.rectangle(overlay, (x, y), (x + tw + pad_x*2, y + th + pad_y*2), bg_color, -1)
            # Alpha blending untuk transparansi
            cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
            
            # Render teks presisi
            cv2.putText(img, text, (x + pad_x, y + th + pad_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        # PERBAIKAN: Inisialisasi Landmarker per-request agar state timestamps direset
        # dan tidak bocor/tabrakan (race condition) dengan request user lain.
        landmarker = self._initialize_model()
        
        current_sec = 0.0 # Default fallback
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

            # --- STATE MANAGEMENT ---
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
                    
                # Hitung timestamp berdasarkan frame
                calculated_timestamp_ms = int((frame_idx / fps) * 1000)
                
                # Pastikan timestamp SELALU LEBIH BESAR dari frame sebelumnya secara lokal
                if calculated_timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                else:
                    timestamp_ms = calculated_timestamp_ms
                    
                last_timestamp_ms = timestamp_ms
                current_sec = timestamp_ms / 1000.0
                frame_idx += 1

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                
                # Gunakan model instance yang BARU khusus request ini
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    
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

                        # TAMPILAN VISUALISASI KINEMATIK (Enterprise Style)
                        color_bone = (255, 180, 50)  # Skema warna Soft Blue/Cyan untuk frame rangka
                        
                        # Garis Lengan (Bones)
                        cv2.line(frame, pt_ls, pt_lw, color_bone, 3, cv2.LINE_AA)
                        cv2.line(frame, pt_rs, pt_rw, color_bone, 3, cv2.LINE_AA)
                        
                        # Titik Sendi (Wrists dengan outer halo)
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
                                color_baseline = (100, 200, 100) # Soft Green
                                
                                # Baseline trackers presisi tinggi
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
                                    
                                    # Peringatan Visual Box
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

                # --- ENTERPRISE HEADER BAR ---
                overlay_header = frame.copy()
                cv2.rectangle(overlay_header, (0, 0), (w, 45), (20, 20, 20), -1)
                cv2.addWeighted(overlay_header, 0.85, frame, 0.15, 0, frame)
                cv2.putText(frame, "CLINICAL KINEMATICS: ARM MOTOR DRIFT ANALYSIS", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)

                out.write(frame)

            cap.release()
            out.release()
        
        finally:
            # PERBAIKAN: Selalu pastikan instance dihapus dari memory walau terjadi error / crash
            landmarker.close()

        # KALKULASI SKOR SEVERITY (0-100)
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