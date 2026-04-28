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
        self.landmarker = self._initialize_model()

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        if not os.path.exists(self.model_path):
            print("Downloading MediaPipe Pose model...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
            urllib.request.urlretrieve(url, self.model_path)

    def _initialize_model(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
            running_mode=vision.RunningMode.VIDEO
        )
        return vision.PoseLandmarker.create_from_options(options)

    def analyze_arm_weakness(self, input_video_path: str, output_video_path: str):
        cap = cv2.VideoCapture(input_video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or math.isnan(fps): fps = 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Konfigurasi Video Writer (Gunakan mp4v atau avc1)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

        # --- STATE MANAGEMENT ---
        test_active = False
        test_start_sec = 0.0
        baseline_ly, baseline_ry = 0, 0
        violation_start_sec = None
        current_violation = None
        
        # Metrics tracking
        max_drift = 0
        max_asymmetry = 0
        drift_threshold_ref = 1
        final_result_label = "Normal / Kekuatan Penuh"
        
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Hitung timestamp berdasarkan frame untuk MediaPipe Video Mode
            timestamp_ms = int((frame_idx / fps) * 1000)
            current_sec = frame_idx / fps
            frame_idx += 1

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                
                # Fungsi helper koordinat
                def get_pt(idx): return int(landmarks[idx].x * w), int(landmarks[idx].y * h)

                ls, rs = landmarks[11], landmarks[12] # Bahu
                lw, rw = landmarks[15], landmarks[16] # Pergelangan Tangan
                
                pt_ls, pt_rs = get_pt(11), get_pt(12)
                pt_lw, pt_rw = get_pt(15), get_pt(16)

                shoulder_width = math.sqrt((ls.x - rs.x)**2 + (ls.y - rs.y)**2)
                
                if shoulder_width > 0:
                    DRIFT_THRESHOLD = shoulder_width * 0.35 
                    ASYM_THRESHOLD = shoulder_width * 0.20
                    drift_threshold_ref = DRIFT_THRESHOLD
                    
                    arms_raised = (lw.y < ls.y + 0.25) and (rw.y < rs.y + 0.25)

                    # VISUALISASI DASAR
                    cv2.line(frame, pt_ls, pt_lw, (255, 255, 0), 2)
                    cv2.line(frame, pt_rs, pt_rw, (255, 255, 0), 2)
                    cv2.circle(frame, pt_lw, 8, (0, 0, 255), -1)
                    cv2.circle(frame, pt_rw, 8, (0, 0, 255), -1)

                    # LOGIKA TES
                    if not test_active and arms_raised:
                        # Mulai tes saat lengan pertama kali terangkat
                        test_active = True
                        test_start_sec = current_sec
                        baseline_ly = lw.y
                        baseline_ry = rw.y

                    elif test_active:
                        elapsed = current_sec - test_start_sec
                        
                        # Jika sudah lewat 10 detik, set selesai
                        if elapsed <= 10.0:
                            # Gambar Baseline
                            cv2.line(frame, (0, int(baseline_ly * h)), (w, int(baseline_ly * h)), (0, 255, 0), 1)
                            cv2.line(frame, (0, int(baseline_ry * h)), (w, int(baseline_ry * h)), (0, 255, 0), 1)

                            drift_left = lw.y - baseline_ly
                            drift_right = rw.y - baseline_ry
                            asymmetry = abs(drift_left - drift_right)

                            # Track maksimum deviasi untuk skor keparahan
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
                                
                                cv2.putText(frame, f"WARNING: {current_violation}!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                                
                                # Jika melebihi 2 detik, catat status akhirnya
                                if violation_duration > 2.0:
                                    final_result_label = current_violation
                            else:
                                violation_start_sec = None

                            # Tampilkan Waktu
                            remaining = max(0, 10.0 - elapsed)
                            cv2.putText(frame, f"Time: {remaining:.1f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        
                        else:
                            # Tampilkan Hasil jika tes sudah lebih dari 10 detik
                            color = (0, 255, 0) if "Normal" in final_result_label else (0, 0, 255)
                            cv2.putText(frame, "HASIL ANALISIS:", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.putText(frame, final_result_label, (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            else:
                cv2.putText(frame, "No Body Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            out.write(frame)

        cap.release()
        out.release()

        # KALKULASI SKOR SEVERITY (0-100)
        # Jika max_drift mencapai drift_threshold -> severity = 50%
        # Jika max_drift mencapai 2x drift_threshold -> severity = 100%
        severity_ratio = (max_drift / drift_threshold_ref) if drift_threshold_ref > 0 else 0
        severity_score = int(min(100, max(0, severity_ratio * 50)))

        # Override severity jika tidak ada kegagalan tapi ada asimetri
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