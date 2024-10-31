import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import threading
from pygame import mixer
import time

class PoseDetector:
    def __init__(self, fold_callback=None, width=1280, height=720, fps=30, sound_path='sound.mp3'):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            smooth_landmarks=True
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.fold_callback = fold_callback
        self.fold_threshold = 80
        self.width = width
        self.height = height
        self.fps = fps
        self.last_sound = 0
        self.sound_cooldown = 1.0  # Minimum seconds between sounds
        
        # Initialize pygame mixer for MP3 playback
        mixer.init()
        self.sound = mixer.Sound(sound_path)
        
        # Track detection state
        self.is_folded = False
        self.current_angle = 0
        
    def play_alert_sound(self):
        """Play custom alert sound in a separate thread"""
        current_time = time.time()
        if current_time - self.last_sound >= self.sound_cooldown:
            self.last_sound = current_time
            threading.Thread(target=self.sound.play).start()

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points with vectorization"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)

    def detect_fold(self, landmarks):
        if not landmarks.landmark:
            return False
            
        # Check both left and right side angles
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        
        # Calculate angles for both sides
        left_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        right_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
        
        # Use the average angle
        self.current_angle = (left_angle + right_angle) / 2
        
        return self.current_angle < self.fold_threshold

    @staticmethod
    def draw_leg_lines(frame, landmarks, mp_pose, color):
        """Draw additional lines for leg tracking"""
        pairs = [
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
            (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
            (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
            (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL),
            (mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
            (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL),
            (mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
        ]
        
        for start_point, end_point in pairs:
            start = landmarks.landmark[start_point]
            end = landmarks.landmark[end_point]
            
            start_pos = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
            end_pos = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
            
            cv2.line(frame, start_pos, end_pos, color, 4)
            cv2.circle(frame, start_pos, 6, color, -1)
            cv2.circle(frame, end_pos, 6, color, -1)

    def process_frame(self, frame):
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Resize frame for better performance
        frame = cv2.resize(frame, (self.width, self.height))
        
        # Convert BGR to RGB more efficiently
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.pose.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        if results.pose_landmarks:
            is_folded = self.detect_fold(results.pose_landmarks)
            color = (0, 0, 255) if is_folded else (0, 255, 0)
            
            # Draw enhanced skeleton
            self.mp_draw.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw.DrawingSpec(color=color, thickness=4, circle_radius=6),
                connection_drawing_spec=self.mp_draw.DrawingSpec(color=color, thickness=4)
            )
            
            # Add detailed leg tracking
            self.draw_leg_lines(frame, results.pose_landmarks, self.mp_pose, color)
            
            if is_folded:
                if self.fold_callback:
                    self.fold_callback()
                
                # Play custom sound alert
                self.play_alert_sound()
                
                # Visual alert
                cv2.putText(frame, 'FOLD DETECTED!', 
                           (int(self.width/20), int(self.height/10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            
            # Display angle
            cv2.putText(frame, f'Angle: {self.current_angle:.1f}', 
                       (int(self.width/20), int(self.height/6)),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
                
        return frame

def alert_fold():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Fold detected at {timestamp}!")

def main():
    # Create detector with custom sound file
    detector = PoseDetector(
        fold_callback=alert_fold,
        width=1280,
        height=720,
        fps=30,
        sound_path='get_out_meme.mp3'  # Specify your sound file path here
    )
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, detector.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, detector.height)
    cap.set(cv2.CAP_PROP_FPS, detector.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    cv2.namedWindow('Optimized Pose Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Optimized Pose Detection', detector.width, detector.height)
    
    prev_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = detector.process_frame(frame)
        
        current_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (current_time - prev_time)
        prev_time = current_time
        
        cv2.putText(processed_frame, f'FPS: {int(fps)}', 
                    (detector.width - 300, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        
        cv2.imshow('Optimized Pose Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    mixer.quit()

if __name__ == "__main__":
    main()