"""
Room Guardian - Face recognition security system for JARVIS
Monitors room while you're away and provides updates when you return
Uses OpenCV Haar Cascade for face detection
"""

import cv2
import pickle
import os
from datetime import datetime
from pathlib import Path
import threading
import time
import numpy as np
from scipy.spatial import distance

# Get base directory
BASE_DIR = Path(__file__).resolve().parent.parent
GUARDIAN_DIR = BASE_DIR / "data" / "room_guardian"
FACE_EMBEDDINGS_FILE = GUARDIAN_DIR / "face_embeddings.pkl"
EVENTS_LOG_FILE = GUARDIAN_DIR / "events.log"

# Ensure directory exists
GUARDIAN_DIR.mkdir(parents=True, exist_ok=True)

# Load cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Global state
guardian_state = {
    "active": False,
    "user_face_embedding": None,
    "events": [],
    "last_detection": None,
}


def _ensure_face_embeddings():
    """Load or initialize face embeddings."""
    if FACE_EMBEDDINGS_FILE.exists():
        try:
            with open(FACE_EMBEDDINGS_FILE, "rb") as f:
                guardian_state["user_face_embedding"] = pickle.load(f)
        except Exception as e:
            print(f"[Guardian] Error loading embeddings: {e}")
            guardian_state["user_face_embedding"] = None
    else:
        guardian_state["user_face_embedding"] = None


def _save_face_embedding(embedding):
    """Save user's face embedding."""
    try:
        with open(FACE_EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(embedding, f)
        guardian_state["user_face_embedding"] = embedding
        print(f"[Guardian] ✅ Face embedding saved")
    except Exception as e:
        print(f"[Guardian] Error saving embedding: {e}")


def _log_event(event_type: str, details: str):
    """Log an event to the events file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    event_entry = f"[{timestamp}] {event_type}: {details}"
    
    guardian_state["events"].append({
        "timestamp": timestamp,
        "type": event_type,
        "details": details
    })
    
    try:
        with open(EVENTS_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(event_entry + "\n")
    except Exception as e:
        print(f"[Guardian] Error writing log: {e}")
    
    print(f"[Guardian] 📝 {event_entry}")


def _extract_face_embedding(frame, face_rect):
    """Extract face embedding from a detected face region."""
    try:
        x, y, w, h = face_rect
        
        # Add padding to face region
        pad = 10
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(frame.shape[1] - x, w + 2*pad)
        h = min(frame.shape[0] - y, h + 2*pad)
        
        # Extract face region
        face_region = frame[y:y+h, x:x+w]
        
        # Simple embedding: average RGB values + histogram
        if face_region.size > 0:
            avg_color = np.mean(face_region, axis=(0, 1))
            hist = cv2.calcHist([face_region], [0, 1, 2], None, [8, 8, 8], 
                                [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            embedding = np.concatenate([avg_color, hist[:20]])  # Simplified embedding
            return embedding
    except Exception as e:
        print(f"[Guardian] Error extracting embedding: {e}")
    
    return None


def learn_face(player=None) -> str:
    """
    Learn the user's face for recognition.
    Call this once to register your face.
    Silent mode - no window display.
    """
    try:
        if player:
            player.ui.write_log("🔍 Face registration starting... Hold still for 3 seconds")
        
        print("[Guardian] 📸 Opening camera for face registration...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "❌ Camera not found or unavailable"
        
        face_embeddings = []
        start_time = time.time()
        frame_count = 0
        
        print("[Guardian] 📸 Capturing face data...")
        while time.time() - start_time < 3:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Find faces using Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
            
            if len(faces) > 0:
                # Use the largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                embedding = _extract_face_embedding(frame, largest_face)
                if embedding is not None:
                    face_embeddings.append(embedding)
                    print(f"[Guardian] 📷 Captured face #{len(face_embeddings)}")
        
        cap.release()
        
        if not face_embeddings:
            msg = "❌ No face detected during registration. Please ensure:\n- Camera is working\n- Your face is visible\n- Good lighting\nTry again!"
            if player:
                player.ui.write_log(msg)
            return msg
        
        # Average the embeddings for better accuracy
        avg_embedding = np.mean(face_embeddings, axis=0)
        
        _save_face_embedding(avg_embedding)
        _log_event("REGISTRATION", f"User face registered with {len(face_embeddings)} samples")
        
        result = f"✅ Face registered successfully! ({len(face_embeddings)} samples captured). Room Guardian is ready!"
        if player:
            player.ui.write_log(result)
        
        return result
    
    except Exception as e:
        error_msg = f"❌ Registration error: {str(e)}"
        print(f"[Guardian] {error_msg}")
        if player:
            player.ui.write_log(error_msg)
        return error_msg


def start_monitoring(player=None) -> str:
    """Start room monitoring when you leave."""
    try:
        _ensure_face_embeddings()
        
        if guardian_state["user_face_embedding"] is None:
            return "❌ Please register your face first. Say 'learn my face'"
        
        guardian_state["active"] = True
        guardian_state["events"] = []
        _log_event("MONITORING_START", "Room Guardian activated")
        
        if player:
            player.ui.write_log("🛡️ Room Guardian activated - monitoring started")
        
        # Start monitoring in background thread
        monitor_thread = threading.Thread(
            target=_monitor_room,
            args=(player,),
            daemon=True
        )
        monitor_thread.start()
        
        return "🛡️ Room Guardian activated. Have a safe trip!"
    
    except Exception as e:
        error_msg = f"❌ Monitoring error: {str(e)}"
        print(f"[Guardian] {error_msg}")
        return error_msg


def stop_monitoring(player=None) -> str:
    """Stop room monitoring."""
    guardian_state["active"] = False
    _log_event("MONITORING_STOP", "Room Guardian deactivated")
    
    if player:
        player.ui.write_log("🛡️ Room Guardian deactivated")
    
    return "🛡️ Room Guardian deactivated"


def _monitor_room(player=None):
    """Background thread that monitors the room."""
    try:
        print("[Guardian] 🎥 Opening camera for monitoring...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Guardian] ❌ Camera not available")
            guardian_state["active"] = False
            return
        
        print("[Guardian] 🎥 Monitoring started")
        detection_cooldown = 0
        
        while guardian_state["active"]:
            ret, frame = cap.read()
            if not ret:
                break
            
            detection_cooldown -= 1
            
            # Find faces using Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
            
            for face_rect in faces:
                embedding = _extract_face_embedding(frame, face_rect)
                
                if embedding is not None and guardian_state["user_face_embedding"] is not None:
                    # Calculate distance between embeddings
                    dist = distance.euclidean(embedding, guardian_state["user_face_embedding"])
                    
                    # Cooldown to avoid spam
                    if detection_cooldown <= 0:
                        # If distance is small, it's the user
                        if dist < 20:  # Threshold for similarity
                            event_msg = "You detected in room"
                            _log_event("USER_DETECTED", event_msg)
                            guardian_state["last_detection"] = {
                                "type": "user",
                                "time": datetime.now()
                            }
                            detection_cooldown = 30  # 30 second cooldown
                        else:
                            # Unknown person
                            event_msg = "Unknown person detected in room!"
                            _log_event("UNKNOWN_DETECTED", event_msg)
                            
                            if player:
                                player.ui.write_log(f"⚠️ {event_msg}")
                            
                            guardian_state["last_detection"] = {
                                "type": "unknown",
                                "time": datetime.now()
                            }
                            detection_cooldown = 60  # 60 second cooldown for unknown
            
            time.sleep(1)  # Check every second
        
        cap.release()
        print("[Guardian] 🎥 Monitoring stopped")
    
    except Exception as e:
        print(f"[Guardian] Monitoring error: {e}")
        guardian_state["active"] = False


def get_welcome_report(player=None) -> str:
    """
    Generate a welcome report when user returns.
    Shows all events that occurred while away.
    """
    try:
        guardian_state["active"] = False
        
        if not guardian_state["events"]:
            report = "✅ Welcome back! No activity detected. Your room is secure."
        else:
            report = "📋 ROOM REPORT - Events while you were away:\n"
            report += "=" * 50 + "\n"
            
            for event in guardian_state["events"]:
                timestamp = event["timestamp"]
                event_type = event["type"]
                details = event["details"]
                
                if "UNKNOWN" in event_type:
                    report += f"⚠️  [{timestamp}] {details}\n"
                elif "USER" in event_type:
                    report += f"✅ [{timestamp}] {details}\n"
                else:
                    report += f"📝 [{timestamp}] {details}\n"
            
            report += "=" * 50
        
        _log_event("REPORT_GENERATED", f"Welcome report with {len(guardian_state['events'])} events")
        
        if player:
            player.ui.write_log(report)
        
        return report
    
    except Exception as e:
        error_msg = f"❌ Report error: {str(e)}"
        print(f"[Guardian] {error_msg}")
        return error_msg


# Main handler for JARVIS
def room_guardian(parameters: dict, player=None, speak=None) -> str:
    """Main entry point for room guardian actions."""
    try:
        action = parameters.get("action", "").lower().strip()
        
        print(f"[Guardian] Action requested: {action}")
        
        if action == "learn_face":
            result = learn_face(player)
        
        elif action == "start":
            result = start_monitoring(player)
        
        elif action == "stop":
            result = stop_monitoring(player)
        
        elif action == "report":
            result = get_welcome_report(player)
        
        elif action == "status":
            status = "🟢 Active" if guardian_state["active"] else "🔴 Inactive"
            events_count = len(guardian_state["events"])
            result = f"Room Guardian Status:\n{status}\nEvents logged: {events_count}"
        
        else:
            result = f"❌ Unknown action: {action}. Use: learn_face, start, stop, report, or status"
        
        print(f"[Guardian] Result: {result[:80]}")
        
        return result
    
    except Exception as e:
        error_msg = f"❌ Room Guardian error: {str(e)}"
        print(f"[Guardian] {error_msg}")
        return error_msg
