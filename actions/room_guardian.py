"""
Room Guardian - Face recognition security system for JARVIS (MediaPipe version)
Monitors room while you're away and provides updates when you return
Uses MediaPipe for better compatibility
"""

import cv2
import pickle
import os
from datetime import datetime
from pathlib import Path
import threading
import time
import numpy as np
import mediapipe as mp
from scipy.spatial import distance

# Get base directory
BASE_DIR = Path(__file__).resolve().parent.parent
GUARDIAN_DIR = BASE_DIR / "data" / "room_guardian"
FACE_EMBEDDINGS_FILE = GUARDIAN_DIR / "face_embeddings.pkl"
EVENTS_LOG_FILE = GUARDIAN_DIR / "events.log"

# Ensure directory exists
GUARDIAN_DIR.mkdir(parents=True, exist_ok=True)

# MediaPipe initialization
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

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
        with open(FACE_EMBEDDINGS_FILE, "rb") as f:
            guardian_state["user_face_embedding"] = pickle.load(f)
    else:
        guardian_state["user_face_embedding"] = None


def _save_face_embedding(embedding):
    """Save user's face embedding."""
    with open(FACE_EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embedding, f)
    guardian_state["user_face_embedding"] = embedding


def _log_event(event_type: str, details: str):
    """Log an event to the events file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    event_entry = f"[{timestamp}] {event_type}: {details}"
    
    guardian_state["events"].append({
        "timestamp": timestamp,
        "type": event_type,
        "details": details
    })
    
    with open(EVENTS_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(event_entry + "\n")
    
    print(f"[Guardian] 📝 {event_entry}")


def _extract_face_embedding(frame, detection):
    """Extract face embedding from a detected face using bounding box."""
    h, w, c = frame.shape
    bbox = detection.location_data.relative_bounding_box
    
    # Convert relative coordinates to pixel coordinates
    x_min = int(bbox.xmin * w)
    y_min = int(bbox.ymin * h)
    x_max = int((bbox.xmin + bbox.width) * w)
    y_max = int((bbox.ymin + bbox.height) * h)
    
    # Ensure coordinates are within bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)
    
    # Extract face region
    face_region = frame[y_min:y_max, x_min:x_max]
    
    # Simple embedding: average RGB values + histogram
    if face_region.size > 0:
        avg_color = np.mean(face_region, axis=(0, 1))
        hist = cv2.calcHist([face_region], [0, 1, 2], None, [8, 8, 8], 
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        embedding = np.concatenate([avg_color, hist[:20]])  # Simplified embedding
        return embedding
    
    return None


def learn_face(player=None) -> str:
    """
    Learn the user's face for recognition.
    Call this once to register your face.
    """
    try:
        if player:
            player.ui.write_log("🔍 Starting face registration... Look at camera for 3 seconds")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "❌ Camera not found"
        
        face_embeddings = []
        start_time = time.time()
        
        with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
            
            while time.time() - start_time < 3:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)
                
                if results.detections:
                    for detection in results.detections:
                        embedding = _extract_face_embedding(frame, detection)
                        if embedding is not None:
                            face_embeddings.append(embedding)
                
                # Show preview
                cv2.imshow("Face Registration", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if not face_embeddings:
            return "❌ No face detected. Please try again in good lighting"
        
        # Average the embeddings for better accuracy
        avg_embedding = np.mean(face_embeddings, axis=0)
        
        _save_face_embedding(avg_embedding)
        _log_event("REGISTRATION", "User face registered")
        
        if player:
            player.ui.write_log("✅ Face registered successfully!")
        
        return "✅ Your face has been registered. Room Guardian is ready!"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"


def start_monitoring(player=None) -> str:
    """Start room monitoring when you leave."""
    try:
        _ensure_face_embeddings()
        
        if guardian_state["user_face_embedding"] is None:
            return "❌ Please register your face first with 'learn my face'"
        
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
        return f"❌ Error: {str(e)}"


def stop_monitoring(player=None) -> str:
    """Stop room monitoring."""
    guardian_state["active"] = False
    _log_event("MONITORING_STOP", "Room Guardian deactivated")
    
    if player:
        player.ui.write_log("🛡️ Room Guardian deactivated")
    
    return "🛡️ Room Guardian deactivated"


def _monitor_room(player=None):
    """Background thread that monitors the room."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Guardian] ❌ Camera not available")
        return
    
    print("[Guardian] 🎥 Monitoring started")
    
    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
        
        while guardian_state["active"]:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            
            if results.detections:
                for detection in results.detections:
                    embedding = _extract_face_embedding(frame, detection)
                    
                    if embedding is not None and guardian_state["user_face_embedding"] is not None:
                        # Calculate distance between embeddings
                        dist = distance.euclidean(embedding, guardian_state["user_face_embedding"])
                        
                        # If distance is small, it's the user
                        if dist < 20:  # Threshold for similarity
                            event_msg = "You detected in room"
                            _log_event("USER_DETECTED", event_msg)
                            guardian_state["last_detection"] = {
                                "type": "user",
                                "time": datetime.now()
                            }
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
            
            time.sleep(1)  # Check every second
    
    cap.release()
    print("[Guardian] 🎥 Monitoring stopped")


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
        return f"❌ Error generating report: {str(e)}"


# Main handler for JARVIS
def room_guardian(parameters: dict, player=None, speak=None) -> str:
    """Main entry point for room guardian actions."""
    action = parameters.get("action", "").lower()
    
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
        result = "❌ Unknown action. Use: learn_face, start, stop, report, status"
    
    if speak and "❌" not in result:
        speak(result)
    
    return result
