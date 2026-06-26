"""
Room Guardian - Face recognition security system for JARVIS
Monitors room while you're away and provides updates when you return
"""

import cv2
import pickle
import os
from datetime import datetime
from pathlib import Path
import threading
import time
import face_recognition
from collections import defaultdict

# Get base directory
BASE_DIR = Path(__file__).resolve().parent.parent
GUARDIAN_DIR = BASE_DIR / "data" / "room_guardian"
FACE_ENCODINGS_FILE = GUARDIAN_DIR / "face_encodings.pkl"
EVENTS_LOG_FILE = GUARDIAN_DIR / "events.log"

# Ensure directory exists
GUARDIAN_DIR.mkdir(parents=True, exist_ok=True)

# Global state
guardian_state = {
    "active": False,
    "user_face_encoding": None,
    "events": [],
    "last_detection": None,
}


def _ensure_face_encodings():
    """Load or initialize face encodings."""
    if FACE_ENCODINGS_FILE.exists():
        with open(FACE_ENCODINGS_FILE, "rb") as f:
            guardian_state["user_face_encoding"] = pickle.load(f)
    else:
        guardian_state["user_face_encoding"] = None


def _save_face_encoding(encoding):
    """Save user's face encoding."""
    with open(FACE_ENCODINGS_FILE, "wb") as f:
        pickle.dump(encoding, f)
    guardian_state["user_face_encoding"] = encoding


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
        
        face_encodings = []
        start_time = time.time()
        
        while time.time() - start_time < 3:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Find faces in frame
            face_locations = face_recognition.face_locations(frame)
            frame_encodings = face_recognition.face_encodings(frame, face_locations)
            
            if frame_encodings:
                face_encodings.extend(frame_encodings)
            
            # Show preview (optional)
            cv2.imshow("Face Registration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if not face_encodings:
            return "❌ No face detected. Please try again in good lighting"
        
        # Average the encodings for better accuracy
        import numpy as np
        avg_encoding = np.mean(face_encodings, axis=0)
        
        _save_face_encoding(avg_encoding)
        _log_event("REGISTRATION", "User face registered")
        
        if player:
            player.ui.write_log("✅ Face registered successfully!")
        
        return "✅ Your face has been registered. Room Guardian is ready!"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"


def start_monitoring(player=None) -> str:
    """Start room monitoring when you leave."""
    try:
        _ensure_face_encodings()
        
        if not guardian_state["user_face_encoding"] is not None:
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
    
    while guardian_state["active"]:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Find faces in frame
        face_locations = face_recognition.face_locations(frame)
        frame_encodings = face_recognition.face_encodings(frame, face_locations)
        
        for face_encoding in frame_encodings:
            # Compare with user's face
            matches = face_recognition.compare_faces(
                [guardian_state["user_face_encoding"]],
                face_encoding,
                tolerance=0.6
            )
            
            if matches[0]:
                # User detected
                event_msg = "You detected in room"
                _log_event("USER_DETECTED", event_msg)
                guardian_state["last_detection"] = {
                    "type": "user",
                    "time": datetime.now()
                }
            else:
                # Unknown person detected
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
