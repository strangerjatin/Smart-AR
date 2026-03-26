# ============================================================
# Android Navigation App for Visually Impaired
# Converted from PC version — works with Kivy + Buildozer
# ============================================================

import cv2
import torch
from ultralytics import YOLO
import time
import queue
from threading import Thread

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from plyer import tts, vibrator   # replaces pyttsx3 + simpleaudio

# ---------------------------
# TTS Setup (Android native)
# ---------------------------
speech_queue = queue.Queue()
speech_thread_running = True

def speech_worker():
    while speech_thread_running:
        try:
            text = speech_queue.get(timeout=0.1)
            if text:
                tts.speak(text)   # uses Android TTS directly
            speech_queue.task_done()
        except queue.Empty:
            continue

speech_thread = Thread(target=speech_worker, daemon=True)
speech_thread.start()

# ---------------------------
# Vibration (replaces beep)
# ---------------------------
def play_beep():
    try:
        vibrator.vibrate(0.4)   # vibrate 0.4 seconds
    except Exception as e:
        print("Vibrate error:", e)

# ---------------------------
# Load YOLO model
# ---------------------------
device = "cpu"   # Android has no CUDA
model = YOLO("yolov8n.pt").to(device)   # nano = fastest on phone
class_names = model.names

# ---------------------------
# Distance Estimation
# ---------------------------
FOCAL_LENGTH = 650
PERSON_REAL_HEIGHT = 165

def estimate_distance(y1, y2):
    pixel_height = abs(y2 - y1)
    if pixel_height < 1:
        return 999
    return round((PERSON_REAL_HEIGHT * FOCAL_LENGTH) / pixel_height, 1)

# ---------------------------
# Regions
# ---------------------------
REGIONS = {
    "left":   (0, 213),
    "center": (214, 426),
    "right":  (427, 640)
}
MIN_BOX_HEIGHT = 80

# ---------------------------
# Kivy UI
# ---------------------------
class NavLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)

        # Camera feed display
        self.img_widget = Image(size_hint=(1, 0.75))
        self.add_widget(self.img_widget)

        # Status label
        self.status = Label(
            text="Tap START to begin",
            font_size='18sp',
            size_hint=(1, 0.1),
            color=(1, 1, 1, 1)
        )
        self.add_widget(self.status)

        # Distance label
        self.dist_label = Label(
            text="Distance: --",
            font_size='16sp',
            size_hint=(1, 0.05)
        )
        self.add_widget(self.dist_label)

        # Buttons row
        btn_row = BoxLayout(size_hint=(1, 0.1), spacing=10)
        self.btn_start = Button(text="START", font_size='16sp',
                                background_color=(0.2, 0.8, 0.2, 1))
        self.btn_start.bind(on_press=self.start_navigation)

        self.btn_stop = Button(text="STOP", font_size='16sp',
                               background_color=(0.8, 0.2, 0.2, 1))
        self.btn_stop.bind(on_press=self.stop_navigation)

        btn_row.add_widget(self.btn_start)
        btn_row.add_widget(self.btn_stop)
        self.add_widget(btn_row)

        # State
        self.cap = None
        self.running = False
        self.last_speak_time = time.time()
        self.last_cmd = ""

    def start_navigation(self, instance):
        if not self.running:
            self.cap = cv2.VideoCapture(0)   # Android back camera
            self.running = True
            self.status.text = "Scanning..."
            tts.speak("Navigation started")
            Clock.schedule_interval(self.update_frame, 1.0 / 10)  # 10 FPS

    def stop_navigation(self, instance):
        self.running = False
        Clock.unschedule(self.update_frame)
        if self.cap:
            self.cap.release()
        self.status.text = "Stopped"
        tts.speak("Navigation stopped")

    def update_frame(self, dt):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        # Run YOLO detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            results = model(rgb, verbose=False)

        detections = results[0].boxes.xyxy.cpu().numpy()
        classes    = results[0].boxes.cls.cpu().numpy()

        blocked = {"left": False, "center": False, "right": False}
        nearest_distance = 9999

        for box, cls in zip(detections, classes):
            x1, y1, x2, y2 = box
            if (y2 - y1) < MIN_BOX_HEIGHT:
                continue
            cx = (x1 + x2) / 2
            distance = estimate_distance(y1, y2)
            nearest_distance = min(nearest_distance, distance)
            for region, (start, end) in REGIONS.items():
                if start <= cx <= end:
                    blocked[region] = True

        # Navigation logic (same as your original)
        if blocked["center"]:
            if nearest_distance < 120:
                cmd = "Stop immediately, obstacle ahead"
                play_beep()
            elif not blocked["left"]:
                cmd = "Move left"
            elif not blocked["right"]:
                cmd = "Move right"
            else:
                cmd = "Stop, all paths blocked"
                play_beep()
        else:
            cmd = "Move forward"

        # Speak only if changed
        if cmd != self.last_cmd and time.time() - self.last_speak_time > 1.5:
            if speech_queue.empty():
                speech_queue.put(cmd)
            self.last_speak_time = time.time()
            self.last_cmd = cmd

        # Update UI labels
        self.status.text = cmd
        dist_text = f"{nearest_distance} cm" if nearest_distance < 9999 else "--"
        self.dist_label.text = f"Nearest: {dist_text}"

        # Show annotated camera frame on screen
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        annotated_flipped = cv2.flip(annotated_rgb, 0)
        buf = annotated_flipped.tobytes()
        tex = Texture.create(
            size=(annotated_flipped.shape[1], annotated_flipped.shape[0]),
            colorfmt='rgb'
        )
        tex.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.img_widget.texture = tex

    def on_stop(self):
        global speech_thread_running
        speech_thread_running = False
        if self.cap:
            self.cap.release()


class NavApp(App):
    def build(self):
        return NavLayout()

    def on_stop(self):
        self.root.on_stop()


NavApp().run()