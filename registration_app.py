"""registration app - just camera feed for now"""
import customtkinter as ctk
import cv2
import threading
import numpy as np
from PIL import Image
import logging

logging.basicConfig(format='%(asctime)s | %(levelname)-8s | %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class RegistrationApp(ctk.CTk):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.title("Face Registration System")
        self.geometry("900x600")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack(side="left", padx=10, pady=10)

        self._running = True
        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

    def _camera_loop(self):
        while self._running:
            ret, frame = self.camera.cap.read()
            if not ret:
                break
            frame = self.camera.process_frame(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
            self.video_label.configure(image=ctk_img)
            self.video_label.image = ctk_img

    def on_close(self):
        self._running = False
        self.camera.release_resources()
        self.destroy()


if __name__ == "__main__":
    from cam_on import Camera
    cam = Camera()
    app = RegistrationApp(cam)
    app.mainloop()
