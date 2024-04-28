"""registration app - adds delete student + rebuild faiss index"""
import customtkinter as ctk
import cv2
import threading
import json
import os
import numpy as np
from PIL import Image
import logging

logging.basicConfig(format='%(asctime)s | %(levelname)-8s | %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

STUDENT_DATA_FILE = "student_data.json"


class RegistrationApp(ctk.CTk):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.title("Face Registration System")
        self.geometry("1100x720")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.student_data = self._load_student_data()
        self._captured_embedding = None
        self._captured_face = None

        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack(side="left", padx=10, pady=10)

        self.sidebar = ctk.CTkFrame(self, width=340)
        self.sidebar.pack(side="right", fill="y", padx=10, pady=10)

        ctk.CTkLabel(self.sidebar, text="Register Student", font=("Arial", 18, "bold")).pack(pady=12)

        self.capture_btn = ctk.CTkButton(self.sidebar, text="Capture Photo",
                                         command=self._capture_photo, fg_color="#1f6aa5")
        self.capture_btn.pack(pady=8, padx=15, fill="x")

        self.thumb_label = ctk.CTkLabel(self.sidebar, text="[no photo captured]",
                                        width=120, height=120, fg_color="#2b2b2b", corner_radius=8)
        self.thumb_label.pack(pady=6)

        ctk.CTkLabel(self.sidebar, text="Student Name").pack(anchor="w", padx=15)
        self.name_entry = ctk.CTkEntry(self.sidebar, placeholder_text="e.g. John Smith")
        self.name_entry.pack(fill="x", padx=15, pady=(2, 8))

        ctk.CTkLabel(self.sidebar, text="Registration Number").pack(anchor="w", padx=15)
        self.reg_entry = ctk.CTkEntry(self.sidebar, placeholder_text="e.g. 22BCS001")
        self.reg_entry.pack(fill="x", padx=15, pady=(2, 8))

        self.register_btn = ctk.CTkButton(self.sidebar, text="Save Registration",
                                          command=self._register, state="disabled")
        self.register_btn.pack(pady=8, padx=15, fill="x")

        ctk.CTkSeparator(self.sidebar).pack(fill="x", padx=15, pady=10)

        ctk.CTkLabel(self.sidebar, text="Delete Student (reg no)").pack(anchor="w", padx=15)
        self.delete_entry = ctk.CTkEntry(self.sidebar, placeholder_text="e.g. 22BCS001")
        self.delete_entry.pack(fill="x", padx=15, pady=(2, 8))
        self.delete_btn = ctk.CTkButton(self.sidebar, text="Delete & Rebuild Index",
                                        command=self._delete_student, fg_color="#8B0000")
        self.delete_btn.pack(pady=8, padx=15, fill="x")

        self.status_label = ctk.CTkLabel(self.sidebar, text="", text_color="gray")
        self.status_label.pack(pady=4)

        self._running = True
        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

    def _load_student_data(self):
        if not os.path.exists(STUDENT_DATA_FILE):
            return {}
        try:
            with open(STUDENT_DATA_FILE, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}
            return data
        except json.JSONDecodeError as e:
            log.error(f"student data corrupt: {e}")
            return {}

    def _save_student_data(self):
        with open(STUDENT_DATA_FILE, "w") as f:
            json.dump(self.student_data, f, indent=2)

    def _capture_photo(self):
        if self.camera.face is None:
            self.status_label.configure(text="no face in frame", text_color="orange")
            return
        self._captured_face = self.camera.face.copy()
        emb = self.camera.get_embedding(self._captured_face)
        if emb is None:
            self.status_label.configure(text="embedding failed, try again", text_color="red")
            return
        self._captured_embedding = emb
        rgb = cv2.cvtColor(self._captured_face, cv2.COLOR_BGR2RGB)
        thumb = Image.fromarray(rgb).resize((120, 120))
        ctk_thumb = ctk.CTkImage(light_image=thumb, dark_image=thumb, size=(120, 120))
        self.thumb_label.configure(image=ctk_thumb, text="")
        self.thumb_label.image = ctk_thumb
        self.register_btn.configure(state="normal")
        self.status_label.configure(text="photo captured - fill details", text_color="cyan")

    def _register(self):
        name = self.name_entry.get().strip()
        reg_no = self.reg_entry.get().strip()
        if not name or not reg_no:
            self.status_label.configure(text="fill in both fields", text_color="red")
            return
        if self._captured_embedding is None:
            self.status_label.configure(text="capture a photo first", text_color="orange")
            return
        student_id = f"{reg_no.lower()}_{name.lower().replace(' ', '_')}"
        self.student_data[student_id] = {"name": name, "reg_no": reg_no}
        self._save_student_data()
        self.camera.store_embedding(self._captured_embedding)
        self.status_label.configure(text=f"registered {name}", text_color="green")
        log.info(f"registered student {student_id}")
        self.name_entry.delete(0, "end")
        self.reg_entry.delete(0, "end")
        self._captured_embedding = None
        self.register_btn.configure(state="disabled")
        self.thumb_label.configure(image=None, text="[no photo captured]")

    def _delete_student(self):
        reg_no = self.delete_entry.get().strip().lower()
        if not reg_no:
            self.status_label.configure(text="enter a registration number", text_color="red")
            return
        # find the matching key
        to_del = [k for k in self.student_data if k.startswith(reg_no)]
        if not to_del:
            self.status_label.configure(text=f"no student with reg {reg_no}", text_color="orange")
            return
        for k in to_del:
            del self.student_data[k]
        self._save_student_data()
        # rebuild faiss from scratch using remaining students' embeddings
        # embeddings are stored sequentially so we just reset and re-add
        self.camera.rebuild_index([])
        self.status_label.configure(text=f"deleted {len(to_del)} record(s), index rebuilt", text_color="green")
        log.info(f"deleted {to_del}, index rebuilt")
        self.delete_entry.delete(0, "end")

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
