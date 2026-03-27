"""
HandGest — Real-time hand mudra recognition via webcam.

Dependencies:
    pip install opencv-python mediapipe numpy

Run:
    python main.py
"""

from __future__ import annotations

import urllib.request
import os
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ──────────────────────────────────────────────
# Model download helper
# ──────────────────────────────────────────────

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"


def ensure_model() -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading hand landmark model to '{MODEL_PATH}' ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

class Landmark(NamedTuple):
    x: float
    y: float
    z: float


# MediaPipe hand connection pairs (landmark indices)
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


@dataclass(frozen=True)
class HandState:
    """Pre-computed per-frame hand features used by every detector."""
    landmarks: list[Landmark]

    THUMB_TIP:  int = 4
    INDEX_TIP:  int = 8
    MIDDLE_TIP: int = 12
    RING_TIP:   int = 16
    PINKY_TIP:  int = 20

    INDEX_MCP:  int = 5
    MIDDLE_MCP: int = 9
    RING_MCP:   int = 13
    PINKY_MCP:  int = 17

    INDEX_PIP:  int = 6
    MIDDLE_PIP: int = 10
    RING_PIP:   int = 14
    PINKY_PIP:  int = 18

    WRIST: int = 0

    def _dist(self, a: int, b: int) -> float:
        la, lb = self.landmarks[a], self.landmarks[b]
        return float(np.linalg.norm([la.x - lb.x, la.y - lb.y, la.z - lb.z]))

    def _hand_scale(self) -> float:
        return self._dist(self.WRIST, self.MIDDLE_MCP) + 1e-6

    def tips_touching(self, tip_a: int, tip_b: int, threshold: float = 0.10) -> bool:
        return self._dist(tip_a, tip_b) / self._hand_scale() < threshold

    def finger_extended(self, tip: int, pip: int, mcp: int) -> bool:
        return self._dist(self.WRIST, tip) > self._dist(self.WRIST, pip) * 1.05

    def finger_curled(self, tip: int, pip: int, mcp: int) -> bool:
        return not self.finger_extended(tip, pip, mcp)

    def tip_below_pip(self, tip: int, pip: int) -> bool:
        return self.landmarks[tip].y > self.landmarks[pip].y

    def thumb_covers(self, tip_a: int, tip_b: int, threshold: float = 0.14) -> bool:
        s = self._hand_scale()
        return (
            self._dist(self.THUMB_TIP, tip_a) / s < threshold and
            self._dist(self.THUMB_TIP, tip_b) / s < threshold
        )


# ──────────────────────────────────────────────
# Individual mudra detectors
# ──────────────────────────────────────────────

def detect_gyan(h: HandState) -> bool:
    return (
        h.tips_touching(h.INDEX_TIP, h.THUMB_TIP) and
        h.finger_extended(h.MIDDLE_TIP, h.MIDDLE_PIP, h.MIDDLE_MCP) and
        h.finger_extended(h.RING_TIP,   h.RING_PIP,   h.RING_MCP)   and
        h.finger_extended(h.PINKY_TIP,  h.PINKY_PIP,  h.PINKY_MCP)
    )


def detect_shunya(h: HandState) -> bool:
    return (
        h.tips_touching(h.MIDDLE_TIP, h.THUMB_TIP) and
        h.finger_extended(h.INDEX_TIP,  h.INDEX_PIP,  h.INDEX_MCP) and
        h.finger_extended(h.RING_TIP,   h.RING_PIP,   h.RING_MCP)  and
        h.finger_extended(h.PINKY_TIP,  h.PINKY_PIP,  h.PINKY_MCP)
    )


def detect_prithvi(h: HandState) -> bool:
    return (
        h.tips_touching(h.RING_TIP, h.THUMB_TIP) and
        h.finger_extended(h.INDEX_TIP,  h.INDEX_PIP,  h.INDEX_MCP)  and
        h.finger_extended(h.MIDDLE_TIP, h.MIDDLE_PIP, h.MIDDLE_MCP) and
        h.finger_extended(h.PINKY_TIP,  h.PINKY_PIP,  h.PINKY_MCP)
    )

def detect_dedo_do_meio(h: HandState) -> bool:
    return (
        h.finger_extended(h.MIDDLE_TIP, h.MIDDLE_PIP, h.MIDDLE_MCP) and
        h.finger_curled(h.INDEX_TIP,  h.INDEX_PIP,  h.INDEX_MCP)   and
        h.finger_curled(h.RING_TIP,   h.RING_PIP,   h.RING_MCP)    and
        h.finger_curled(h.PINKY_TIP,  h.PINKY_PIP,  h.PINKY_MCP)
    )

def detect_joinha(h: HandState) -> bool:
    thumb_up = h.landmarks[h.THUMB_TIP].y < h.landmarks[h.INDEX_MCP].y
    return (
        thumb_up and
        h.finger_curled(h.INDEX_TIP,  h.INDEX_PIP,  h.INDEX_MCP)  and
        h.finger_curled(h.MIDDLE_TIP, h.MIDDLE_PIP, h.MIDDLE_MCP) and
        h.finger_curled(h.RING_TIP,   h.RING_PIP,   h.RING_MCP)   and
        h.finger_curled(h.PINKY_TIP,  h.PINKY_PIP,  h.PINKY_MCP)
    )

def detect_arminha(h: HandState) -> bool:
    """Mão deitada: indicador horizontal, polegar para cima."""
    indicador_horizontal = (
        h.finger_extended(h.INDEX_TIP, h.INDEX_PIP, h.INDEX_MCP) and
        abs(h.landmarks[h.INDEX_TIP].y - h.landmarks[h.INDEX_MCP].y) < 0.10
    )
    polegar_para_cima = (
        h.landmarks[h.THUMB_TIP].y < h.landmarks[h.INDEX_MCP].y - 0.05
    )
    return (
        indicador_horizontal and
        polegar_para_cima    and
        h.finger_curled(h.MIDDLE_TIP, h.MIDDLE_PIP, h.MIDDLE_MCP) and
        h.finger_curled(h.RING_TIP,   h.RING_PIP,   h.RING_MCP)   and
        h.finger_curled(h.PINKY_TIP,  h.PINKY_PIP,  h.PINKY_MCP)
    )


def detect_L_lula(h: HandState) -> bool:
    """Mão em pé: indicador vertical, polegar para o lado."""
    indicador_vertical = (
        h.finger_extended(h.INDEX_TIP, h.INDEX_PIP, h.INDEX_MCP) and
        h.landmarks[h.INDEX_TIP].y < h.landmarks[h.INDEX_MCP].y - 0.08
    )
    polegar_horizontal = (
        abs(h.landmarks[h.THUMB_TIP].x - h.landmarks[h.INDEX_MCP].x) > 0.08 and
        abs(h.landmarks[h.THUMB_TIP].y - h.landmarks[h.INDEX_MCP].y) < 0.08
    )
    return (
        indicador_vertical  and
        polegar_horizontal  and
        h.finger_curled(h.MIDDLE_TIP, h.MIDDLE_PIP, h.MIDDLE_MCP) and
        h.finger_curled(h.RING_TIP,   h.RING_PIP,   h.RING_MCP)   and
        h.finger_curled(h.PINKY_TIP,  h.PINKY_PIP,  h.PINKY_MCP)
    )
# ──────────────────────────────────────────────
# Ordered detector registry
# ──────────────────────────────────────────────

MUDRA_DETECTORS: list[tuple[str, callable]] = [
    ("Gyan Mudra", detect_gyan), 
    ("Shunya Mudra", detect_shunya),
    ("Prithvi Mudra", detect_prithvi),
    ("Que imaturo...", detect_dedo_do_meio),
    ("Joinha", detect_joinha),
    ("É BOLSONARONAE?!!", detect_arminha),
    ("FAZ O L!!", detect_L_lula),
]


def classify_mudra(h: HandState) -> str:
    for name, detector in MUDRA_DETECTORS:
        if detector(h):
            return name
    return "Unknown"


# ──────────────────────────────────────────────
# Overlay helpers
# ──────────────────────────────────────────────

OVERLAY_COLOR    = (255, 255, 255)
MUDRA_COLOR      = (0, 220, 120)
UNKNOWN_COLOR    = (80, 80, 80)
LANDMARK_COLOR   = (0, 200, 255)
CONNECTION_COLOR = (180, 180, 180)


def draw_landmarks(frame: np.ndarray, landmarks: list[Landmark], fh: int, fw: int) -> None:
    pts = [(int(lm.x * fw), int(lm.y * fh)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], CONNECTION_COLOR, 1, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 4, LANDMARK_COLOR, -1, cv2.LINE_AA)


def draw_label(frame: np.ndarray, text: str, position: tuple[int, int]) -> None:
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 1.0
    thickness = 2
    color     = MUDRA_COLOR if text != "Unknown" else UNKNOWN_COLOR
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x - 6, y - th - 8), (x + tw + 6, y + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, hands_detected: int) -> None:
    fh, fw = frame.shape[:2]
    cv2.putText(frame, "HandGest  |  press Q to quit",
                (10, fh - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Hands detected: {hands_detected}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, OVERLAY_COLOR, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────

def main() -> None:
    ensure_model()

    base_options    = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    landmarker_opts = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
        running_mode=mp_vision.RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check that a camera is connected.")

    with mp_vision.HandLandmarker.create_from_options(landmarker_opts) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame      = cv2.flip(frame, 1)
            fh, fw     = frame.shape[:2]
            rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            result         = landmarker.detect_for_video(mp_image, timestamp_ms)
            hands_detected = len(result.hand_landmarks) if result.hand_landmarks else 0

            if result.hand_landmarks:
                for hand_lms in result.hand_landmarks:
                    landmarks = [Landmark(lm.x, lm.y, lm.z) for lm in hand_lms]
                    state     = HandState(landmarks=landmarks)

                    draw_landmarks(frame, landmarks, fh, fw)

                    mudra   = classify_mudra(state)
                    wrist_x = int(landmarks[0].x * fw)
                    wrist_y = int(landmarks[0].y * fh)
                    draw_label(frame, mudra, (max(wrist_x - 60, 10), max(wrist_y - 20, 50)))

            draw_hud(frame, hands_detected)
            cv2.imshow("HandGest", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
