# hand-gest-cv | branch: main (mudras)
Small Python/Mediapipe prototype that recognizes a handful of hand mudras and communicative gestures in real time from a webcam feed.

![Image](https://github.com/user-attachments/assets/fbc31b82-97df-4c3c-9816-7f49b95214fa)

## Purpose
Capture a webcam stream, run MediaPipe’s `HandLandmarker` to extract joint positions, compute simple geometric heuristics, and label each hand as one of several predefined mudras or gestures. The overlay provides immediate visual feedback, so the leap from detection to interaction stays in real time.

## Requirements
- Python 3.8 or newer.
- `opencv-python`, `mediapipe`, and `numpy` (see Install).
- A working webcam accessible as device `0` (or change the index in `cv2.VideoCapture`).

## Install
1. Clone/refresh this repository inside your preferred workspace.
2. Create a virtual environment (recommended) and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
   ```
3. Install the dependencies:
   ```bash
   pip install opencv-python mediapipe numpy
   ```
4. Ensure the repository root contains (or downloads) `hand_landmarker.task`. Running the app will download it to the repo root automatically if it is missing.

## Run
```bash
python main.py
```
The script flips the captured image, overlays detected hand landmarks, shows the current mudra label near the wrist, and exits when you press `q`.

## Real-World Usage
- Experimentation with gesture-driven controls for accessibility demos.
- Classroom or workshop tools that provide instant feedback on mudra practice.
- Prototyping signage or kiosk software that reacts to simple thumbs-up/peace/political gestures.

## Architecture
- `main.py` bootstraps MediaPipe’s `HandLandmarker` with `mp_vision.RunningMode.VIDEO` and a downloadable `.task` asset.
- For every detected hand, the script wraps the 21 landmark points in a `HandState` data class that exposes normalized distance and extension calculations.
- A fixed registry of detector functions checks the geometric relationships of fingertips, MCPs, and wrists; the first match determines the gesture label.
- The overlay helpers draw landmarks, skeleton connections, and HUD information before showing the frame with OpenCV’s `imshow`.

## Environment Configuration
- Swap `cv2.VideoCapture(0)` for another index if you need to use a different camera.
- To force a different model asset location, update `MODEL_PATH` (and `MODEL_URL` if you host your own task file) at the top of `main.py`.
- Running under constrained lighting or noisy backgrounds may require tuning the confidence thresholds in `HandLandmarkerOptions`.

## Testing
No automated tests are provided. Exercise the system by running `python main.py`, pointing the webcam at your hand, and verifying that the desired mudras/gestures are recognized and labeled.

## Limitations
- Text labels are limited to the hard-coded gestures in `MUDRA_DETECTORS`; the heuristics can miss variations of the same mudra or misfire on other hand poses.
- The detector supports at most two hands simultaneously because `num_hands=2` in the MediaPipe options.
- Lighting, occlusion, or extremely fast movement can prevent consistent landmark detection, which stops the mudra classifier from firing.
