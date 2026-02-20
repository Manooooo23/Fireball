import cv2
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import mediapipe.python.solutions.face_detection as mp_face
import numpy as np
import random
import math
import threading
import wave
import os
import pygame

# ============================================================
# CHARGER L'IMAGE DU SORCIER (supprime le fond damier)
# ============================================================

WIZARD_PATH = os.path.join(os.path.dirname(__file__), "assets", "wizard.jpg")


def load_wizard_with_alpha(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"ERREUR: image introuvable: {path}")
        return None, None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_bg = cv2.inRange(hsv, (0, 0, 180), (180, 30, 255))
    mask_fg = cv2.bitwise_not(mask_bg)
    mask_fg = cv2.GaussianBlur(mask_fg, (5, 5), 0)
    return img, mask_fg


WIZARD_IMG, WIZARD_MASK = load_wizard_with_alpha(WIZARD_PATH)


def overlay_wizard(frame, x, y, w_size):
    if WIZARD_IMG is None:
        return
    aspect = WIZARD_IMG.shape[0] / WIZARD_IMG.shape[1]
    new_w = w_size
    new_h = int(new_w * aspect)
    resized = cv2.resize(WIZARD_IMG, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(WIZARD_MASK, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x1 = x - new_w // 2
    y1 = y - new_h // 2
    x2 = x1 + new_w
    y2 = y1 + new_h
    fh, fw = frame.shape[:2]
    src_x1, src_y1 = max(0, -x1), max(0, -y1)
    src_x2 = new_w - max(0, x2 - fw)
    src_y2 = new_h - max(0, y2 - fh)
    dst_x1, dst_y1 = max(0, x1), max(0, y1)
    dst_x2, dst_y2 = min(fw, x2), min(fh, y2)
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return
    roi = frame[dst_y1:dst_y2, dst_x1:dst_x2]
    src = resized[src_y1:src_y2, src_x1:src_x2]
    m = mask[src_y1:src_y2, src_x1:src_x2]
    m_f = m.astype(np.float32) / 255.0
    m_3 = cv2.merge([m_f, m_f, m_f])
    blended = (src.astype(np.float32) * m_3 + roi.astype(np.float32) * (1 - m_3)).astype(np.uint8)
    frame[dst_y1:dst_y2, dst_x1:dst_x2] = blended


# ============================================================
# SON : volume + saturation selon la puissance
# ============================================================

SOUND_PATH = os.path.join(os.path.dirname(__file__), "assets", "fireball.wav")

# Charger les samples bruts du wav
def load_wav_samples(path):
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return samples, n_channels, sampwidth, framerate

WAV_SAMPLES, WAV_CHANNELS, WAV_SAMPWIDTH, WAV_RATE = load_wav_samples(SOUND_PATH)

pygame.mixer.init(frequency=WAV_RATE, size=-16, channels=WAV_CHANNELS)


def make_distorted_sound(power):
    """
    power 0→0.3  : timide   → volume bas, pas de distorsion
    power 0.3→1  : normal   → volume moyen
    power 1→2    : saturé   → gain élevé + clipping = voix saturée/criée
    """
    # Volume : de 0.15 (timide) à 1.0 (max)
    volume = np.clip(0.15 + power * 0.45, 0.15, 1.0)

    # Gain : de 0.5 (doux) à 5.0 (très saturé/clipping)
    gain = 0.5 + power * 2.25

    processed = WAV_SAMPLES * gain
    # Clipping → crée la saturation
    processed = np.clip(processed, -32768, 32767).astype(np.int16)

    sound = pygame.mixer.Sound(buffer=processed.tobytes())
    sound.set_volume(volume)
    return sound


def play_fireball_sound(power):
    sound = make_distorted_sound(power)
    sound.play()


def play_sound_async(power):
    threading.Thread(target=play_fireball_sound, args=(power,), daemon=True).start()


# ============================================================
# BOULE LANCÉE (vole vers la caméra puis disparaît)
# ============================================================

class ProjectedFireball:
    def __init__(self, x, y, power):
        self.x = float(x)
        self.y = float(y)
        self.power = power
        self.radius = 12 + 30 * math.log1p(power * 4)
        self.life = 1.0
        self.grow_speed = 8 + 12 * power
        self.fade_started = False

    def update(self):
        self.radius += self.grow_speed
        self.grow_speed *= 1.04
        if self.radius > 150:
            self.fade_started = True
        if self.fade_started:
            self.life -= 0.06

    def is_alive(self):
        return self.life > 0

    def draw(self, frame):
        cx, cy = int(self.x), int(self.y)
        r = int(self.radius)
        l = max(0.0, self.life)
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r + 25, (0, int(15 * l), int(80 * l)), -1)
        cv2.circle(overlay, (cx, cy), r + 12, (0, int(60 * l), int(180 * l)), -1)
        cv2.circle(overlay, (cx, cy), r, (0, int(120 * l), int(255 * l)), -1)
        cv2.circle(overlay, (cx, cy), int(r * 0.6), (0, int(200 * l), int(255 * l)), -1)
        cv2.circle(overlay, (cx, cy), int(r * 0.25), (int(150 * l), int(240 * l), 255), -1)
        alpha = 0.75 * l
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ============================================================
# BOULE DE CHARGE (pendant que le poing est fermé)
# ============================================================

class ChargeBall:
    def __init__(self):
        self.charge = 0
        self.orbiters = []

    def update(self, cx, cy):
        self.charge += 1
        radius = self._radius()
        if self.charge % 3 == 0:
            angle = random.uniform(0, 2 * math.pi)
            self.orbiters.append({
                'angle': angle,
                'dist': radius + random.uniform(5, 15),
                'speed': random.uniform(2, 5) * random.choice([-1, 1]),
                'size': random.randint(2, 5),
                'life': random.uniform(0.5, 1.0),
            })
        alive = []
        for o in self.orbiters:
            o['angle'] += o['speed'] * 0.05
            o['life'] -= 0.015
            if o['life'] > 0:
                alive.append(o)
        self.orbiters = alive

    def _radius(self):
        return int(12 + 30 * math.log1p(self.charge / 15))

    def power(self):
        return min(2.0, self.charge / 60)

    def draw(self, frame, cx, cy):
        radius = self._radius()
        t = cv2.getTickCount() / cv2.getTickFrequency()
        pulse = int(math.sin(t * 8) * (3 + radius * 0.08))
        r = radius + pulse
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r + 20, (0, 15, 80), -1)
        cv2.circle(overlay, (cx, cy), r + 10, (0, 60, 180), -1)
        cv2.circle(overlay, (cx, cy), r, (0, 120, 255), -1)
        cv2.circle(overlay, (cx, cy), int(r * 0.65), (0, 200, 255), -1)
        cv2.circle(overlay, (cx, cy), int(r * 0.3), (150, 240, 255), -1)
        for o in self.orbiters:
            ox = int(cx + math.cos(o['angle']) * o['dist'])
            oy = int(cy + math.sin(o['angle']) * o['dist'])
            cv2.circle(overlay, (ox, oy), o['size'], (0, int(160 * o['life']), int(255 * o['life'])), -1)
        alpha = min(0.85, 0.5 + self.power() * 0.2)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        pwr = int(self.power() * 100)
        cv2.putText(frame, f"CHARGE: {pwr}%", (cx - 50, cy - r - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    def reset(self):
        self.charge = 0
        self.orbiters.clear()


# ============================================================
# DÉTECTION MAIN
# ============================================================

def is_fist(hand_landmarks):
    tips_ids = [8, 12, 16, 20]
    pip_ids  = [6, 10, 14, 18]
    closed = sum(
        1 for tip, pip in zip(tips_ids, pip_ids)
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y
    )
    return closed >= 3


def get_palm_center(hand_landmarks, h, w):
    wrist   = hand_landmarks.landmark[0]
    mid_mcp = hand_landmarks.landmark[9]
    cx = int((wrist.x + mid_mcp.x) / 2 * w)
    cy = int((wrist.y + mid_mcp.y) / 2 * h)
    return cx, cy


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================

def run_fireball():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    fireballs = []
    charge_balls = {}
    was_fist = {}
    show_wizard = False
    wizard_timer = 0

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands, mp_face.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    ) as face_detection:

        while cam.isOpened():
            success, frame = cam.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            detected = {}

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    fist = is_fist(hand_landmarks)
                    detected[idx] = fist
                    cx, cy = get_palm_center(hand_landmarks, h, w)

                    if fist:
                        if idx not in charge_balls:
                            charge_balls[idx] = ChargeBall()
                        charge_balls[idx].update(cx, cy)
                        charge_balls[idx].draw(frame, cx, cy)
                    else:
                        if was_fist.get(idx, False) and idx in charge_balls:
                            power = charge_balls[idx].power()

                            # Son avec saturation selon la puissance
                            play_sound_async(power)

                            fireballs.append(ProjectedFireball(cx, cy, power))

                            # Sorcier pendant 2 secondes
                            show_wizard = True
                            wizard_timer = 60

                            del charge_balls[idx]
                        elif idx in charge_balls:
                            charge_balls[idx].reset()
                            del charge_balls[idx]

                    was_fist[idx] = fist

            for idx in list(was_fist.keys()):
                if idx not in detected:
                    was_fist[idx] = False
                    if idx in charge_balls:
                        del charge_balls[idx]

            # Dessiner les boules de feu en vol
            alive = []
            for fb in fireballs:
                fb.update()
                if fb.is_alive():
                    fb.draw(frame)
                    alive.append(fb)
            fireballs = alive

            # Tête du sorcier sur le visage pendant 2 sec après le lancer
            if show_wizard:
                wizard_timer -= 1
                if wizard_timer <= 0:
                    show_wizard = False
                face_results = face_detection.process(frame_rgb)
                if face_results.detections:
                    for detection in face_results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        face_x = int((bbox.xmin + bbox.width / 2) * w)
                        face_y = int((bbox.ymin + bbox.height / 2) * h)
                        face_size = int(bbox.width * w * 1.8)
                        overlay_wizard(frame, face_x, face_y, face_size)

            # HUD
            cv2.putText(frame, "Poing ferme = charge | Ouvre = LANCE | Q = quitter",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow("Boule de Feu", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cam.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()


if __name__ == "__main__":
    run_fireball()
