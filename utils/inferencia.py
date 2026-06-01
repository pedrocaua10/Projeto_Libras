"""Pipeline de inferência em tempo real: frame BGR → sinal Libras + confiança."""

import json
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from pathlib import Path
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

RAIZ = Path(__file__).parent.parent
MODELO_DIR = RAIZ / "modelo"

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

N_FRAMES = 30
N_FEATURES = 258


class Pipeline:
    def __init__(
        self,
        modelo_path: str | None = None,
        limiar_confianca: float = 0.3,
        estabilizacao: int = 5,
    ):
        self.estabilizacao = estabilizacao
        self._buffer: deque = deque(maxlen=N_FRAMES)
        self._historico_pred: deque = deque(maxlen=estabilizacao)
        self._ultimo_falado: str | None = None

        modelo_path = modelo_path or str(MODELO_DIR / "modelo_dense_77.h5")
        self.modelo = tf.keras.models.load_model(modelo_path, compile=False)

        self.mean = np.load(MODELO_DIR / "scaler_mean.npy")
        self.std  = np.load(MODELO_DIR / "scaler_std.npy")

        with open(MODELO_DIR / "classes_modelo.json", encoding="utf-8") as f:
            self.classes: list[str] = json.load(f)

        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(MODELO_DIR / "holistic_landmarker.task")
            ),
            running_mode=RunningMode.IMAGE,
            min_face_detection_confidence=limiar_confianca,
            min_pose_detection_confidence=limiar_confianca,
            min_hand_landmarks_confidence=limiar_confianca,
        )
        self.landmarker = HolisticLandmarker.create_from_options(options)

    def _extrair_landmarks(self, frame_bgr: np.ndarray):
        """Retorna (vetor_258, resultado_mp) para o frame dado."""
        h, w = frame_bgr.shape[:2]
        if w > 640 or h > 480:
            frame_bgr = cv2.resize(frame_bgr, (640, 480))
            h, w = 480, 640

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_img)

        mao_esq = np.zeros(21 * 3, dtype=np.float32)
        mao_dir = np.zeros(21 * 3, dtype=np.float32)
        pose    = np.zeros(33 * 4, dtype=np.float32)

        if result.left_hand_landmarks:
            mao_esq = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks]
            ).flatten().astype(np.float32)

        if result.right_hand_landmarks:
            mao_dir = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks]
            ).flatten().astype(np.float32)

        if result.pose_landmarks:
            pose = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks]
            ).flatten().astype(np.float32)

        vetor = np.concatenate([mao_esq, mao_dir, pose])
        return vetor, result, (w, h)

    def processar(self, frame_bgr: np.ndarray) -> dict:
        """Processa um frame e retorna predição, confiança e pontos para desenho."""
        vetor, resultado_mp, (w, h) = self._extrair_landmarks(frame_bgr)
        self._buffer.append(vetor)

        # Monta janela de 30 frames (padding com zeros se buffer incompleto)
        buf = list(self._buffer)
        if len(buf) < N_FRAMES:
            pad = [np.zeros(N_FEATURES, dtype=np.float32)] * (N_FRAMES - len(buf))
            buf = pad + buf
        X = np.array(buf, dtype=np.float32).reshape(1, N_FRAMES, N_FEATURES)
        X = (X - self.mean) / self.std

        probs = self.modelo.predict(X, verbose=0)[0]
        idx_top = int(probs.argmax())
        confianca = float(probs[idx_top])
        classe = self.classes[idx_top]

        top_k = [
            (self.classes[i], float(probs[i]))
            for i in probs.argsort()[::-1][:5]
        ]

        self._historico_pred.append(classe)

        # Predição estável: mesma classe em todas as últimas N frames
        pred_estavel = None
        if len(self._historico_pred) == self.estabilizacao:
            if len(set(self._historico_pred)) == 1 and classe != "vazio":
                pred_estavel = classe

        # Pontos de desenho (coordenadas pixel do frame ORIGINAL)
        orig_h, orig_w = frame_bgr.shape[:2]
        pontos_pose = []
        pontos_maos = []
        if resultado_mp.pose_landmarks:
            pontos_pose = [
                (int(lm.x * orig_w), int(lm.y * orig_h))
                for lm in resultado_mp.pose_landmarks
            ]
        for lado in [resultado_mp.left_hand_landmarks, resultado_mp.right_hand_landmarks]:
            if lado:
                pontos_maos.extend([
                    (int(lm.x * orig_w), int(lm.y * orig_h))
                    for lm in lado
                ])

        return {
            "classe":        classe,
            "confianca":     confianca,
            "top_k":         top_k,
            "pred_estavel":  pred_estavel,
            "pontos_pose":   pontos_pose,
            "pontos_maos":   pontos_maos,
            "tem_landmarks": bool(resultado_mp.pose_landmarks
                                  or resultado_mp.left_hand_landmarks
                                  or resultado_mp.right_hand_landmarks),
        }

    def fechar(self):
        self.landmarker.close()
