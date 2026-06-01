"""Mede a latência fim-a-fim do pipeline de inferência em tempo real.

Uso:
    python utils/medir_latencia.py            # 200 frames da webcam
    python utils/medir_latencia.py --frames 500
    python utils/medir_latencia.py --sem-webcam  # usa frames sinteticos (benchmark puro)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import tensorflow as tf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RAIZ = Path(__file__).parent.parent
MODELO_DIR = RAIZ / "modelo"
RELATORIO_DIR = RAIZ / "relatorio"

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


def criar_frame_sintetico(largura=640, altura=480) -> np.ndarray:
    """Gera frame RGB sintético para benchmark sem webcam."""
    frame = np.random.randint(0, 255, (altura, largura, 3), dtype=np.uint8)
    return frame


def medir(n_frames: int, usar_webcam: bool):
    print("Carregando modelo e MediaPipe...")

    tf.get_logger().setLevel("ERROR")
    modelo = tf.keras.models.load_model(
        str(MODELO_DIR / "modelo_dense_77.h5"), compile=False
    )
    mean = np.load(MODELO_DIR / "scaler_mean.npy")
    std  = np.load(MODELO_DIR / "scaler_std.npy")

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODELO_DIR / "holistic_landmarker.task")
        ),
        running_mode=RunningMode.IMAGE,
        min_face_detection_confidence=0.3,
        min_pose_detection_confidence=0.3,
        min_hand_landmarks_confidence=0.3,
    )
    landmarker = HolisticLandmarker.create_from_options(options)

    cap = None
    if usar_webcam:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print("[AVISO] Webcam nao disponivel. Usando frames sinteticos.")
            cap = None

    print(f"Medindo latencia em {n_frames} frames...")

    t_mediapipe = []
    t_modelo    = []
    t_total     = []

    # Warmup (primeiros 10 frames nao contam)
    frame_warmup = criar_frame_sintetico()
    mp_img_w = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_warmup)
    for _ in range(10):
        landmarker.detect(mp_img_w)
        X_w = np.zeros((1, 30, 258), dtype=np.float32)
        modelo(X_w, training=False).numpy()

    for i in range(n_frames):
        t0 = time.perf_counter()

        # 1. Captura / gera frame
        if cap is not None:
            ret, frame_bgr = cap.read()
            if not ret:
                frame_bgr = criar_frame_sintetico()
        else:
            frame_bgr = criar_frame_sintetico()

        if frame_bgr.shape[1] > 640:
            frame_bgr = cv2.resize(frame_bgr, (640, 480))

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # 2. MediaPipe
        t1 = time.perf_counter()
        result = landmarker.detect(mp_img)
        t2 = time.perf_counter()

        # 3. Extração e normalização
        mao_esq = np.zeros(63, dtype=np.float32)
        mao_dir = np.zeros(63, dtype=np.float32)
        pose    = np.zeros(132, dtype=np.float32)
        if result.left_hand_landmarks:
            mao_esq = np.array([[l.x, l.y, l.z] for l in result.left_hand_landmarks]).flatten()
        if result.right_hand_landmarks:
            mao_dir = np.array([[l.x, l.y, l.z] for l in result.right_hand_landmarks]).flatten()
        if result.pose_landmarks:
            pose = np.array([[l.x, l.y, l.z, l.visibility] for l in result.pose_landmarks]).flatten()

        vetor = np.concatenate([mao_esq, mao_dir, pose])
        X = np.tile(vetor, (30, 1)).reshape(1, 30, 258).astype(np.float32)
        X = (X - mean) / std

        # 4. Inferência
        t3 = time.perf_counter()
        _ = modelo(X, training=False).numpy()
        t4 = time.perf_counter()

        t_mediapipe.append((t2 - t1) * 1000)
        t_modelo.append((t4 - t3) * 1000)
        t_total.append((t4 - t0) * 1000)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_frames} frames — total médio: {np.mean(t_total):.1f} ms")

    landmarker.close()
    if cap:
        cap.release()

    return np.array(t_mediapipe), np.array(t_modelo), np.array(t_total)


def relatorio(t_mp, t_mod, t_tot, n_frames: int):
    META_MS = 50.0
    RELATORIO_DIR.mkdir(exist_ok=True)

    def stats(arr):
        return {
            "min":  arr.min(),
            "max":  arr.max(),
            "mean": arr.mean(),
            "p50":  np.percentile(arr, 50),
            "p95":  np.percentile(arr, 95),
            "p99":  np.percentile(arr, 99),
        }

    s_mp  = stats(t_mp)
    s_mod = stats(t_mod)
    s_tot = stats(t_tot)

    linhas = [
        "=" * 56,
        "  RELATÓRIO DE LATÊNCIA — Tradutor de Libras",
        "=" * 56,
        f"  Frames medidos   : {n_frames}",
        f"  Meta (inferência): {META_MS:.0f} ms",
        "",
        "  Componente         min     p50     p95     max",
        "  " + "-" * 52,
        f"  MediaPipe       {s_mp['min']:6.1f}  {s_mp['p50']:6.1f}  {s_mp['p95']:6.1f}  {s_mp['max']:6.1f} ms",
        f"  Modelo Dense    {s_mod['min']:6.1f}  {s_mod['p50']:6.1f}  {s_mod['p95']:6.1f}  {s_mod['max']:6.1f} ms",
        f"  TOTAL pipeline  {s_tot['min']:6.1f}  {s_tot['p50']:6.1f}  {s_tot['p95']:6.1f}  {s_tot['max']:6.1f} ms",
        "",
        f"  Frames dentro da meta ({META_MS} ms): "
        f"{(t_tot <= META_MS).mean()*100:.1f}%",
        "=" * 56,
    ]
    texto = "\n".join(linhas)
    print("\n" + texto)

    caminho_txt = RELATORIO_DIR / "latencia.txt"
    caminho_txt.write_text(texto + "\n", encoding="utf-8")
    print(f"\nRelatório salvo em {caminho_txt}")

    # Histograma
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, dados, titulo, cor in zip(
        axes,
        [t_mp, t_mod, t_tot],
        ["MediaPipe (ms)", "Modelo Dense (ms)", "Total pipeline (ms)"],
        ["#89b4fa", "#a6e3a1", "#f9e2af"],
    ):
        ax.hist(dados, bins=30, color=cor, edgecolor="black", alpha=0.85)
        ax.axvline(np.percentile(dados, 95), color="red", linestyle="--",
                   label=f"p95={np.percentile(dados, 95):.1f} ms")
        if "Total" in titulo:
            ax.axvline(META_MS, color="orange", linestyle=":",
                       label=f"Meta={META_MS:.0f} ms")
        ax.set_title(titulo)
        ax.set_xlabel("ms")
        ax.set_ylabel("Frames")
        ax.legend(fontsize=8)

    fig.suptitle("Distribuição de Latência — Tradutor de Libras", fontsize=13)
    fig.tight_layout()
    caminho_png = RELATORIO_DIR / "latencia_histograma.png"
    fig.savefig(caminho_png, dpi=120)
    plt.close(fig)
    print(f"Histograma salvo em {caminho_png}")

    return s_tot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames",      type=int, default=200)
    parser.add_argument("--sem-webcam",  action="store_true")
    args = parser.parse_args()

    t_mp, t_mod, t_tot = medir(args.frames, usar_webcam=not args.sem_webcam)
    relatorio(t_mp, t_mod, t_tot, args.frames)


if __name__ == "__main__":
    main()
