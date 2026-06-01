"""Extrai landmarks MediaPipe de vídeos e imagens em dataset/raw/ e salva como .npy."""

import sys
import urllib.request
import yaml
import cv2
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
MODEL_PATH = Path(__file__).parent.parent / "modelo" / "holistic_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
)

FORMATOS_VIDEO = {"mp4", "mov", "avi"}
FORMATOS_IMAGEM = {"jpg", "jpeg", "png"}


def carregar_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def baixar_modelo():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"Baixando modelo holistic (~50 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Modelo salvo em {MODEL_PATH}")


def extrair_frame_landmarks(result) -> np.ndarray:
    """Extrai landmarks de mao esquerda, mao direita e pose: shape (258,)."""
    mao_esq = np.zeros(21 * 3, dtype=np.float32)
    mao_dir = np.zeros(21 * 3, dtype=np.float32)
    pose = np.zeros(33 * 4, dtype=np.float32)

    if result.left_hand_landmarks:
        mao_esq = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks[0]]
        ).flatten().astype(np.float32)

    if result.right_hand_landmarks:
        mao_dir = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks[0]]
        ).flatten().astype(np.float32)

    if result.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility]
             for lm in result.pose_landmarks[0]]
        ).flatten().astype(np.float32)

    return np.concatenate([mao_esq, mao_dir, pose])


def frame_para_mp_image(frame_bgr: np.ndarray) -> mp.Image:
    # Redimensiona para 640x480 — MediaPipe é otimizado para essa resolução
    # e imagens HEIC convertidas chegam em 3088x1737 (muito lentas sem resize)
    h, w = frame_bgr.shape[:2]
    if w > 640 or h > 480:
        frame_bgr = cv2.resize(frame_bgr, (640, 480))
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def processar_imagem(
    caminho: Path, n_frames: int, n_features: int, landmarker
) -> np.ndarray | None:
    """Lê imagem, extrai landmarks e tila para shape (n_frames, n_features)."""
    img = cv2.imread(str(caminho))
    if img is None:
        return None
    result = landmarker.detect(frame_para_mp_image(img))
    vetor = extrair_frame_landmarks(result)
    return np.tile(vetor, (n_frames, 1)).astype(np.float32)


def processar_video_dinamico(
    caminho: Path, n_frames: int, n_features: int, landmarker, fps_alvo: int
) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(caminho))
    if not cap.isOpened():
        return None

    fps_original = cap.get(cv2.CAP_PROP_FPS) or 30.0
    intervalo = max(1, round(fps_original / fps_alvo))

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % intervalo == 0:
            result = landmarker.detect(frame_para_mp_image(frame))
            frames.append(extrair_frame_landmarks(result))
        idx += 1
    cap.release()

    if not frames:
        return None

    sequencia = np.zeros((n_frames, n_features), dtype=np.float32)
    if len(frames) >= n_frames:
        sequencia[:] = np.array(frames[:n_frames], dtype=np.float32)
    else:
        sequencia[: len(frames)] = np.array(frames, dtype=np.float32)

    return sequencia


def processar_video_estatico(
    caminho: Path, n_frames: int, n_features: int, landmarker
) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(caminho))
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    result = landmarker.detect(frame_para_mp_image(frame))
    vetor = extrair_frame_landmarks(result)
    # Repete o frame para manter shape (n_frames, n_features) igual aos dinamicos
    return np.tile(vetor, (n_frames, 1)).astype(np.float32)


def coletar_arquivos(raw: Path, formatos: set) -> list[tuple[Path, str, str]]:
    arquivos = []
    for tipo in ["dinamicos", "estaticos"]:
        pasta_tipo = raw / tipo
        if not pasta_tipo.exists():
            continue
        for sinal in sorted(pasta_tipo.iterdir()):
            if not sinal.is_dir():
                continue
            for arquivo in sorted(sinal.iterdir()):
                if arquivo.is_file() and arquivo.suffix.lstrip(".").lower() in formatos:
                    arquivos.append((arquivo, tipo, sinal.name))
    return arquivos


def main():
    cfg = carregar_config()
    raw = Path(cfg["paths"]["raw"])
    landmarks_base = Path(cfg["paths"]["landmarks"])
    n_frames = cfg["pipeline"]["n_frames"]
    n_features = cfg["pipeline"]["n_features"]
    fps_alvo = cfg["pipeline"]["fps_alvo"]
    limiar = cfg["pipeline"]["limiar_confianca"]
    formatos = set(cfg["formatos_validos"])

    baixar_modelo()

    arquivos = coletar_arquivos(raw, formatos)
    if not arquivos:
        print("Nenhum arquivo encontrado. Execute importar_dataset.py e validar_dataset.py primeiro.")
        sys.exit(1)

    landmarks_base.mkdir(parents=True, exist_ok=True)
    log_path = landmarks_base / "log_extracao.txt"

    contadores = {"ok": 0, "erros": 0}
    amostras_por_sinal: dict[str, int] = {}

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        min_face_detection_confidence=limiar,
        min_pose_detection_confidence=limiar,
        min_hand_landmarks_confidence=limiar,
    )

    with HolisticLandmarker.create_from_options(options) as landmarker:
        with open(log_path, "w", encoding="utf-8") as log_f:
            log_f.write(f"Extracao iniciada: {datetime.now().isoformat()}\n\n")

            for caminho, tipo, sinal in tqdm(arquivos, desc="Extraindo landmarks", unit="arquivo"):
                chave = f"{tipo}/{sinal}"
                idx_amostra = amostras_por_sinal.get(chave, 0)
                pasta_saida = landmarks_base / tipo / sinal
                pasta_saida.mkdir(parents=True, exist_ok=True)

                try:
                    ext = caminho.suffix.lstrip(".").lower()

                    if ext in FORMATOS_IMAGEM:
                        seq = processar_imagem(caminho, n_frames, n_features, landmarker)
                    elif tipo == "estaticos":
                        seq = processar_video_estatico(caminho, n_frames, n_features, landmarker)
                    else:
                        seq = processar_video_dinamico(caminho, n_frames, n_features, landmarker, fps_alvo)

                    if seq is None:
                        raise ValueError(f"Nao foi possivel ler: {caminho.name}")

                    np.save(pasta_saida / f"sample_{idx_amostra:03d}.npy", seq)
                    amostras_por_sinal[chave] = idx_amostra + 1
                    contadores["ok"] += 1

                except Exception as e:
                    contadores["erros"] += 1
                    log_f.write(f"ERRO | {tipo}/{sinal}/{caminho.name} | {e}\n")

    print(f"\nExtracao concluida:")
    print(f"  Processados : {contadores['ok']}")
    print(f"  Erros       : {contadores['erros']}")
    print(f"  Log         : {log_path.resolve()}")

    if contadores["erros"] > 0:
        print(f"\n[ATENCAO] Verifique {log_path.name} para detalhes dos erros.")


if __name__ == "__main__":
    main()
