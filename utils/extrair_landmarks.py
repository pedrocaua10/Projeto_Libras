"""Extrai landmarks MediaPipe de vídeos e imagens em dataset/raw/ e salva como .npy."""

import sys
import yaml
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

FORMATOS_VIDEO = {"mp4", "mov", "avi"}
FORMATOS_IMAGEM = {"jpg", "jpeg", "png"}


def carregar_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def extrair_frame_landmarks(results) -> np.ndarray:
    """Concatena landmarks de mao esquerda, mao direita e pose em vetor (258,)."""
    mao_esq = np.zeros(21 * 3)
    mao_dir = np.zeros(21 * 3)
    pose = np.zeros(33 * 4)

    if results.left_hand_landmarks:
        mao_esq = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        ).flatten()

    if results.right_hand_landmarks:
        mao_dir = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        ).flatten()

    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
        ).flatten()

    return np.concatenate([mao_esq, mao_dir, pose])


def processar_imagem(
    caminho: Path, n_frames: int, n_features: int, holistic
) -> np.ndarray | None:
    """Lê imagem estática, extrai landmarks e tila para shape (n_frames, n_features)."""
    img = cv2.imread(str(caminho))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    vetor = extrair_frame_landmarks(holistic.process(rgb))
    return np.tile(vetor, (n_frames, 1)).astype(np.float32)


def processar_video_dinamico(
    caminho: Path, n_frames: int, n_features: int, holistic, fps_alvo: int
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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            frames.append(extrair_frame_landmarks(holistic.process(rgb)))
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
    caminho: Path, n_frames: int, n_features: int, holistic
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

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    vetor = extrair_frame_landmarks(holistic.process(rgb))

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
    complexidade = cfg["mediapipe"]["model_complexity"]
    smooth = cfg["mediapipe"]["smooth_landmarks"]

    arquivos = coletar_arquivos(raw, formatos)
    if not arquivos:
        print("Nenhum arquivo encontrado. Execute importar_dataset.py e validar_dataset.py primeiro.")
        sys.exit(1)

    landmarks_base.mkdir(parents=True, exist_ok=True)
    log_path = landmarks_base / "log_extracao.txt"

    contadores = {"ok": 0, "erros": 0}
    amostras_por_sinal: dict[str, int] = {}

    # static_image_mode=True para imagens, False para vídeos — usamos False para reutilizar o
    # objeto em todo o loop; para imagens estáticas isso é aceitável pois não há tracking entre frames
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=complexidade,
        smooth_landmarks=smooth,
        min_detection_confidence=limiar,
        min_tracking_confidence=limiar,
    )

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
                    seq = processar_imagem(caminho, n_frames, n_features, holistic)
                elif tipo == "estaticos":
                    seq = processar_video_estatico(caminho, n_frames, n_features, holistic)
                else:
                    seq = processar_video_dinamico(caminho, n_frames, n_features, holistic, fps_alvo)

                if seq is None:
                    raise ValueError(f"Nao foi possivel ler: {caminho.name}")

                np.save(pasta_saida / f"sample_{idx_amostra:03d}.npy", seq)
                amostras_por_sinal[chave] = idx_amostra + 1
                contadores["ok"] += 1

            except Exception as e:
                contadores["erros"] += 1
                log_f.write(f"ERRO | {tipo}/{sinal}/{caminho.name} | {e}\n")

    holistic.close()

    print(f"\nExtracao concluida:")
    print(f"  Processados : {contadores['ok']}")
    print(f"  Erros       : {contadores['erros']}")
    print(f"  Log         : {log_path.resolve()}")

    if contadores["erros"] > 0:
        print(f"\n[ATENCAO] Verifique {log_path.name} para detalhes dos erros.")


if __name__ == "__main__":
    main()
