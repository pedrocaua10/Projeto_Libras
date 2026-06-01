"""Carrega landmarks .npy, faz split estratificado e salva X/y em dataset/splits/."""

import sys
import json
import yaml
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def carregar_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def carregar_amostras(landmarks_base: Path) -> tuple[list, list]:
    X, y = [], []
    for tipo in ["dinamicos", "estaticos"]:
        pasta_tipo = landmarks_base / tipo
        if not pasta_tipo.exists():
            continue
        for sinal in sorted(pasta_tipo.iterdir()):
            if not sinal.is_dir():
                continue
            for npy_file in sorted(sinal.glob("*.npy")):
                X.append(np.load(npy_file))
                y.append(sinal.name)
    return X, y


def imprimir_distribuicao(y: np.ndarray, classes: list[str]):
    print("\nDistribuicao de classes no treino:")
    indices, contagens = np.unique(y, return_counts=True)
    for idx, cnt in zip(indices, contagens):
        print(f"  {classes[idx]:<20} {cnt:>4} amostras")


def main():
    cfg = carregar_config()
    landmarks_base = Path(cfg["paths"]["landmarks"])
    splits_base = Path(cfg["paths"]["splits"])
    split_cfg = cfg["splits"]
    random_state = split_cfg["random_state"]
    val_ratio = split_cfg["validacao"]
    test_ratio = split_cfg["teste"]

    print("Carregando amostras...")
    X_list, y_list = carregar_amostras(landmarks_base)

    if not X_list:
        print("Nenhuma amostra encontrada em dataset/landmarks/. Execute extrair_landmarks.py primeiro.")
        sys.exit(1)

    X = np.array(X_list, dtype=np.float32)
    le = LabelEncoder()
    y = le.fit_transform(y_list)
    classes = le.classes_.tolist()

    print(f"Total de amostras : {len(X)}")
    print(f"Total de classes  : {len(classes)}")
    print(f"Shape por amostra : {X[0].shape}")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=random_state
    )

    # Ajuste do ratio de validacao sobre o subset restante para garantir 80/10/10 exatos
    val_ratio_ajustado = val_ratio / (1.0 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_ajustado, stratify=y_temp, random_state=random_state
    )

    splits_base.mkdir(parents=True, exist_ok=True)

    np.save(splits_base / "X_train.npy", X_train)
    np.save(splits_base / "X_val.npy", X_val)
    np.save(splits_base / "X_test.npy", X_test)
    np.save(splits_base / "y_train.npy", y_train)
    np.save(splits_base / "y_val.npy", y_val)
    np.save(splits_base / "y_test.npy", y_test)

    with open(splits_base / "classes.json", "w", encoding="utf-8") as f:
        json.dump({str(i): nome for i, nome in enumerate(classes)}, f, ensure_ascii=False, indent=2)

    imprimir_distribuicao(y_train, classes)

    print(f"\nSplits salvos em {splits_base.resolve()}:")
    print(f"  Treino     : {len(X_train)} amostras")
    print(f"  Validacao  : {len(X_val)} amostras")
    print(f"  Teste      : {len(X_test)} amostras")
    print(f"  classes.json: {len(classes)} classes: {classes}")


if __name__ == "__main__":
    main()
