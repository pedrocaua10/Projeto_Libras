"""Treina modelo LSTM ou GRU para classificacao de sinais em Libras.

Uso:
    python modelo/treinar.py              # LSTM (padrao)
    python modelo/treinar.py --arq gru    # GRU
    python modelo/treinar.py --epochs 200 --batch 64
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suprime logs verbosos do TF

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

RAIZ = Path(__file__).parent.parent
SPLITS = RAIZ / "dataset" / "splits"
MODELO_DIR = Path(__file__).parent


def carregar_dados():
    X_train = np.load(SPLITS / "X_train.npy")
    X_val   = np.load(SPLITS / "X_val.npy")
    X_test  = np.load(SPLITS / "X_test.npy")
    y_train = np.load(SPLITS / "y_train.npy")
    y_val   = np.load(SPLITS / "y_val.npy")
    y_test  = np.load(SPLITS / "y_test.npy")
    with open(SPLITS / "classes.json", encoding="utf-8") as f:
        classes_dict = json.load(f)
    nomes = [classes_dict[str(i)] for i in range(len(classes_dict))]
    return X_train, X_val, X_test, y_train, y_val, y_test, nomes


def normalizar(X_train, X_val, X_test):
    """Z-score por feature usando estatísticas do treino."""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    np.save(MODELO_DIR / "scaler_mean.npy", mean)
    np.save(MODELO_DIR / "scaler_std.npy",  std)
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std


def filtrar_classes(X_train, X_val, X_test, y_train, y_val, y_test, nomes, min_n):
    """Remove classes com menos de min_n amostras no conjunto de treino."""
    classes, contagens = np.unique(y_train, return_counts=True)
    classes_ok = classes[contagens >= min_n]
    print(f"  Filtrando: {len(classes_ok)}/{len(classes)} classes com >= {min_n} amostras no treino")

    def filtrar(X, y):
        mask = np.isin(y, classes_ok)
        return X[mask], y[mask]

    X_train, y_train = filtrar(X_train, y_train)
    X_val,   y_val   = filtrar(X_val,   y_val)
    X_test,  y_test  = filtrar(X_test,  y_test)

    # Remapeia labels para range contíguo 0..N-1
    mapa = {int(old): new for new, old in enumerate(sorted(classes_ok))}
    y_train = np.array([mapa[y] for y in y_train], dtype=np.int64)
    y_val   = np.array([mapa[y] for y in y_val],   dtype=np.int64)
    y_test  = np.array([mapa[y] for y in y_test],  dtype=np.int64)
    nomes_f = [nomes[int(c)] for c in sorted(classes_ok)]
    return X_train, X_val, X_test, y_train, y_val, y_test, nomes_f


def aumentar_dados(X, y, min_amostras: int = 30, seed: int = 42):
    """Upsampling de classes minoritárias com ruído gaussiano leve."""
    rng = np.random.default_rng(seed)
    classes, contagens = np.unique(y, return_counts=True)
    X_extra, y_extra = [], []

    for cls, n in zip(classes, contagens):
        if n >= min_amostras:
            continue
        idx = np.where(y == cls)[0]
        faltam = min_amostras - n
        while faltam > 0:
            escolha = idx[rng.choice(len(idx), min(faltam, len(idx)), replace=False)]
            ruido = rng.normal(0, 0.015, X[escolha].shape).astype(np.float32)
            X_extra.append(X[escolha] + ruido)
            y_extra.append(y[escolha])
            faltam -= len(escolha)

    if not X_extra:
        return X, y

    X_out = np.concatenate([X] + X_extra)
    y_out = np.concatenate([y] + y_extra)
    perm  = rng.permutation(len(X_out))
    return X_out[perm], y_out[perm]


def construir_modelo(arq: str, n_frames: int, n_features: int, n_classes: int):
    Camada = LSTM if arq == "lstm" else GRU
    model = Sequential(
        [
            Camada(128, return_sequences=True, input_shape=(n_frames, n_features)),
            Dropout(0.3),
            Camada(64),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(n_classes, activation="softmax"),
        ],
        name=f"tradutor_{arq}",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plotar_historico(hist, caminho: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(hist.history["accuracy"],     label="treino")
    ax1.plot(hist.history["val_accuracy"], label="validacao")
    ax1.set_title("Acuracia")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax2.plot(hist.history["loss"],     label="treino")
    ax2.plot(hist.history["val_loss"], label="validacao")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(caminho, dpi=100)
    plt.close(fig)
    print(f"Curvas salvas em {caminho.name}")


def plotar_matriz_confusao(y_true, y_pred, nomes: list, caminho: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(22, 20))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(nomes)))
    ax.set_yticks(range(len(nomes)))
    ax.set_xticklabels(nomes, rotation=90, fontsize=6)
    ax.set_yticklabels(nomes, fontsize=6)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusao")
    fig.tight_layout()
    fig.savefig(caminho, dpi=100)
    plt.close(fig)
    print(f"Matriz de confusao salva em {caminho.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arq",     choices=["lstm", "gru"], default="lstm",
                        help="Arquitetura recorrente (padrao: lstm)")
    parser.add_argument("--epochs",  type=int,   default=100)
    parser.add_argument("--batch",   type=int,   default=32)
    parser.add_argument("--no-aug",  action="store_true",
                        help="Desativa data augmentation")
    parser.add_argument("--min-amostras-reais", type=int, default=0,
                        help="Remove classes com menos de N amostras reais no treino "
                             "(ex: 20 mantém só as letras estáticas com dados suficientes)")
    args = parser.parse_args()

    tf.random.set_seed(42)
    np.random.seed(42)

    print("Carregando dados...")
    X_train, X_val, X_test, y_train, y_val, y_test, nomes = carregar_dados()
    n_classes  = len(nomes)
    n_frames   = X_train.shape[1]
    n_features = X_train.shape[2]
    print(f"  Treino={len(X_train)} | Val={len(X_val)} | Teste={len(X_test)}")
    print(f"  {n_classes} classes | {n_frames} frames | {n_features} features/frame")

    if args.min_amostras_reais > 0:
        print(f"Filtrando classes com < {args.min_amostras_reais} amostras reais...")
        X_train, X_val, X_test, y_train, y_val, y_test, nomes = filtrar_classes(
            X_train, X_val, X_test, y_train, y_val, y_test, nomes, args.min_amostras_reais
        )
        n_classes = len(nomes)
        print(f"  {len(X_train)} treino | {len(X_val)} val | {len(X_test)} teste | {n_classes} classes")

    print("Normalizando (z-score)...")
    X_train, X_val, X_test = normalizar(X_train, X_val, X_test)

    if not args.no_aug:
        print("Data augmentation (upsampling classes < 30 amostras)...")
        X_train, y_train = aumentar_dados(X_train, y_train, min_amostras=30)
        print(f"  Treino apos augmentation: {len(X_train)} amostras")

    pesos = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    pesos_dict = dict(enumerate(pesos))

    print(f"\nConstruindo modelo {args.arq.upper()}...")
    modelo = construir_modelo(args.arq, n_frames, n_features, n_classes)
    modelo.summary()

    checkpoint = MODELO_DIR / f"modelo_{args.arq}_melhor.keras"
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor="val_accuracy",
                      verbose=1),
        ModelCheckpoint(str(checkpoint), save_best_only=True, monitor="val_accuracy",
                        verbose=0),
        ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-5, monitor="val_loss",
                          verbose=1),
    ]

    print(f"\nTreinando {args.arq.upper()} por ate {args.epochs} epochs "
          f"(batch={args.batch})...")
    hist = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=pesos_dict,
        callbacks=callbacks,
        verbose=1,
    )

    loss_teste, acc_teste = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"\nResultado no conjunto de teste:")
    print(f"  Acuracia : {acc_teste*100:.1f}%")
    print(f"  Loss     : {loss_teste:.4f}")

    acc_pct = int(acc_teste * 100)
    caminho_final = MODELO_DIR / f"modelo_{args.arq}_{acc_pct}.h5"
    modelo.save(str(caminho_final))
    print(f"  Modelo salvo: {caminho_final.name}")

    y_pred = modelo.predict(X_test, verbose=0).argmax(axis=1)
    print("\nRelatorio por classe:")
    print(classification_report(
        y_test, y_pred,
        target_names=nomes,
        labels=list(range(n_classes)),
        zero_division=0,
    ))

    plotar_historico(hist, MODELO_DIR / f"historico_{args.arq}.png")
    plotar_matriz_confusao(y_test, y_pred, nomes,
                           MODELO_DIR / f"matriz_confusao_{args.arq}.png")


if __name__ == "__main__":
    main()
