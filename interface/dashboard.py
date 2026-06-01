"""Dashboard Tkinter para testar o modelo de reconhecimento de Libras em tempo real.

Uso:
    python interface/dashboard.py
    python interface/dashboard.py --modelo modelo/modelo_dense_78.h5
"""

import os
import sys
import argparse
import threading
import tkinter as tk
from tkinter import ttk, font as tkfont
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.inferencia import Pipeline

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Paleta de cores ────────────────────────────────────────────────────────────
COR_FUNDO      = "#1e1e2e"
COR_PAINEL     = "#2a2a3e"
COR_DESTAQUE   = "#89b4fa"
COR_VERDE      = "#a6e3a1"
COR_AMARELO    = "#f9e2af"
COR_VERMELHO   = "#f38ba8"
COR_TEXTO      = "#cdd6f4"
COR_SUBTEXT    = "#6c7086"
COR_BARRA      = "#313244"

LARGURA_VIDEO = 640
ALTURA_VIDEO  = 480
INTERVALO_MS  = 40  # ~25 fps


class TTS:
    """Síntese de voz em thread separada para não travar a UI."""

    def __init__(self):
        self._fila: deque = deque()
        self._lock = threading.Lock()
        self._ativo = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def falar(self, texto: str):
        with self._lock:
            self._fila.clear()  # descarta fila — fala o mais recente
            self._fila.append(texto)

    def _loop(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            for voz in engine.getProperty("voices"):
                if "pt" in voz.id.lower() or "brazil" in voz.id.lower():
                    engine.setProperty("voice", voz.id)
                    break
        except Exception:
            engine = None

        while True:
            with self._lock:
                texto = self._fila.popleft() if self._fila else None
            if texto and engine:
                try:
                    engine.say(texto)
                    engine.runAndWait()
                except Exception:
                    pass
            else:
                threading.Event().wait(0.05)


class Dashboard:
    def __init__(self, modelo_path: str | None = None):
        self.pipeline = Pipeline(modelo_path=modelo_path)
        self.tts = TTS()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  LARGURA_VIDEO)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA_VIDEO)

        self.tts_ligado = True
        self.mostrar_landmarks = True
        self.historico: deque = deque(maxlen=20)
        self.ultimo_falado: str | None = None
        self._rodando = True

        self._construir_ui()

    # ── Construção da UI ───────────────────────────────────────────────────────

    def _construir_ui(self):
        self.root = tk.Tk()
        self.root.title("Tradutor de Libras — Dashboard de Teste")
        self.root.configure(bg=COR_FUNDO)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._ao_fechar)

        fonte_titulo  = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        fonte_sign    = tkfont.Font(family="Segoe UI", size=48, weight="bold")
        fonte_pct     = tkfont.Font(family="Segoe UI", size=18)
        fonte_normal  = tkfont.Font(family="Segoe UI", size=10)
        fonte_small   = tkfont.Font(family="Segoe UI", size=9)

        # ── Frame principal ────────────────────────────────────────────────────
        main = tk.Frame(self.root, bg=COR_FUNDO, padx=12, pady=12)
        main.pack()

        # ── Título ─────────────────────────────────────────────────────────────
        tk.Label(
            main, text="Tradutor de Libras  •  Modo Teste",
            font=fonte_titulo, bg=COR_FUNDO, fg=COR_DESTAQUE
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # ── Coluna esquerda: vídeo ─────────────────────────────────────────────
        col_video = tk.Frame(main, bg=COR_FUNDO)
        col_video.grid(row=1, column=0, padx=(0, 12))

        self.canvas_video = tk.Canvas(
            col_video,
            width=LARGURA_VIDEO, height=ALTURA_VIDEO,
            bg="#000000", highlightthickness=2,
            highlightbackground=COR_PAINEL,
        )
        self.canvas_video.pack()

        # Barra de status da câmera
        self.lbl_status = tk.Label(
            col_video, text="⬤  Câmera ativa",
            font=fonte_small, bg=COR_FUNDO, fg=COR_VERDE
        )
        self.lbl_status.pack(anchor="w", pady=(4, 0))

        # ── Coluna direita: predição ───────────────────────────────────────────
        col_pred = tk.Frame(main, bg=COR_FUNDO)
        col_pred.grid(row=1, column=1, sticky="n", padx=(0, 0))

        # — Painel: sinal atual ————————————────────────────────────————————————
        painel_sinal = tk.Frame(col_pred, bg=COR_PAINEL, padx=16, pady=16)
        painel_sinal.pack(fill="x", pady=(0, 10))

        tk.Label(painel_sinal, text="SINAL DETECTADO",
                 font=fonte_small, bg=COR_PAINEL, fg=COR_SUBTEXT).pack(anchor="w")

        self.lbl_sinal = tk.Label(
            painel_sinal, text="—",
            font=fonte_sign, bg=COR_PAINEL, fg=COR_TEXTO, width=6
        )
        self.lbl_sinal.pack()

        self.lbl_confianca = tk.Label(
            painel_sinal, text="0%",
            font=fonte_pct, bg=COR_PAINEL, fg=COR_SUBTEXT
        )
        self.lbl_confianca.pack()

        # Barra de confiança principal
        self.barra_conf = ttk.Progressbar(
            painel_sinal, length=260, mode="determinate", maximum=100
        )
        self.barra_conf.pack(fill="x", pady=(6, 0))

        # — Painel: top 5 ——————————————————————————————————————————————————————
        painel_top = tk.Frame(col_pred, bg=COR_PAINEL, padx=16, pady=12)
        painel_top.pack(fill="x", pady=(0, 10))

        tk.Label(painel_top, text="TOP 5 CANDIDATOS",
                 font=fonte_small, bg=COR_PAINEL, fg=COR_SUBTEXT).pack(anchor="w", pady=(0, 8))

        self._linhas_top: list[dict] = []
        for _ in range(5):
            row = tk.Frame(painel_top, bg=COR_PAINEL)
            row.pack(fill="x", pady=1)
            lbl_nome = tk.Label(row, text="", width=10, anchor="w",
                                font=fonte_normal, bg=COR_PAINEL, fg=COR_TEXTO)
            lbl_nome.pack(side="left")
            barra = ttk.Progressbar(row, length=150, mode="determinate", maximum=100)
            barra.pack(side="left", padx=(4, 4))
            lbl_pct = tk.Label(row, text="", width=5, anchor="e",
                               font=fonte_normal, bg=COR_PAINEL, fg=COR_SUBTEXT)
            lbl_pct.pack(side="left")
            self._linhas_top.append({"nome": lbl_nome, "barra": barra, "pct": lbl_pct})

        # — Painel: histórico ——————————————————————————————————————————————————
        painel_hist = tk.Frame(col_pred, bg=COR_PAINEL, padx=16, pady=12)
        painel_hist.pack(fill="x", pady=(0, 10))

        tk.Label(painel_hist, text="HISTÓRICO",
                 font=fonte_small, bg=COR_PAINEL, fg=COR_SUBTEXT).pack(anchor="w", pady=(0, 4))

        self.lbl_historico = tk.Label(
            painel_hist, text="",
            font=fonte_normal, bg=COR_PAINEL, fg=COR_TEXTO,
            wraplength=240, justify="left"
        )
        self.lbl_historico.pack(anchor="w")

        # — Controles ——————————————————————————————————————————————————————————
        painel_ctrl = tk.Frame(col_pred, bg=COR_FUNDO)
        painel_ctrl.pack(fill="x")

        self.btn_tts = tk.Button(
            painel_ctrl, text="🔊  TTS: LIGADO",
            font=fonte_normal, bg=COR_VERDE, fg=COR_FUNDO,
            relief="flat", padx=10, pady=6,
            command=self._toggle_tts
        )
        self.btn_tts.pack(fill="x", pady=(0, 6))

        self.btn_landmarks = tk.Button(
            painel_ctrl, text="👁  Landmarks: ON",
            font=fonte_normal, bg=COR_BARRA, fg=COR_TEXTO,
            relief="flat", padx=10, pady=6,
            command=self._toggle_landmarks
        )
        self.btn_landmarks.pack(fill="x", pady=(0, 6))

        tk.Button(
            painel_ctrl, text="🗑  Limpar histórico",
            font=fonte_normal, bg=COR_BARRA, fg=COR_TEXTO,
            relief="flat", padx=10, pady=6,
            command=self._limpar_historico
        ).pack(fill="x", pady=(0, 6))

        tk.Button(
            painel_ctrl, text="✕  Fechar",
            font=fonte_normal, bg=COR_VERMELHO, fg=COR_FUNDO,
            relief="flat", padx=10, pady=6,
            command=self._ao_fechar
        ).pack(fill="x")

    # ── Loop principal ────────────────────────────────────────────────────────

    def _loop(self):
        if not self._rodando:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.lbl_status.config(text="⬤  Câmera indisponível", fg=COR_VERMELHO)
            self.root.after(INTERVALO_MS, self._loop)
            return

        frame = cv2.flip(frame, 1)  # espelho

        try:
            resultado = self.pipeline.processar(frame)
        except Exception as e:
            self.root.after(INTERVALO_MS, self._loop)
            return

        # Overlay de landmarks no frame
        if self.mostrar_landmarks:
            self._desenhar_landmarks(frame, resultado)

        # Atualiza canvas com o vídeo
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (LARGURA_VIDEO, ALTURA_VIDEO))
        foto = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        self.canvas_video.create_image(0, 0, anchor="nw", image=foto)
        self.canvas_video._foto = foto  # evita garbage collection

        # Atualiza painel de predição
        self._atualizar_predicao(resultado)

        # TTS quando predição estável
        if resultado["pred_estavel"] and resultado["pred_estavel"] != self.ultimo_falado:
            self.ultimo_falado = resultado["pred_estavel"]
            if self.tts_ligado:
                self.tts.falar(resultado["pred_estavel"])

        self.root.after(INTERVALO_MS, self._loop)

    # ── Atualização de widgets ────────────────────────────────────────────────

    def _atualizar_predicao(self, r: dict):
        classe    = r["classe"]
        conf_pct  = int(r["confianca"] * 100)
        top_k     = r["top_k"]

        # Cor do sinal baseada na confiança
        if conf_pct >= 75:
            cor = COR_VERDE
        elif conf_pct >= 50:
            cor = COR_AMARELO
        else:
            cor = COR_SUBTEXT

        self.lbl_sinal.config(text=classe if classe != "vazio" else "—", fg=cor)
        self.lbl_confianca.config(text=f"{conf_pct}%", fg=cor)
        self.barra_conf["value"] = conf_pct

        for i, linha in enumerate(self._linhas_top):
            if i < len(top_k):
                nome, prob = top_k[i]
                pct = int(prob * 100)
                linha["nome"].config(text=nome)
                linha["barra"]["value"] = pct
                linha["pct"].config(text=f"{pct}%")
            else:
                linha["nome"].config(text="")
                linha["barra"]["value"] = 0
                linha["pct"].config(text="")

        # Adiciona ao histórico quando estável
        if r["pred_estavel"] and r["pred_estavel"] != (self.historico[-1] if self.historico else None):
            self.historico.append(r["pred_estavel"])
            self.lbl_historico.config(text="  ".join(self.historico))

    def _desenhar_landmarks(self, frame: np.ndarray, r: dict):
        for (x, y) in r["pontos_pose"]:
            cv2.circle(frame, (x, y), 3, (100, 200, 100), -1)
        for (x, y) in r["pontos_maos"]:
            cv2.circle(frame, (x, y), 5, (255, 180, 0), -1)

        # Label do sinal no canto
        conf_pct = int(r["confianca"] * 100)
        cor_bgr = (100, 220, 100) if conf_pct >= 75 else (100, 180, 255)
        texto = f"{r['classe']}  {conf_pct}%"
        cv2.putText(frame, texto, (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, cor_bgr, 2, cv2.LINE_AA)

    # ── Callbacks dos botões ──────────────────────────────────────────────────

    def _toggle_tts(self):
        self.tts_ligado = not self.tts_ligado
        if self.tts_ligado:
            self.btn_tts.config(text="🔊  TTS: LIGADO",  bg=COR_VERDE,  fg=COR_FUNDO)
        else:
            self.btn_tts.config(text="🔇  TTS: DESLIGADO", bg=COR_BARRA, fg=COR_TEXTO)

    def _toggle_landmarks(self):
        self.mostrar_landmarks = not self.mostrar_landmarks
        estado = "ON" if self.mostrar_landmarks else "OFF"
        self.btn_landmarks.config(text=f"👁  Landmarks: {estado}")

    def _limpar_historico(self):
        self.historico.clear()
        self.lbl_historico.config(text="")
        self.ultimo_falado = None

    def _ao_fechar(self):
        self._rodando = False
        self.cap.release()
        self.pipeline.fechar()
        self.root.destroy()

    # ── Inicialização ─────────────────────────────────────────────────────────

    def iniciar(self):
        self.root.after(500, self._loop)  # aguarda 0.5s para UI renderizar
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default=None,
                        help="Caminho para o arquivo .h5 do modelo (padrao: modelo_dense_78.h5)")
    args = parser.parse_args()

    print("Carregando modelo e MediaPipe...")
    app = Dashboard(modelo_path=args.modelo)
    print("Dashboard pronto. Abrindo janela...")
    app.iniciar()


if __name__ == "__main__":
    main()
