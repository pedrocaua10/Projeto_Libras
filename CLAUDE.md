# Projeto Tradutor de Libras para Áudio

Protótipo funcional de sistema baseado em IA que reconhece sinais em Libras via câmera e converte em áudio sintetizado. Projeto prático da disciplina de Inteligência Artificial — Ciência da Computação, UCB.

**Equipe:** Pedro Cauã, Nicole Dias, Ricardo Oliveira, Samuel Gomes, Victor Salvador, Rodrigo  
**Professor:** João Evangelista de Souza

## Stack

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.11 |
| Captura de vídeo | OpenCV (`cv2`) |
| Extração de landmarks | MediaPipe Holistic (190+ landmarks 3D: mãos + pose + face) |
| Modelo de classificação | TensorFlow/Keras — LSTM ou GRU |
| Síntese de voz (TTS) | `pyttsx3` (offline) ou `gTTS` (online) |
| Interface | Tkinter (desktop) ou HTML5/JS (web) |
| Pré-processamento | NumPy, Pandas |
| Avaliação | Scikit-learn, Matplotlib |
| Versionamento | Git + GitHub |

## Estrutura de Pastas

```
Projeto_Libras/
├── captura/          # Scripts de captura de webcam e extração de landmarks ao vivo
├── dataset/          # Sequências de landmarks rotuladas (.npy por sinal/amostra)
│   └── <sinal>/      # Ex: dataset/oi/, dataset/obrigado/
├── modelo/           # Treinamento, pesos salvos (.h5 / SavedModel) e avaliação
├── interface/        # UI desktop (Tkinter) ou web (HTML/JS)
├── utils/            # Funções compartilhadas (normalização, padding, TTS, etc.)
└── CLAUDE.md
```

## Pipeline (5 estágios)

1. **Captura** — OpenCV lê webcam a 15–30 FPS
2. **Landmarks** — MediaPipe Holistic extrai vetor numérico por frame
3. **Classificação** — Janela deslizante de 30–60 frames → LSTM/GRU → classe ou "nenhum"
4. **Texto** — Palavra reconhecida exibida na tela com buffer de estabilização
5. **TTS** — Texto enviado ao mecanismo de síntese e reproduzido como áudio

Meta de latência: **≤ 50 ms por inferência** (processamento local/edge).

## Vocabulário-piloto

10–20 sinais isolados (ISLR), mínimo 30 amostras por sinal. Escolher sinais bem distintos visualmente para reduzir confusão no modelo.

## Convenções

- Python 3.11; dependências em `requirements.txt`
- Docstrings: uma linha curta, apenas quando o propósito não for óbvio pelo nome
- Landmarks salvos como arrays NumPy: shape `(n_frames, n_features)` por amostra
- Splits: 80% treino / 10% validação / 10% teste, reprodutíveis via `random_state=42`
- Nomes de arquivo de peso: `modelo_<arquitetura>_<acuracia>.h5` (ex: `modelo_lstm_87.h5`)
- Acurácia mínima aceitável no conjunto de teste: **80%**

## Sprints

| Sprint | Entrega principal |
|---|---|
| 1 | Captura + visualização de landmarks em tempo real |
| 2 | Definição do vocabulário + gravação do dataset |
| 3 | Pré-processamento, normalização, padding, split |
| 4 | Treinamento LSTM/GRU + matriz de confusão |
| 5 | Integração pipeline completo + TTS + interface |
| 6 | Testes com usuários, medição de latência, relatório |
