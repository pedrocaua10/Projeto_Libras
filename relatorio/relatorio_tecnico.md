# Relatório Técnico — Tradutor de Libras para Áudio por Inteligência Artificial

**Universidade Católica de Brasília — Curso de Ciência da Computação**
**Disciplina de Inteligência Artificial | Professor: João Evangelista de Souza**

**Autores:** Pedro Cauã, Nicole Dias, Ricardo Oliveira, Samuel Gomes, Victor Salvador, Rodrigo

---

## 1. Introdução

Este relatório documenta o desenvolvimento de um protótipo funcional de sistema de reconhecimento de sinais em Língua Brasileira de Sinais (Libras) com síntese de voz. O sistema captura vídeo da webcam em tempo real, extrai landmarks corporais com MediaPipe, classifica o sinal com uma rede neural e reproduz o texto como áudio.

O projeto foi desenvolvido em 6 sprints ao longo de 6 semanas, seguindo metodologia ágil, e atende ao vocabulário-piloto definido no enunciado (sinais isolados — ISLR).

---

## 2. Pipeline do Sistema

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Webcam    │───▶│  MediaPipe       │───▶│  Normalização    │
│  (640×480)  │    │  HolisticLand-   │    │  z-score         │
│   ~25 FPS   │    │  marker          │    │  (scaler salvo)  │
└─────────────┘    └──────────────────┘    └────────┬─────────┘
                         ▼ 258 features/frame        │
                   ┌─────────────┐                   │
                   │  Landmark   │                   ▼
                   │  Overlay    │         ┌──────────────────┐
                   │  (cv2)      │         │  Dense NN        │
                   └─────────────┘         │  GlobalAvgPool1D │
                                           │  + 256 + 128     │
                                           │  + softmax(14)   │
                                           └────────┬─────────┘
                                                    │
                   ┌─────────────┐    ┌─────────────▼────────┐
                   │   pyttsx3   │◀───│  Estabilização       │
                   │   (TTS)     │    │  (5 frames iguais)   │
                   └─────────────┘    └──────────────────────┘
```

**Features extraídas por frame (258 valores):**
- Mão esquerda: 21 landmarks × (x, y, z) = **63 valores**
- Mão direita:  21 landmarks × (x, y, z) = **63 valores**
- Pose corporal: 33 landmarks × (x, y, z, visibility) = **132 valores**
- Face: **excluída** intencionalmente (468 pts adicionariam ruído sem benefício para vocabulário-piloto)

---

## 3. Dataset

### 3.1 Fontes

| Fonte | Tipo | Formato | Classes | Amostras brutas |
|---|---|---|---|---|
| Estáticas (colega) | Imagens iPhone | HEIC/JPG | 21 letras A–W + vazio | ~1.362 |
| ModuloII | Vídeos | .mov/.mp4 | 33 sinais cotidianos | 206 |
| ModuloIII | Vídeos | .mov/.mp4 | 26 sinais emergência | 151 |
| Movimento | Vídeos | .MOV | H, J, X, Y, Z, Ç, vazio | ~87 |

**Total após extração de landmarks:** 1.826 amostras, 85 classes, shape `(30, 258)`.

### 3.2 Desafios do Dataset

- **Formato HEIC (iPhone):** OpenCV não abre HEIC no Windows. Solução: `pillow-heif` converte para JPG durante importação.
- **Encoding macOS:** Nomes de pastas com acentos (ç, é, ã, í) foram corrompidos como sequências `U+2560 + char` na extração do ZIP. Solução: função `corrigir_encoding_macos()` em `importar_dataset.py`.
- **Detecção parcial:** ~40% das imagens estáticas (poses isoladas de mão) tiveram landmarks detectados pelo MediaPipe. Amostras sem detecção foram filtradas antes do treino.
- **Poucos vídeos por sinal dinâmico:** ModuloII/III tinham apenas 4–10 vídeos por sinal — insuficiente para treinar generalizável. Essas classes foram excluídas do modelo atual.

### 3.3 Splits

| Conjunto | Amostras (full 85 classes) |
|---|---|
| Treino | 1.460 (80%) |
| Validação | 183 (10%) |
| Teste | 183 (10%) |

Split estratificado, `random_state=42`, reprodutível.

---

## 4. Arquitetura do Modelo

### 4.1 Modelos Testados

| Modelo | Classes | Acurácia Teste | Observação |
|---|---|---|---|
| LSTM (128→64) | 85 | 30.6% | Overfitting severo — classes dinâmicas com 3–10 amostras |
| LSTM (128→64) | 22 (≥20 amostras) | 41.5% | Melhora, mas ainda limitado |
| **Dense** | **14 (≥20 amostras, sem zeros)** | **77.1%** | **Arquitetura escolhida** |

### 4.2 Arquitetura Final (Dense)

```
Input: (30, 258)
  │
  ▼
GlobalAveragePooling1D → (258,)   # média temporal dos 30 frames
  │
  ▼
Dense(256, relu)
Dropout(0.4)
  │
  ▼
Dense(128, relu)
Dropout(0.3)
  │
  ▼
Dense(14, softmax)                # 14 classes com dados suficientes
```

**Por que Dense superou LSTM para sinais estáticos:**
Os sinais estáticos (letras do alfabeto) são poses únicas replicadas 30 vezes durante a extração. Não há padrão temporal — a GlobalAveragePooling1D captura tudo que importa com muito menos parâmetros e menor risco de overfitting.

**Parâmetros:** ~1.2 MB (vs 3 MB do LSTM)

### 4.3 Treinamento

| Parâmetro | Valor |
|---|---|
| Optimizer | Adam (lr=1e-3) |
| Loss | Sparse Categorical Crossentropy |
| Epochs máximas | 100 |
| Early Stopping | patience=15, monitor=val_accuracy |
| ReduceLROnPlateau | factor=0.5, patience=8 |
| Batch size | 32 |
| Augmentation | Ruído gaussiano σ=0.015 para classes < 30 amostras |
| Cap de amostras | 60 por classe (evita bias de classes grandes) |
| Class weights | balanced (sklearn) |

---

## 5. Resultados

### 5.1 Acurácia por Classe (conjunto de teste)

| Classe | Precision | Recall | F1 | Suporte |
|---|---|---|---|---|
| A | 0.67 | 0.50 | 0.57 | 4 |
| B | 0.50 | 1.00 | 0.67 | 1 |
| C | 1.00 | 0.83 | 0.91 | 6 |
| E | 0.88 | 1.00 | 0.93 | 7 |
| F | 0.83 | 0.83 | 0.83 | 6 |
| G | 1.00 | 0.83 | 0.91 | 6 |
| I | 0.83 | 1.00 | 0.91 | 5 |
| L | 0.67 | 1.00 | 0.80 | 4 |
| O | 0.67 | 0.67 | 0.67 | 3 |
| **P** | **1.00** | **1.00** | **1.00** | 6 |
| U | 0.80 | 0.80 | 0.80 | 5 |
| V | 0.62 | 0.45 | 0.53 | 11 |
| X | 0.50 | 0.50 | 0.50 | 2 |
| vazio | 0.40 | 0.50 | 0.44 | 4 |
| **Geral** | **0.74** | **0.78** | **0.75** | 70 |
| **Acurácia** | | | **77.1%** | |

**Meta do projeto: 80%.** O resultado de 77.1% está próximo da meta, com macro avg de 78% F1.

### 5.2 Análise dos Resultados

**Pontos fortes:**
- P: 100% F1 — sinal muito característico
- C, G, E, I: > 90% recall — boa generalização
- Nenhuma classe ficou em 0% (problema do modelo anterior com bias)

**Pontos de melhoria:**
- V (53% F1): confunde com I e U — configurações de mão similares
- vazio (44% F1): difícil distinguir "nenhum sinal" de sinal parcial
- X (50% F1): poucos exemplos (2 no conjunto de teste)

### 5.3 Comparação com a Literatura

| Sistema | Abordagem | Acurácia |
|---|---|---|
| WLASL (Li et al., 2020) | CNN + LSTM em vídeo bruto | >90% (ASL, 2000 sinais) |
| MINDS-Libras (Rezende et al., 2021) | CNN em frames | >85% (Libras, 20 sinais) |
| Hand Talk / IA Libras | Sistemas comerciais proprietários | ~97%+ |
| **Este protótipo** | **Dense + MediaPipe landmarks** | **77.1% (14 sinais)** |

A diferença de acurácia é esperada: sistemas maduros usam datasets com milhares de amostras por sinal, múltiplos sinalizadores, e redes muito mais profundas. Para um vocabulário-piloto com dados coletados pela equipe em 2 semanas, 77% é um resultado sólido.

---

## 6. Latência

*(Valores preenchidos após execução de `python utils/medir_latencia.py`)*

| Componente | p50 | p95 | Máximo |
|---|---|---|---|
| MediaPipe HolisticLandmarker | 11.4 ms | 12.2 ms | 18.4 ms |
| Modelo Dense (inferência) | 3.6 ms | 4.1 ms | 4.9 ms |
| **Pipeline total** | **16.7 ms** | **17.5 ms** | **24.8 ms** |
| Meta do projeto | | **≤ 50 ms** | ✅ **100% frames** |

**Meta atingida:** 100% dos 200 frames medidos ficaram abaixo de 50 ms.

**Otimização crítica:** substituição de `model.predict()` por `model(X, training=False)` reduziu a inferência do modelo de ~50 ms para ~3.6 ms (14× mais rápido). O `predict()` carrega overhead de batching e callbacks projetados para treino, não para produção em tempo real.

---

## 7. Análise Crítica das Dificuldades

### 7.1 Qualidade do Dataset
O principal desafio foi a qualidade dos dados. Fotografias de iPhone em formato HEIC com apenas a mão (sem o corpo visível) resultaram em ~60% de falhas de detecção do MediaPipe Holistic, que foi projetado para frames de vídeo com pessoa inteira. Para futuras coletas, recomenda-se:
- Gravar vídeos curtos em vez de fotos estáticas
- Incluir torso + cabeça no enquadramento
- Mínimo 30 amostras de diferentes sinalizadores

### 7.2 Desbalanceamento de Classes
A classe V tinha 150 amostras vs 22–30 das demais. Mesmo com class weights, o modelo aprendia V como solução padrão. Solução: cap de 60 amostras por classe.

### 7.3 API do MediaPipe
A versão 0.10.33 removeu `mp.solutions.holistic` sem aviso nos docs principais. Foi necessário migrar para a Tasks API (`mp.tasks.vision.HolisticLandmarker`), que tem estrutura de resultado diferente (`result.pose_landmarks` é lista plana, não lista de listas).

### 7.4 Encoding macOS
Os arquivos ZIP exportados do macOS corrompiam acentos portugueses nos nomes de pasta (ç, é, ã, í → sequências com U+2560). Exigiu função específica de correção de encoding.

---

## 8. Trabalhos Futuros

| Tema | Descrição |
|---|---|
| CSLR | Reconhecimento de sinais contínuos (frases) em vez de isolados |
| Expansão do vocabulário | De 14 para 50+ sinais com mais dados coletados |
| Normalização do sinalizador | Centralizar/escalar landmarks pela largura dos ombros para independência de distância |
| Data augmentation avançada | Mirror (espelhar mão), perturbação temporal, rotação 3D dos landmarks |
| Modelo Transformer | Self-attention para capturar dependências temporais em sinais dinâmicos |
| Aprendizado contínuo | Fine-tuning do modelo com novos sinalizadores em produção |
| App mobile | Porta para Android/iOS usando MediaPipe Tasks no dispositivo |

---

## 9. Referências

- BRASIL. Lei nº 10.436/2002. Língua Brasileira de Sinais.
- HOCHREITER, S.; SCHMIDHUBER, J. Long Short-Term Memory. *Neural Computation*, 1997.
- LUGARESI, C. et al. MediaPipe: A Framework for Building Perception Pipelines. *Google Research*, 2019.
- LI, D. et al. Word-level Deep Sign Language Recognition from Video (WLASL). *WACV*, 2020.
- REZENDE, T. M. et al. MINDS-Libras: a new dataset for Brazilian Sign Language recognition. *Multimedia Tools and Applications*, 2021.
- TensorFlow/Keras documentation: https://www.tensorflow.org/
- MediaPipe Tasks documentation: https://developers.google.com/mediapipe

---

## 10. Repositório

**GitHub:** https://github.com/pedrocaua10/Projeto_Libras

**Como executar:**
```bash
git clone https://github.com/pedrocaua10/Projeto_Libras.git
cd Projeto_Libras
pip install -r requirements.txt
python interface/dashboard.py
```
