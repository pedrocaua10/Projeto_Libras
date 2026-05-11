# Como organizar os vídeos do dataset

Obrigado por contribuir com o dataset! Siga este guia para que os scripts de
pré-processamento funcionem sem ajustes.

## Estrutura de pastas esperada

```
dataset/raw/
├── dinamicos/
│   └── <nome_do_sinal>/
│       ├── amostra_01.mp4
│       ├── amostra_02.mp4
│       └── ...
└── estaticos/
    └── <letra_ou_palavra>/
        ├── amostra_01.mp4
        ├── amostra_02.mov
        └── ...
```

- **dinamicos/** — sinais com movimento (ex: "oi", "obrigado", "ajuda", "como_vai")
- **estaticos/** — letras do alfabeto e palavras sem movimento (ex: "A", "B", "paz", "amor")

## Regras de nomenclatura das pastas

| Regra | Correto | Errado |
|---|---|---|
| Letras minúsculas | `obrigado` | `Obrigado` |
| Sem acentos | `mae` | `mãe` |
| Underline para espaços | `como_vai` | `como vai` |
| Sem caracteres especiais | `oi` | `oi!` |
| Letras do alfabeto em maiúsculo | `A`, `B`, `C` | `a`, `b`, `c` |

## Formatos aceitos

`.mp4`, `.mov` e `.avi` — qualquer um dos três funciona.

> **Aviso para Windows:** arquivos `.mov` às vezes não abrem sem codec instalado.
> Se aparecer erro de leitura no log, converta para `.mp4` antes de rodar o script.

## Quantidade mínima por sinal

Mínimo de **30 vídeos** por pasta. O script de validação vai alertar automaticamente
quais classes estão com poucas amostras.

## Nome dos arquivos de vídeo

Pode ser qualquer nome. Sugestão: `amostra_01.mp4`, `amostra_02.mp4`, etc.
Não use espaços no nome dos arquivos.

## Exemplo concreto

```
dataset/raw/
├── dinamicos/
│   ├── oi/
│   │   ├── amostra_01.mp4
│   │   ├── amostra_02.mp4
│   │   └── ... (30+ vídeos)
│   └── obrigado/
│       ├── amostra_01.mp4
│       └── ... (30+ vídeos)
└── estaticos/
    ├── A/
    │   ├── amostra_01.mp4
    │   └── ... (30+ vídeos)
    └── B/
        └── ... (30+ vídeos)
```

## Após colocar os vídeos

Execute na raiz do projeto:

```bash
python utils/validar_dataset.py
```

Ele vai mostrar uma tabela com o status de cada sinal. Só passe para o próximo
passo quando todos os sinais estiverem com status **OK**.

## Dúvidas

Fale com Pedro (pedro.caua4@gmail.com) ou abra uma issue no repositório do projeto.
