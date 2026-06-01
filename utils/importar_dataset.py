"""Importa e organiza o dataset para a estrutura do projeto."""

import re
import shutil
import unicodedata
from pathlib import Path
from tqdm import tqdm

try:
    import pillow_heif
    from PIL import Image
    HEIC_SUPORTADO = True
except ImportError:
    HEIC_SUPORTADO = False

RAIZ = Path(__file__).parent.parent

FONTES_ESTATICOS = [
    Path(r"C:\Users\pedro\Downloads\Estaticas\Estaticas"),
]

FONTES_DINAMICOS = [
    Path(r"C:\Users\pedro\Downloads\ModuloII\ModuloII"),
    Path(r"C:\Users\pedro\Downloads\ModuloIII\ModuloIII"),
    Path(r"C:\Users\pedro\Downloads\Movimento"),
]

# Tutoriais e Numeros - copia sao ignorados intencionalmente:
# Tutoriais: apenas 1 video demonstrativo por sinal, nao sao amostras de treino
# Numeros - copia: pasta vazia

FORMATOS_VIDEO = {".mp4", ".mov", ".avi"}
FORMATOS_IMAGEM = {".heic", ".jpg", ".jpeg", ".png"}
MINIMO_AMOSTRAS = 30


def remover_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return nfkd.encode("ASCII", "ignore").decode("ASCII")


def normalizar_sinal(nome: str) -> str:
    """Converte nome de pasta para convenção do projeto."""
    nome = remover_acentos(nome).strip()
    # Letra única do alfabeto → maiúsculo
    if len(nome) == 1 and nome.isalpha():
        return nome.upper()
    # Duas letras maiúsculas (ex: "CH") → mantém
    if len(nome) == 2 and nome.isupper():
        return nome
    # Classe vazio
    if nome.lower() == "vazio":
        return "vazio"
    # CamelCase → snake_case minúsculo
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", nome)
    s = s.lower()
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def converter_heic(origem: Path, destino: Path):
    if not HEIC_SUPORTADO:
        raise RuntimeError(
            "pillow-heif nao instalado. Execute: pip install pillow-heif Pillow"
        )
    pillow_heif.register_heif_opener()
    img = Image.open(str(origem)).convert("RGB")
    img.save(str(destino), "JPEG", quality=95)


def importar_fonte(fonte: Path, destino_base: Path, contadores: dict, erros: list):
    if not fonte.exists():
        print(f"  [IGNORADO] Pasta nao encontrada: {fonte}")
        return

    for pasta_sinal in sorted(fonte.iterdir()):
        if not pasta_sinal.is_dir() or pasta_sinal.name in {"__MACOSX"}:
            continue

        nome = normalizar_sinal(pasta_sinal.name)
        destino = destino_base / nome
        destino.mkdir(parents=True, exist_ok=True)

        arquivos = [
            f for f in sorted(pasta_sinal.iterdir())
            if f.is_file()
            and f.suffix.lower() in FORMATOS_VIDEO | FORMATOS_IMAGEM
            and not f.name.startswith("._")
            and "(1)" not in f.name  # ignora duplicatas com "(1)" no nome
        ]

        if not arquivos:
            continue

        idx = contadores.get(nome, 0)
        for arq in tqdm(arquivos, desc=f"  {nome}", unit="arq", leave=False):
            try:
                ext = arq.suffix.lower()
                if ext == ".heic":
                    destino_arq = destino / f"amostra_{idx:03d}.jpg"
                    converter_heic(arq, destino_arq)
                else:
                    destino_arq = destino / f"amostra_{idx:03d}{ext}"
                    shutil.copy2(arq, destino_arq)
                idx += 1
            except Exception as e:
                erros.append(f"{pasta_sinal.name}/{arq.name}: {e}")

        contadores[nome] = idx


def imprimir_resumo(titulo: str, contadores: dict):
    total_classes = len(contadores)
    total_amostras = sum(contadores.values())
    insuficientes = [(s, n) for s, n in contadores.items() if n < MINIMO_AMOSTRAS]

    print(f"\n{titulo}: {total_classes} classes, {total_amostras} amostras")
    for sinal, n in sorted(contadores.items()):
        status = "OK" if n >= MINIMO_AMOSTRAS else f"INSUFICIENTE ({n}/{MINIMO_AMOSTRAS})"
        print(f"  {sinal:<30} {n:>4}  {status}")

    if insuficientes:
        print(f"\n  [ATENCAO] {len(insuficientes)} classe(s) abaixo de {MINIMO_AMOSTRAS} amostras.")
        print("  Considere gravar mais amostras ou usar data augmentation no Sprint 4.")


def main():
    destino_est = RAIZ / "dataset" / "raw" / "estaticos"
    destino_din = RAIZ / "dataset" / "raw" / "dinamicos"
    contadores_est: dict[str, int] = {}
    contadores_din: dict[str, int] = {}
    erros: list[str] = []

    print("Importando estaticos (imagens)...")
    for fonte in FONTES_ESTATICOS:
        importar_fonte(fonte, destino_est, contadores_est, erros)

    print("\nImportando dinamicos (videos)...")
    for fonte in FONTES_DINAMICOS:
        importar_fonte(fonte, destino_din, contadores_din, erros)

    imprimir_resumo("Estaticos", contadores_est)
    imprimir_resumo("Dinamicos", contadores_din)

    if erros:
        print(f"\n[ERROS] {len(erros)} arquivo(s) com problema:")
        for e in erros:
            print(f"  {e}")
    else:
        print("\nImportacao concluida sem erros.")

    print("\nProximo passo: python utils/validar_dataset.py")


if __name__ == "__main__":
    main()
