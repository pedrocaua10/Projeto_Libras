"""Valida a estrutura e quantidade de amostras do dataset bruto."""

import sys
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def carregar_config() -> tuple[Path, set, int]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    formatos = set(cfg["formatos_validos"])
    minimo = cfg["amostras_minimas_por_classe"]
    raw = Path(cfg["paths"]["raw"])
    return raw, formatos, minimo


def contar_amostras(pasta_sinal: Path, formatos: set) -> int:
    return sum(
        1 for f in pasta_sinal.iterdir()
        if f.is_file() and f.suffix.lstrip(".").lower() in formatos
    )


def inspecionar_tipo(pasta_tipo: Path, nome_tipo: str, formatos: set, minimo: int) -> list[dict]:
    resultados = []
    if not pasta_tipo.exists():
        print(f"  [AVISO] Pasta '{pasta_tipo}' nao encontrada.")
        return resultados
    for sinal in sorted(pasta_tipo.iterdir()):
        if not sinal.is_dir():
            continue
        n = contar_amostras(sinal, formatos)
        status = "OK" if n >= minimo else f"INSUFICIENTE ({n}/{minimo})"
        resultados.append({"sinal": sinal.name, "tipo": nome_tipo, "amostras": n, "status": status})
    return resultados


def imprimir_tabela(linhas: list[dict]):
    col = max((len(r["sinal"]) for r in linhas), default=6)
    col = max(col, 6)
    cab = f"{'Sinal':<{col}}  {'Tipo':<10}  {'Amostras':>8}  Status"
    print(cab)
    print("-" * len(cab))
    for r in linhas:
        print(f"{r['sinal']:<{col}}  {r['tipo']:<10}  {r['amostras']:>8}  {r['status']}")


def main():
    raw, formatos, minimo = carregar_config()

    resultados = []
    resultados += inspecionar_tipo(raw / "dinamicos", "dinamico", formatos, minimo)
    resultados += inspecionar_tipo(raw / "estaticos", "estatico", formatos, minimo)

    if not resultados:
        print("Nenhum sinal encontrado. Verifique se os videos foram colocados em dataset/raw/.")
        sys.exit(0)

    print(f"\nDataset: {raw.resolve()}\n")
    imprimir_tabela(resultados)

    problemas = [r for r in resultados if "INSUFICIENTE" in r["status"]]
    total = sum(r["amostras"] for r in resultados)

    print(f"\nTotal de amostras : {total}")
    print(f"Total de sinais   : {len(resultados)}")

    if problemas:
        print(f"\n[ATENCAO] {len(problemas)} sinal(is) com amostras insuficientes:")
        for p in problemas:
            print(f"  - {p['tipo']}/{p['sinal']}: {p['amostras']} amostras")
        sys.exit(1)
    else:
        print("\nTodos os sinais estao com amostras suficientes. Pode prosseguir.")


if __name__ == "__main__":
    main()
