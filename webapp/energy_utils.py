"""
energy_utils.py  –  utilitários para extração e análise de consumo/irradiação

Requisitos:
    • pandas ≥ 2.2
    • numpy, matplotlib, requests, geopandas, shapely, concurrent-futures
    • opcional: seaborn (apenas para styling do Matplotlib)

Uso rápido (CLI):
    python -m energy_utils --debug
"""

from __future__ import annotations

# ──────────────── IMPORTS ────────────────────────────────────────────────
import argparse
import logging
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")     # backend sem Tkinter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

# CONSTANTES ──────────────────────────────────────────────────────────────
DATASET_INTERVALS: list[tuple[int, int, str]] = [
    (1000, 2000, "1000a2000"),
    (2001, 2500, "2001a2500"),
    (2501, 2695, "2501a2695"),
    (2696, 2820, "2696a2820"),
    (2821, 3080, "2821a3080"),
    (3081, 3780, "3081a3780"),
    (3781, 4420, "3781a4420"),
    (4421, 4550, "4421a4550"),
    (4551, 4770, "4551a4770"),
    (4771, 5200, "4771a5200"),
    (5201, 7400, "5201a7400"),
    (7401, 8970, "7401a8970"),
]

HEADERS_DEFAULT = {"User-Agent": "energy-utils/1.0 (+https://github.com/seu-usuario/projeto)"}  # CHANGE: boa prática


# ──────────────── LOGGING ────────────────────────────────────────────────
logger = logging.getLogger("energy_utils")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(_handler)

# ──────────────── 1. ENDEREÇOS & CP-4 ────────────────────────────────────
def autocomplete_address(query: str, limit: int = 5) -> list[dict]:
    """
    Usa o serviço Photon para sugerir endereços a partir de uma string.
    Exemplo de uso: autocomplete_address("Rua José Pereira Tavares, Aveiro")
    """
    url = "https://photon.komoot.io/api/"
    params = {"q": query, "limit": str(limit)}

    # ✅ Cabeçalho compatível com latin-1 / ASCII
    headers = {
        "User-Agent": "energy-utils/1.0 (+https://github.com/seu-usuario/projeto)"
    }

    try:
        r = requests.get(url, params=params, timeout=15, headers=headers)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Erro na chamada à API de endereços: {e}")
        return []

    out = []
    for f in r.json().get("features", []):
        p = f["properties"]
        out.append({
            "name": p.get("name", ""),
            "city": p.get("city", ""),
            "country": p.get("country", ""),
            "lat": f["geometry"]["coordinates"][1],
            "lon": f["geometry"]["coordinates"][0],
            "full": f'{p.get("name", "")}, {p.get("city", "")}, {p.get("country", "")}',
        })

    return out


def get_dataset_identifier(cp4: int) -> str:
    """Mapeia o CP-4 para o identificador de dataset da API e-Redes."""
    for a, b, suffix in DATASET_INTERVALS:
        if a <= cp4 <= b:
            return f"consumoshorariocodigopostal{suffix}"
    raise ValueError(f"CP4 {cp4} fora do intervalo 1000-8970.")


# ──────────────── 2. API e-Redes (paginado) ─────────────────────────────
def call_api(
    dataset_identifier: str,
    cp7: str,
    *,
    page_size: int = 20,
    max_registos: int | None = None,
) -> list[dict]:
    """Devolve todos (ou até *max_registos*) os registos para o CP-7 dado."""
    out: list[dict] = []
    offset = 0
    while True:
        params = {
            "limit": page_size,
            "offset": offset,
            "refine": f'codigo_postal:"{cp7}"',
        }
        url = (
            "https://e-redes.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
            f"{dataset_identifier}/records?"
            + urllib.parse.urlencode(params, safe=':"')
        )
        r = requests.get(url, timeout=30, headers=HEADERS_DEFAULT)
        if r.status_code != 200:
            logger.warning("e-Redes %s – %s", r.status_code, cp7)
            break

        js = r.json()
        rows = js.get("results", [])
        out.extend(rows)

        total = js.get("total_count", len(out))
        logger.debug("Pág. %d – +%d (%d/%s)", offset // page_size + 1, len(rows), len(out), total)

        if not rows or len(out) >= total:
            break
        if max_registos and len(out) >= max_registos:
            out = out[:max_registos]
            break
        offset += page_size

    logger.info("» %s – %d registos", cp7, len(out))
    return out


# ──────────────── 3. PVGIS ───────────────────────────────────────────────
def obter_irradiacao_pvgis(lat: float, lon: float) -> pd.DataFrame:
    """Download da irradiância horária (kWh/m²) via PVGIS para 2024."""
    url = "https://re.jrc.ec.europa.eu/api/seriescalc"
    params = {
        "lat": lat,
        "lon": lon,
        "startyear": 2020,
        "endyear": 2020,
        "outputformat": "json",
        "angle": 0,
        "aspect": 0,
        "global": 1,
        "localtime": 1,
        "usehorizon": 1,
        "pvcalculation": 0,
        "radiation_db": "PVGIS-SARAH2",
        "hourlyvalues": 1,
    }
    r = requests.get(url, params=params, timeout=30, headers=HEADERS_DEFAULT)
    r.raise_for_status()

    rows = [
        {
            "Latitude": lat,
            "Longitude": lon,
            "DataHora": pd.to_datetime(h["time"], format="%Y%m%d:%H%M")
            .replace(year=2024)
            .floor("h"),
            "producao_kWh_m2": h["G(i)"] / 1000,
        }
        for h in r.json()["outputs"]["hourly"]
    ]
    return pd.DataFrame(rows)


# ──────────────── 4. CONSOLIDAÇÃO & IMPUTAÇÃO ────────────────────────────
def _drop_tz(s: pd.Series) -> pd.Series:
    """Remove fuso horário mantendo hora local."""
    if getattr(s.dt, "tz", None) is not None:
        return s.dt.tz_convert(None).dt.tz_localize(None)
    return s


def consolidar_dados(df_cons: pd.DataFrame, df_irr: pd.DataFrame) -> pd.DataFrame:
    """Merge consumo (PTD) + irradiância; garante calendário horário completo."""
    df_cons = df_cons.copy()
    df_cons["DataHora"] = _drop_tz(pd.to_datetime(df_cons["datahora"], utc=True))

    df_irr = df_irr.copy()
    df_irr["DataHora"] = _drop_tz(pd.to_datetime(df_irr["DataHora"]).dt.floor("h"))

    ano = df_cons["DataHora"].dt.year.min() if not df_cons.empty else 2024
    base = pd.date_range(f"{ano}-01-01 00:00", f"{ano}-12-31 23:00", freq="h")
    df_base = pd.DataFrame({"DataHora": base})

    df = (
        df_base.merge(df_cons[["DataHora", "consumo_ptd"]], on="DataHora", how="left")
        .merge(df_irr[["DataHora", "producao_kWh_m2"]], on="DataHora", how="left")
        .sort_values("DataHora")
        .reset_index(drop=True)
    )
    return df


# (…mantêm-se as restantes funções *imputar_consumo_restante*,
#  *normalizar_e_corrigir_por_dia*, *plot_dia_medio_normalizado*,
#  mas com pequenos ajustes de nomenclatura e logging – omissas aqui por brevidade)  # noqa: E501

# ⋯ mantém o código anterior até à função consolidar_dados ⋯


# ──────────────── 4b. HELPERS FALTANTES ──────────────────────────────────
def _buscar_cp7(row: pd.Series) -> list[dict]:
    """
    Descarrega todos os registos de consumo horário para um CP-7
    e adiciona metadados de ponderação por área.

    Retorna lista de dicionários (linhas da API).
    """
    cp7 = row["cod_postal"]
    peso = row["percent_area_in_ptd"] / 100.0
    cp4 = int(cp7.split("-")[0])

    try:
        dataset = get_dataset_identifier(cp4)
    except ValueError as exc:
        logger.debug("Ignorado: %s", exc)
        return []

    logger.debug("→ CP-7 %s (peso %.1f%%)", cp7, peso * 100)
    registos = call_api(dataset, cp7)
    for r in registos:
        r["_ponderacao_cp"] = peso     # peso para somatório ponderado
        r["_cp7"] = cp7
    return registos


def imputar_consumo_restante(
    df_ptd: pd.DataFrame,           # DataHora + consumo_ptd + producao_kWh_m2
    df_cp7: pd.DataFrame,           # todos os registos CP-7 (com _ponderacao_cp)
    df_4_digitos: pd.DataFrame,     # CP-4 × Dia_MM_DD × Consumo_normalizado
    cp_info: pd.DataFrame,          # CP-7 × percent_area_in_ptd
    *,
    dia_ref: str = "2024-02-10",
    debug: bool = False,
) -> pd.DataFrame:
    """
    Preenche consumos horários em falta (fora de Fevereiro) com base num perfil-referência
    de 10-Fev normalizado por CP-4 e ponderado pela percentagem de área do CP-7 no PTD.

    A lógica segue o documento original; apenas foram trocados prints por logger.
    """
    if debug:
        logger.setLevel(logging.DEBUG)

    df_cp7 = df_cp7.copy()
    df_cp7["datahora"] = pd.to_datetime(df_cp7["datahora"], utc=True)
    dia_ref_date = pd.to_datetime(dia_ref).date()

    # 1) perfil hora-a-hora de 10-Fev por CP-7 (já ponderado pela área)
    perfis = (
        df_cp7[df_cp7["datahora"].dt.date == dia_ref_date]
        .merge(cp_info[["cod_postal", "percent_area_in_ptd"]],
               left_on="_cp7", right_on="cod_postal")
        .assign(
            hora=lambda d: d["datahora"].dt.hour,
            peso=lambda d: d["percent_area_in_ptd"] / 100,
            base_kwh=lambda d: d["consumo"] * d["peso"],
            CP4=lambda d: d["_cp7"].str[:4].astype(int),
        )
    )

    if perfis.empty or perfis["hora"].nunique() < 24:
        raise ValueError("10-Fev não contém 24 valores por CP-7.")

    base_cp7 = (
        perfis.pivot_table(index="hora", columns="_cp7", values="base_kwh")
        .sort_index()
    )                       # (24h × n_cp7)

        # ── 2) FATOR DIÁRIO POR CP-4 ────────────────────────────────────────
    df4 = df_4_digitos.rename(columns={"Código Postal": "CP4"}).copy()

    # limpa e garante CP4 inteiro
    df4["CP4"] = (
        df4["CP4"].astype(str)
                  .str.extract(r"(\d{4})")[0]   # só 4 dígitos
                  .dropna()
                  .astype(int)
    )
    df4["Data"] = pd.to_datetime("2024-" + df4["Dia_MM_DD"]).dt.date

    # Pivot: datas × CP4  (valor = Consumo_normalizado)
    matriz = df4.pivot(index="Data", columns="CP4", values="Consumo_normalizado")

    # linha-referência 10-Fev
    dia_ref_date = pd.to_datetime(dia_ref).date()
    if dia_ref_date not in matriz.index:
        raise ValueError(f"Dia-referência {dia_ref_date} ausente em consumos 4-dígitos.")

    referencia = matriz.loc[dia_ref_date]

    # Divide cada dia pelo valor desse CP-4 em 10-Fev
    fatores = matriz.div(referencia)

    # 3) mapeia fatores aos CP-7 correspondentes
    mapa_cp7_cp4 = perfis.drop_duplicates("_cp7").set_index("_cp7")["CP4"]
    fatores_cp7 = fatores[mapa_cp7_cp4.values]          # mantém só CP-4 relevantes
    fatores_cp7.columns = mapa_cp7_cp4.index            # renomeia colunas → CP-7

    # dias ausentes → ffill
    idx_full = pd.date_range("2024-01-01", "2024-12-31", freq="D").date
    fatores_cp7 = fatores_cp7.reindex(idx_full).sort_index().ffill()


    # 4) calendário completo
    calendario = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h")
    df_out = (
        pd.DataFrame({"DataHora": calendario})
        .merge(df_ptd, on="DataHora", how="left")
        .sort_values("DataHora")
        .reset_index(drop=True)
    )

    # 5) imputação (apenas horas fora de Fevereiro que ainda são NaN)
    mask_nan = df_out["consumo_ptd"].isna() & (df_out["DataHora"].dt.month != 2)
    logger.debug("Horas a imputar: %d", mask_nan.sum())

    for dt_date in df_out.loc[mask_nan, "DataHora"].dt.date.unique():
        if dt_date == dia_ref_date or dt_date not in fatores_cp7.index:
            continue
        vec_factor = fatores_cp7.loc[dt_date]            # Série index=CP-7
        estim_horas = base_cp7.mul(vec_factor, axis=1).sum(axis=1)

        idxs = df_out.index[df_out["DataHora"].dt.date == dt_date]
        df_out.loc[idxs, "consumo_ptd"] = estim_horas.values

    return df_out

# ──────────────── 4c. NORMALIZAÇÃO & CORRELAÇÃO ──────────────────────────
def normalizar_e_corrigir_por_dia(
    df: pd.DataFrame,
    *,
    limiar: float = 0.02,
    col_ts: str = "DataHora",
    col_cons: str = "consumos",
    col_irr: str = "irradiacao",
    col_id: str = "id_PTD",
) -> dict[int, float]:
    """
    Calcula, para cada PTD, a correlação (Pearson) hora-a-hora entre consumo
    e irradiação **após normalização min-max por dia**.

    Regras:
        • Mantém apenas horas 05-22.
        • Desconsidera horas com irradiância ≤ *limiar* (kWh/m²).
        • Retorna dict {id_PTD: correlação}. PTDs sem dados adequados são omitidos.

    Pré-requisitos de *df*:
        • col_ts    – datetime64[ns]
        • col_cons  – consumo em kWh
        • col_irr   – irradiação em kWh/m²
        • col_id    – identificador do PTD
    """
    resultados: dict[int, float] = {}
    dados = df.copy()

    if col_ts not in dados.columns:
        raise KeyError(f"Coluna '{col_ts}' não encontrada em df")

    dados[col_ts] = pd.to_datetime(dados[col_ts])
    dados["dia"] = dados[col_ts].dt.floor("D")
    dados["hora"] = dados[col_ts].dt.hour

    # mantém horas úteis da curva diurna
    dados = dados[dados["hora"].between(5, 22)]
    logger.debug("Horas elegíveis pós-filtro 05-22: %d", len(dados))

    # percorre cada PTD
    for id_ptd, g in dados.groupby(col_id, sort=False):
        g = g[g[col_irr] > limiar]
        if g.empty:
            logger.debug("PTD %s sem irradiância > limiar", id_ptd)
            continue

        def _minmax(series: pd.Series) -> pd.Series:
            rng = series.max() - series.min()
            return (series - series.min()) / rng if rng else 0.0

        g["cons_norm"] = g.groupby("dia")[col_cons].transform(_minmax)
        g["irr_norm"]  = g.groupby("dia")[col_irr].transform(_minmax)

        # é preciso variação em ambas as séries
        if g["cons_norm"].nunique() > 1 and g["irr_norm"].nunique() > 1:
            resultados[int(id_ptd)] = g["cons_norm"].corr(g["irr_norm"])
            logger.debug("PTD %s → r=%.3f", id_ptd, resultados[id_ptd])

    return resultados


# ⋯ o resto do arquivo mantém-se igual (processar_endereco, CLI, etc.) ⋯

# ──────────────── 5. FLUXO COMPLETO (CLI) ────────────────────────────────
def processar_endereco(
    df_ptds: gpd.GeoDataFrame,
    df_4_digitos: pd.DataFrame,
    *,
    debug: bool = False,
):
    """
    Pipeline completo:
        1. Escolhe endereço (autocomplete Photon)
        2. Localiza PTD
        3. Descarrega consumo CP-7 + irradiância PVGIS
        4. Consolida, imputa, devolve df + correlação
    """
    if debug:
        logger.setLevel(logging.DEBUG)

    q = input("Endereço (ou parte): ")
    sugestoes = autocomplete_address(q)
    if not sugestoes:
        logger.error("Nenhuma sugestão.")
        return

    for i, e in enumerate(sugestoes):
        print(f"{i}: {e['full']}")
    sel = sugestoes[int(input("Escolha nº: "))]

    # PTD que contém o ponto
    ponto = gpd.GeoSeries([Point(sel["lon"], sel["lat"])], crs=4326).to_crs(df_ptds.crs).iloc[0]
    match = df_ptds[df_ptds.contains(ponto)]
    if match.empty:
        logger.error("Fora de qualquer PTD.")
        return

    idx_ptd = int(match.iloc[0]["index_ptd"])
    logger.info("index_ptd: %s", idx_ptd)

    cp_info = (
        df_ptds[df_ptds["index_ptd"] == idx_ptd][["cod_postal", "percent_area_in_ptd"]]
        .drop_duplicates()
    )
    lat, lon = match.iloc[0][["LATITUDE", "LONGITUDE"]]

    dados: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(10, len(cp_info) + 2)) as ex:
        futs_cp = [ex.submit(_buscar_cp7, r) for _, r in cp_info.iterrows()]
        fut_irr = ex.submit(obter_irradiacao_pvgis, lat, lon)

        for f in as_completed(futs_cp):
            dados.extend(f.result())
        df_irr = fut_irr.result()

    if not dados:
        logger.warning("Sem dados de consumo.")
        return

    # Consumo ponderado → PTD
    df_cp7 = pd.DataFrame(dados)
    df_cp7["consumo_ponderado"] = df_cp7["consumo"] * df_cp7["_ponderacao_cp"]
    df_cons = (
        df_cp7.groupby("datahora", as_index=False)["consumo_ponderado"]
        .sum()
        .rename(columns={"consumo_ponderado": "consumo_ptd"})
    )

    # Consolida + imputação
    df_ptd = consolidar_dados(df_cons, df_irr)
    df_ptd = imputar_consumo_restante(
        df_ptd,
        df_cp7,
        df_4_digitos,
        cp_info,
        debug=debug,
    )

    # Correlação
    df_corr = df_ptd.rename(
        columns={"consumo_ptd": "consumos", "producao_kWh_m2": "irradiacao"}
    )
    df_corr["id_PTD"] = idx_ptd
    corr = normalizar_e_corrigir_por_dia(df_corr).get(idx_ptd, np.nan)
    logger.info("Correlação consumo × irradiação (PTD %s): %.3f", idx_ptd, corr)

    return df_ptd, corr, idx_ptd


# ──────────────── PONTO DE ENTRADA ───────────────────────────────────────
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="energy_utils – extração & análise de dados e-Redes/PVGIS"
    )
    p.add_argument(
        "--debug", action="store_true", help="Mostra logs detalhados e chamadas da API"
    )
    p.add_argument(
        "--ptds",
        type=Path,
        required=True,
        help="Shapefile/parquet/GeoPackage dos PTD (deve ter index_ptd, LATITUDE, LONGITUDE, cod_postal, …)",
    )
    p.add_argument(
        "--cp4",
        type=Path,
        required=True,
        help="CSV/Excel com colunas CP4, Dia_MM_DD, Consumo_normalizado",
    )
    return p.parse_args()

# ----------------------------------------------------------------------
def plot_dia_medio_normalizado(
    df: pd.DataFrame,
    id_ptd: int,
    *,
    col_ts: str = "DataHora",
    col_cons: str = "consumo_ptd",
    col_irr: str = "producao_kWh_m2",   # underscore por omissão
) -> Optional[plt.Figure]:
    """
    Gera (e devolve) o gráfico do padrão diário médio normalizado 0-1
    Consumo × Irradiação para um PTD.

    Retorno
    -------
    matplotlib.figure.Figure  |  None se df vazio
    """
    df = df.copy()

    # — garante existência da coluna de irradiação ----------------------
    if col_irr not in df.columns:
        alt = col_irr.replace("_", "/")  # tenta variante com barra
        if alt in df.columns:
            col_irr = alt
        else:
            raise KeyError(f"Coluna '{col_irr}' (ou '{alt}') não encontrada no DataFrame.")

    # — filtra PTD -------------------------------------------------------
    df = df.loc[df["id_PTD"] == id_ptd]
    if df.empty:
        print(f"[plot] Nenhum dado para PTD {id_ptd}.")
        return None

    df.loc[:, col_ts] = pd.to_datetime(df[col_ts])
    df.loc[:, "hour"] = df[col_ts].dt.hour

    # médias horárias
    media = (
        df.groupby("hour")[[col_cons, col_irr]]
          .mean()
          .rename(columns={col_cons: "cons_media", col_irr: "irr_media"})
          .reset_index()
    )

    # normalização min-max ----------------------------------------------
    def _norm(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng else 0.0

    media["cons_norm"] = _norm(media["cons_media"])
    media["irr_norm"]  = _norm(media["irr_media"])

    # — plot -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(media["hour"], media["cons_norm"], linewidth=2, label="Consumo médio (norm.)")
    ax.plot(media["hour"], media["irr_norm"],  linewidth=2, label="Irradiação média (norm.)")
    ax.set(title=f"Padrão Diário Médio Normalizado – PTD {id_ptd}",
           xlabel="Hora do dia", ylabel="Valor normalizado")
    ax.set_xticks(range(0, 24))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return fig

import geopandas as gpd

# ----------------------------------------------------------------------

def carregar_pop_poi(caminho_gpkg: str) -> gpd.GeoDataFrame:
    """Carrega o GeoPackage com dados de população e POIs."""
    return gpd.read_file(caminho_gpkg).to_crs(4326)

# ----------------------------------------------------------------------

def obter_pop_e_poi(gdf_pop_poi: gpd.GeoDataFrame, ptd_geom) -> tuple[int, int]:
    """Retorna pop_sum e poi_count da feature mais próxima ao centroide de ptd_geom."""
    if gdf_pop_poi.empty:
        return 0, 0

    ptd_centroid = ptd_geom.centroid
    nearest_idx = gdf_pop_poi.distance(ptd_centroid).sort_values().index[0]
    row = gdf_pop_poi.loc[nearest_idx]

    pop = int(row.get("pop_sum", 0))
    poi = int(row.get("poi_count", 0))
    return pop, poi

# ----------------------------------------------------------------------

def esta_em_portugal_continental(lat: float, lon: float) -> bool:
    """
    Verifica se o ponto (lat, lon) está dentro de Portugal continental,
    com base numa bounding box simples.
    """
    return 36.8 <= lat <= 42.2 and -9.5 <= lon <= -6.1

# ----------------------------------------------------------------------

def main() -> None:
    args = _parse_cli()

    # Carregamento genérico – ajuste conforme formato real
    df_ptds = gpd.read_file(args.ptds)
    df_4_digitos = pd.read_csv(args.cp4)

    processar_endereco(df_ptds, df_4_digitos, debug=args.debug)


if __name__ == "__main__":  # Executado apenas via “python -m energy_utils”
    main()