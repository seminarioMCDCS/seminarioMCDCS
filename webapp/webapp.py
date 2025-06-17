#!/usr/bin/env python3
"""
webapp.py  –  Flask dentro de FastAPI para Hugging Face Spaces
Estrutura esperada (tudo no mesmo diretório):

├─ webapp.py
├─ index.html
├─ energy_utils.py        (ou pasta src/energy_utils.py)
├─ perc_cp_PTD.csv
├─ consumos_4_digitos.csv
└─ requirements.txt
"""

from __future__ import annotations

import base64
import builtins
import io
import logging
import os
import sys
from pathlib import Path

# ── Terceiros ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")                          # backend sem Tk
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely import wkt
from flask import Flask, render_template, request, flash
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware

# ── Diretório base --------------------------------------------------------
BASE = Path(__file__).parent.resolve()

# ── Importa energy_utils do mesmo diretório ou da pasta src/ -------------
try:
    # 1) mesmo diretório
    sys.path.insert(0, str(BASE))
    import energy_utils as eu
except ModuleNotFoundError:
    # 2) fallback: ./src/
    SRC_DIR = BASE / "src"
    sys.path.insert(0, str(SRC_DIR))
    import energy_utils as eu  # type: ignore

# ── Logging --------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("webapp")

# ── Carga única dos CSV ---------------------------------------------------
CSV_PTDS = BASE / "perc_cp_PTD.csv"
CSV_CP4  = BASE / "consumos_4_digitos.csv"
GPKG_POP_POI = BASE / "pop_poi.gpkg"
GDF_POP_POI = eu.carregar_pop_poi(str(GPKG_POP_POI))

df_ptds_raw = pd.read_csv(CSV_PTDS)
required = {"geometry", "cod_postal", "percent_area_in_ptd", "index_ptd"}
missing  = required - set(df_ptds_raw.columns)
if missing:
    raise SystemExit(f"Faltam colunas no CSV PTDs: {', '.join(missing)}")

df_ptds_raw["geometry"] = df_ptds_raw["geometry"].apply(wkt.loads)
DF_PTDS = gpd.GeoDataFrame(df_ptds_raw, geometry="geometry", crs=3857).to_crs(4326)
DF_CP4  = pd.read_csv(CSV_CP4)

# ── Helpers --------------------------------------------------------------
def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_pipeline(address: str):
    """Executa processar_endereco substituindo input()."""
    answers = iter([address, "0"])
    old_input = builtins.input
    builtins.input = lambda _: next(answers)
    try:
        out = eu.processar_endereco(DF_PTDS, DF_CP4, debug=False)
    finally:
        builtins.input = old_input

    if out is None:
        raise RuntimeError("Sem correspondência com nenhum PTD.")

    df_final, corr, idx_ptd = out
    if "id_PTD" not in df_final.columns:
        df_final["id_PTD"] = idx_ptd

    pop, poi = eu.obter_pop_e_poi(GDF_POP_POI, DF_PTDS.loc[idx_ptd, "geometry"])
    
    fig = eu.plot_dia_medio_normalizado(df_final, idx_ptd)
    return dict(
        correlation=round(corr, 3),
        plot_png=fig_to_base64(fig),
        pop_sum=pop,
        poi_count=poi,
    )


# ── Flask app (template_folder = BASE) -----------------------------------
flask_app = Flask(__name__, template_folder=str(BASE))
flask_app.secret_key = "eco-demo-secret"

@flask_app.route("/", methods=["GET", "POST"])
def index() -> str:
    ctx = dict(show_results=False, address_input="")

    if request.method == "POST":
        address = request.form.get("address", "").strip()
        if not address:
            flash("Por favor, insere uma morada válida.", "error")
            return render_template("index.html", **ctx)

        ctx["address_input"] = address
        try:
            result = run_pipeline(address)
        except ValueError as ve:
            flash("Ainda não temos informações para essa zona. Quem sabe para breve? 🙂", "error")
            return render_template("index.html", **ctx)
        except Exception as exc:
            log.exception("Falha no pipeline")
            flash(f"Erro na análise: {exc}", "error")
            return render_template("index.html", **ctx)

        corr = result["correlation"]
        if corr >= 0.6:
            assessment = ("Os dados mostram forte ligação entre consumo e produção solar. "
                          "Uma comunidade energética pode funcionar muito bem aqui! 💡")
        elif corr >= 0.2:
            assessment = ("Há algum potencial – ainda vale a pena considerar a comunidade "
                          "energética e ajudar o ambiente. 🌿")
        else:
            assessment = ("O potencial é baixo, mas painéis solares individuais ainda podem "
                          "trazer poupança. ⚡")

        ctx.update(
            show_results=True,
            correlation=corr,
            assessment_text=assessment,
            num_residents=result["pop_sum"],
            num_businesses=result["poi_count"],
            plot_png=result["plot_png"],
        )

    return render_template("index.html", **ctx)


# ── FastAPI wrapper para HuggingFace Spaces ------------------------------
app = FastAPI()
app.mount("/", WSGIMiddleware(flask_app))

# ── Execução local -------------------------------------------------------
if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)