#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ForestMODIS — воспроизводимый конвейер оценки динамики и относительной
уязвимости лесного покрова по готовым продуктам MODIS (Collection 6.1).

Этапы:  validate → inventory → plan → download → prepare → landcover →
        treecover → fragmentation → functional → fire → associations →
        vulnerability → sensitivity → export

Запуск:
    python forestmodis.py init --project ./my_project
    python forestmodis.py validate  --config my_project/config.yaml
    python forestmodis.py inventory --config my_project/config.yaml
    python forestmodis.py plan      --config my_project/config.yaml
    python forestmodis.py download  --config my_project/config.yaml
    python forestmodis.py run       --config my_project/config.yaml --resume
    python forestmodis.py run       --config my_project/config.yaml --from-stage functional --to-stage export
    python forestmodis.py sensitivity --config my_project/config.yaml
    python forestmodis.py export    --config my_project/config.yaml
    python forestmodis.py interactive

Скрипт автономный (один файл). Научная интерпретация результатов остаётся
за исследователем: программа рассчитывает показатели и формирует
прослеживаемый пакет материалов.
"""
from __future__ import annotations

import argparse
import copy
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__version__ = "1.0.0"

warnings.filterwarnings("ignore", category=RuntimeWarning)

LOG = logging.getLogger("forestmodis")

# =============================================================================
# 1. Константы MODIS
# =============================================================================
SINU_WKT = ("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 "
            "+units=m +no_defs")
TILE_SIZE_M = 1111950.5196666666
SINU_X0 = -20015109.355798
SINU_Y0 = 10007554.677899
NODATA_U8 = 255
FOREST_CLOSED = [1, 3, 4, 5]
FOREST_SPARSE = [8, 9]

IGBP_NAMES_RU = {
    1: "Вечнозелёные хвойные леса",
    2: "Вечнозелёные широколиственные леса",
    3: "Листопадные хвойные леса (преимущественно лиственничные по IGBP)",
    4: "Листопадные лиственные леса (широко- и мелколиственные)",
    5: "Смешанные леса",
    6: "Сомкнутые кустарники",
    7: "Открытые кустарники",
    8: "Редколесья 30–60% (IGBP Woody Savannas)",
    9: "Разреженные древесные сообщества 10–30% (IGBP Savannas)",
    10: "Травянистые сообщества",
    11: "Постоянные водно-болотные угодья",
    12: "Пашни",
    13: "Застроенные территории",
    14: "Мозаика пашен и естественной растительности",
    15: "Снег и лёд",
    16: "Лишённые растительности",
    17: "Водные объекты",
}
IGBP_NAMES_EN = {
    1: "Evergreen needleleaf forests", 2: "Evergreen broadleaf forests",
    3: "Deciduous needleleaf forests", 4: "Deciduous broadleaf forests",
    5: "Mixed forests", 6: "Closed shrublands", 7: "Open shrublands",
    8: "Woody savannas (tree cover 30-60%)", 9: "Savannas (tree cover 10-30%)",
    10: "Grasslands", 11: "Permanent wetlands", 12: "Croplands",
    13: "Urban and built-up", 14: "Cropland/natural vegetation mosaic",
    15: "Snow and ice", 16: "Barren", 17: "Water bodies",
}
# ранг «древесности» для проверки согласованности переходов с MOD44B
WOODINESS_RANK = {1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 8: 2, 9: 1}


# =============================================================================
# 2. Реестр продуктов (версионируемый; значения сверять с User Guide)
# =============================================================================
def _bits(a, start, n):
    return (a.astype(np.int64) >> start) & ((1 << n) - 1)


def qa_mcd12q1(raw, profile):
    qc = raw["QC"]
    allowed = {"strict": [0], "standard": [0, 8, 9],
               "relaxed": [0, 5, 7, 8, 9]}[profile]
    return np.isin(qc, allowed)


def qa_mod44b(raw, profile):
    ok = np.ones(raw["Percent_Tree_Cover"].shape, bool)
    if profile == "strict" and "Quality" in raw:
        ok &= raw["Quality"] == 0
    return ok


def qa_mod13(raw, profile):
    rel = raw["250m 16 days pixel reliability"]
    allowed = {"strict": [0], "standard": [0, 1], "relaxed": [0, 1, 2]}[profile]
    return np.isin(rel, allowed)


def _scf_cloud(qc, profile, scf_sets):
    scf = _bits(qc, 5, 3)
    ok = np.isin(scf, scf_sets[profile])
    if profile == "strict":
        cloud = _bits(qc, 3, 2)
        ok &= np.isin(cloud, [0, 3])
    return ok


def qa_mcd15(raw, profile):
    return _scf_cloud(raw["FparLai_QC"], profile,
                      {"strict": [0], "standard": [0, 1],
                       "relaxed": [0, 1, 2, 3]})


def qa_mod17(raw, profile):
    if profile == "relaxed":
        return np.ones(raw["Psn_QC_500m"].shape, bool)
    return _scf_cloud(raw["Psn_QC_500m"], profile,
                      {"strict": [0, 1], "standard": [0, 1, 2, 3]})


def qa_mod16(raw, profile):
    if profile == "relaxed":
        return np.ones(raw["ET_QC_500m"].shape, bool)
    return _scf_cloud(raw["ET_QC_500m"], profile,
                      {"strict": [0, 1], "standard": [0, 1, 2, 3]})


def _qa_lst(qc, profile):
    mand = _bits(qc, 0, 2)
    err = _bits(qc, 6, 2)
    if profile == "strict":
        return mand == 0
    if profile == "standard":
        return np.isin(mand, [0, 1]) & (err <= 1)
    return np.isin(mand, [0, 1])


def qa_mod11(raw, profile):
    return {"LST_Day_1km": _qa_lst(raw["QC_Day"], profile),
            "LST_Night_1km": _qa_lst(raw["QC_Night"], profile)}


def qa_mcd64(raw, profile):
    qa = raw["QA"]
    ok = (_bits(qa, 0, 1) == 1) & (_bits(qa, 1, 1) == 1)
    if profile == "strict":
        ok &= _bits(qa, 2, 1) == 0
    return ok


PRODUCTS: Dict[str, Dict[str, Any]] = {
    "MCD12Q1": dict(
        title="Land Cover Type Yearly", cadence="annual", start_doy=1,
        tile_pixels=2400, native_res_m=463.3127, platform="combined",
        layers={"LC_Type1": dict(kind="categorical", scale=1, valid=(1, 17),
                                 fill=255, units="class"),
                "QC": dict(kind="qa"), "LW": dict(kind="qa")},
        qa=qa_mcd12q1, required=True,
        doc="https://lpdaac.usgs.gov/documents/1409/MCD12_User_Guide_V61.pdf"),
    "MOD44B": dict(
        title="Vegetation Continuous Fields Yearly", cadence="annual",
        start_doy=65, tile_pixels=4800, native_res_m=231.6564,
        platform="terra",
        layers={"Percent_Tree_Cover": dict(kind="continuous", scale=1.0,
                                           valid=(0, 100), fill=253,
                                           units="%"),
                "Percent_Tree_Cover_SD": dict(kind="continuous", scale=0.01,
                                              valid=(0, 10000), units="%",
                                              optional=True),
                "Quality": dict(kind="qa", optional=True)},
        qa=qa_mod44b, required=True,
        doc="https://lpdaac.usgs.gov/products/mod44bv061/"),
    "MOD13Q1": dict(
        title="Terra Vegetation Indices 16-Day", cadence=16, start_doy=1,
        tile_pixels=4800, native_res_m=231.6564, platform="terra",
        layers={"250m 16 days NDVI": dict(kind="continuous", scale=1e-4,
                                          valid=(-2000, 10000), units="1"),
                "250m 16 days EVI": dict(kind="continuous", scale=1e-4,
                                         valid=(-2000, 10000), units="1"),
                "250m 16 days pixel reliability": dict(kind="qa"),
                "250m 16 days composite day of the year": dict(kind="doy")},
        qa=qa_mod13, required=True,
        doc="https://lpdaac.usgs.gov/products/mod13q1v061/"),
    "MYD13Q1": dict(
        title="Aqua Vegetation Indices 16-Day", cadence=16, start_doy=9,
        tile_pixels=4800, native_res_m=231.6564, platform="aqua",
        layers={"250m 16 days NDVI": dict(kind="continuous", scale=1e-4,
                                          valid=(-2000, 10000), units="1"),
                "250m 16 days EVI": dict(kind="continuous", scale=1e-4,
                                         valid=(-2000, 10000), units="1"),
                "250m 16 days pixel reliability": dict(kind="qa"),
                "250m 16 days composite day of the year": dict(kind="doy")},
        qa=qa_mod13, required=False,
        doc="https://lpdaac.usgs.gov/products/myd13q1v061/"),
    "MCD15A2H": dict(
        title="LAI/FPAR 8-Day", cadence=8, start_doy=1, tile_pixels=2400,
        native_res_m=463.3127, platform="combined",
        layers={"Lai_500m": dict(kind="continuous", scale=0.1,
                                 valid=(0, 100), units="m2/m2"),
                "Fpar_500m": dict(kind="continuous", scale=0.01,
                                  valid=(0, 100), units="1"),
                "FparLai_QC": dict(kind="qa")},
        qa=qa_mcd15, required=True,
        doc="https://lpdaac.usgs.gov/products/mcd15a2hv061/"),
    "MOD17A2HGF": dict(
        title="Gross Primary Productivity 8-Day Gap-Filled", cadence=8,
        start_doy=1, tile_pixels=2400, native_res_m=463.3127,
        platform="terra",
        layers={"Gpp_500m": dict(kind="accumulated", scale=1e-4,
                                 valid=(0, 30000), units="kgC/m2/interval"),
                "Psn_QC_500m": dict(kind="qa")},
        qa=qa_mod17, required=True,
        doc="https://lpdaac.usgs.gov/products/mod17a2hgfv061/"),
    "MOD16A2GF": dict(
        title="Evapotranspiration 8-Day Gap-Filled", cadence=8, start_doy=1,
        tile_pixels=2400, native_res_m=463.3127, platform="terra",
        layers={"ET_500m": dict(kind="accumulated", scale=0.1,
                                valid=(-32767, 32700), units="kg/m2/interval"),
                "PET_500m": dict(kind="accumulated", scale=0.1,
                                 valid=(-32767, 32700), units="kg/m2/interval"),
                "ET_QC_500m": dict(kind="qa")},
        qa=qa_mod16, required=True,
        doc="https://lpdaac.usgs.gov/products/mod16a2gfv061/"),
    "MOD11A2": dict(
        title="Terra LST/Emissivity 8-Day", cadence=8, start_doy=1,
        tile_pixels=1200, native_res_m=926.6254, platform="terra",
        layers={"LST_Day_1km": dict(kind="continuous", scale=0.02,
                                    valid=(7500, 65535), fill=0, units="K"),
                "LST_Night_1km": dict(kind="continuous", scale=0.02,
                                      valid=(7500, 65535), fill=0, units="K"),
                "QC_Day": dict(kind="qa"), "QC_Night": dict(kind="qa")},
        qa=qa_mod11, required=True,
        doc="https://lpdaac.usgs.gov/products/mod11a2v061/"),
    "MYD11A2": dict(
        title="Aqua LST/Emissivity 8-Day", cadence=8, start_doy=1,
        tile_pixels=1200, native_res_m=926.6254, platform="aqua",
        layers={"LST_Day_1km": dict(kind="continuous", scale=0.02,
                                    valid=(7500, 65535), fill=0, units="K"),
                "LST_Night_1km": dict(kind="continuous", scale=0.02,
                                      valid=(7500, 65535), fill=0, units="K"),
                "QC_Day": dict(kind="qa"), "QC_Night": dict(kind="qa")},
        qa=qa_mod11, required=False,
        doc="https://lpdaac.usgs.gov/products/myd11a2v061/"),
    "MCD64A1": dict(
        title="Burned Area Monthly", cadence="monthly", start_doy=1,
        tile_pixels=2400, native_res_m=463.3127, platform="combined",
        layers={"Burn Date": dict(kind="event"),
                "Burn Date Uncertainty": dict(kind="qa", optional=True),
                "QA": dict(kind="qa")},
        qa=qa_mcd64, required=True,
        doc="https://lpdaac.usgs.gov/products/mcd64a1v061/"),
}
# версии масштабных коэффициентов в атрибутах HDF: MOD13 хранит делитель
DIVIDE_CONVENTION = {"MOD13Q1", "MYD13Q1"}


# =============================================================================
# 3. Конфигурация по умолчанию
# =============================================================================
DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {"name": "forest_modis_project", "random_seed": 42,
                "root": "."},
    "inputs": {"aoi": None, "zones": None, "zones_field": None,
               "validation": None, "buffer_km": 0},
    "period": {"start_year": 2003, "end_year": 2024,
               "comparison_baseline": [2003, 2005],
               "anomaly_baseline": [2003, 2012]},
    "seasons": {"growing": ["05-01", "09-30"], "summer": ["06-01", "08-31"]},
    "grid": {"resolution_m": 500, "crs": "auto_albers"},
    "data": {"backend": "earthdata_https", "collection_version": "061",
             "mode": "online", "fail_on_missing_required": True,
             "products": ["MCD12Q1", "MOD44B", "MOD13Q1", "MYD13Q1",
                          "MCD15A2H", "MOD17A2HGF", "MOD16A2GF", "MOD11A2",
                          "MYD11A2", "MCD64A1"],
             "download_workers": 4, "retries": 5, "timeout_s": 300,
             "cmr_url": "https://cmr.earthdata.nasa.gov/search",
             # источник гранул: auto (проверка сети), lpdaac (CMR + LP DAAC
             # Cloud) или laads (LAADS DAAC; нужен токен Earthdata)
             "source": "auto",
             "laads_url": "https://ladsweb.modaps.eosdis.nasa.gov",
             "search_workers": 6,
             # GEE-бэкенд: те же продукты MODIS C6.1 без скачивания гранул
             "gee_project": None,     # проект Google Cloud (Earth Engine)
             "gee_gpp": "annual",     # annual (MOD17A3HGF, 2003-2024) | growing
             "gee_workers": 4},
    "quality": {"profile": "standard", "min_season_coverage": 0.5,
                "min_window_valid_fraction": 0.95, "interpolate": False},
    "landcover": {"classes": [1, 3, 4, 5, 8, 9],
                  "closed_classes": FOREST_CLOSED,
                  "sparse_classes": FOREST_SPARSE,
                  "persistence_before": 3, "persistence_after": 3,
                  "ptc_consistency_delta_pp": 5.0,
                  "max_transition_events_export": 2_000_000},
    "treecover": {"primary_threshold_pct": 30,
                  "sensitivity_thresholds_pct": [10, 20, 30, 40, 50, 60],
                  "decline_min_pp": 10.0, "decline_k_mad": 2.0,
                  "decline_persistence_years": 2,
                  "recovery_tolerance_pp": 5.0, "pre_event_years": 3},
    "fragmentation": {"main_window_km": 25,
                      "sensitivity_windows_km": [10, 50],
                      "neighborhood": 8, "edge_depth_cells": 1,
                      "sensitivity_edge_depths": [2, 3],
                      "count_outer_boundary_in_ed": False,
                      "per_class": False, "enn": True,
                      "masks": ["ptc", "closed", "sparse", "woody"]},
    "functional": {"variables": ["evi_terra", "evi_aqua", "lai", "fpar",
                                 "gpp", "etpet"],
                   "adverse_percentile": 10, "decline_z": -1.5,
                   "sustained_years": 2, "recovery_sd": 1.0},
    "lst": {"role": "explanatory", "combine_platforms": False,
            "high_percentile": 90},
    "fire": {"prefire_mask": "woody_or_ptc", "recovery_years_max": 5,
             "recovery_tolerance": 0.05},
    "associations": {"lags": [0, 1, 2], "min_years": 8},
    "statistics": {"min_valid_years": 15, "alpha": 0.05,
                   "multiple_testing": "benjamini_hochberg",
                   "autocorrelation_check": True},
    "vulnerability": {
        "status": "provisional",           # 'undefined' → интеграл не считается
        "scenario_id": "provisional_v1",
        "stratify_by": ["dominant_class", "zone"],
        "min_stratum_windows": 20,
        "only_significant_trends": False,
        "weights": {"S_cover": 1, "S_config": 1, "F": 1},
        "s_cover_metrics": ["ptc_mean", "pland"],
        "s_config_metrics": ["cpland", "cohesion", "mesh"],
        "f_metrics": ["evi_terra", "lai", "gpp", "etpet",
                      "functional_nonrecovery"],
        "fragmentation_mask": "ptc",
        "top_fraction": 0.2},
    "sensitivity": {"run": ["threshold", "window", "edge_depth",
                            "persistence", "baseline", "aggregation",
                            "lst_role"],
                    "persistence_values": [2, 3, 5],
                    "include_qa_profiles": False},
    "reporting": {"reference_years": [2004, 2009, 2014, 2019, 2024],
                  "periods": [[2003, 2007], [2008, 2012], [2013, 2017],
                              [2018, 2022], [2023, 2024]],
                  "languages": ["ru", "en"], "figures": True},
    "execution": {"resume": True, "workers": 4, "memory_limit_gb": 12,
                  "block_rows": 256, "keep_intermediate": True},
}

STAGES = ["prepare", "landcover", "treecover", "fragmentation", "functional",
          "fire", "associations", "vulnerability", "export"]


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str]) -> dict:
    import yaml
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        p = Path(path).resolve()
        with open(p, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        cfg = deep_update(cfg, user)
        root = cfg["project"].get("root") or "."
        cfg["project"]["root"] = str((p.parent / root).resolve())
        cfg["_config_path"] = str(p)
    else:
        cfg["project"]["root"] = str(Path(".").resolve())
    return cfg


def config_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


class ConfigError(Exception):
    pass


def validate_config(cfg: dict) -> List[str]:
    """Проверка конфигурации до скачивания. Возвращает предупреждения."""
    warn = []
    p = cfg["period"]
    if p["start_year"] > p["end_year"]:
        raise ConfigError("period.start_year > period.end_year")
    for key in ("comparison_baseline", "anomaly_baseline"):
        b = p[key]
        if not (p["start_year"] <= b[0] <= b[1] <= p["end_year"]):
            raise ConfigError(f"period.{key} вне периода исследования")
    if p["anomaly_baseline"][1] - p["anomaly_baseline"][0] + 1 < 5:
        warn.append("Базовый период аномалий короче 5 лет: стандартные "
                    "отклонения и процентили будут неустойчивы.")
    if cfg["quality"]["profile"] not in ("strict", "standard", "relaxed"):
        raise ConfigError("quality.profile: strict|standard|relaxed")
    for prod in cfg["data"]["products"]:
        if prod not in PRODUCTS:
            raise ConfigError(f"Неизвестный продукт {prod}")
    if str(cfg["data"]["collection_version"]) != "061":
        raise ConfigError("Поддерживается только Collection 061; смешивание "
                          "версий запрещено.")
    for c in cfg["landcover"]["classes"]:
        if c not in IGBP_NAMES_RU:
            raise ConfigError(f"landcover.classes: неизвестный класс {c}")
    if cfg["fragmentation"]["neighborhood"] not in (4, 8):
        raise ConfigError("fragmentation.neighborhood: 4 или 8")
    if cfg["lst"]["role"] not in ("explanatory", "stress_component"):
        raise ConfigError("lst.role: explanatory|stress_component")
    if cfg["vulnerability"]["status"] in (None, "", "undefined"):
        warn.append("vulnerability.status не задан: интегральные индексы "
                    "рассчитываться не будут (только компоненты).")
    n_years = p["end_year"] - p["start_year"] + 1
    if n_years < cfg["statistics"]["min_valid_years"]:
        warn.append(f"Период {n_years} лет меньше statistics.min_valid_years="
                    f"{cfg['statistics']['min_valid_years']}: тренды "
                    "получат статус insufficient_years.")
    if p["end_year"] > 2020:
        warn.append("MCD12Q1 после 2021 г. имеет повышенную неопределённость "
                    "(обучающая база не обновлялась); результаты 2021+ "
                    "маркируются отдельно.")
    for prod in ("MCD12Q1", "MOD44B", "MOD13Q1", "MCD15A2H", "MOD17A2HGF",
                 "MOD16A2GF", "MOD11A2", "MCD64A1"):
        if prod not in cfg["data"]["products"]:
            warn.append(f"Продукт {prod} исключён: зависимые этапы будут "
                        "пропущены.")
    if cfg["data"].get("backend") == "gee":
        if cfg["data"].get("gee_gpp") not in (None, "annual", "growing"):
            raise ConfigError("data.gee_gpp: annual | growing")
        if not (cfg["data"].get("gee_project")
                or os.environ.get("EE_PROJECT")
                or os.environ.get("GOOGLE_CLOUD_PROJECT")):
            warn.append("backend=gee: не задан data.gee_project — будет "
                        "использован проект из настроек earthengine.")
        for prod in cfg["data"]["products"]:
            if prod not in GEE_ASSETS:
                raise ConfigError(f"продукт {prod} недоступен в Earth Engine")
    return warn


# =============================================================================
# 4. Проект, журнал, провенанс
# =============================================================================
class Project:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.root = Path(cfg["project"]["root"])
        self.raw = self.root / "data" / "raw"
        self.inv = self.root / "data" / "inventory"
        self.work = self.root / "work"
        self.state = self.work / "state"
        self.out = self.root / "outputs"
        self.rasters = self.out / "rasters"
        self.tables = self.out / "tables"
        self.vectors = self.out / "vectors"
        self.figures = self.out / "figures"
        self.pkg = self.out / "analysis_package"
        self.logs = self.root / "logs"
        for d in (self.raw, self.inv, self.work, self.state, self.rasters,
                  self.tables, self.vectors, self.figures, self.pkg,
                  self.logs):
            d.mkdir(parents=True, exist_ok=True)
        self.run_id = (dt.datetime.now().strftime("%Y%m%dT%H%M%S") + "_" +
                       config_hash(public_config(cfg))[:8])
        self._grid = None

    # --- годы
    @property
    def years(self) -> List[int]:
        p = self.cfg["period"]
        return list(range(p["start_year"], p["end_year"] + 1))

    def stage_state_path(self, stage):
        return self.state / f"{stage}.json"

    def grid(self) -> "Grid":
        if self._grid is None:
            self._grid = Grid.load_or_create(self)
        return self._grid


def public_config(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


class SecretFilter(logging.Filter):
    PAT = re.compile(r"(password|token|authorization|bearer)\s*[=:]\s*\S+",
                     re.I)

    def filter(self, record):
        record.msg = self.PAT.sub(r"\1=***", str(record.msg))
        return True


def setup_logging(prj: Optional[Project], verbose=False):
    LOG.handlers.clear()
    LOG.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                            "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(fmt)
    sh.addFilter(SecretFilter())
    LOG.addHandler(sh)
    if prj is not None:
        fh = logging.FileHandler(prj.logs / f"run_{prj.run_id}.log",
                                 encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        fh.addFilter(SecretFilter())
        LOG.addHandler(fh)


def env_versions() -> dict:
    v = {"forestmodis": __version__, "python": sys.version.split()[0],
         "platform": platform.platform()}
    for m in ("numpy", "pandas", "scipy", "rasterio", "geopandas", "shapely",
              "pyproj", "pyhdf", "earthaccess", "matplotlib", "pyarrow",
              "yaml"):
        try:
            mod = __import__(m)
            v[m] = getattr(mod, "__version__", "?")
        except Exception:
            v[m] = None
    try:
        import rasterio
        v["gdal"] = rasterio.__gdal_version__
    except Exception:
        pass
    try:
        here = Path(__file__).resolve().parent
        v["git_commit"] = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True,
            text=True, timeout=10).stdout.strip() or None
    except Exception:
        v["git_commit"] = None
    return v


def sha256_file(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# =============================================================================
# 5. Пространственные данные: AOI, сетка, тайлы
# =============================================================================
def pick_files_dialog(title, multiple=False, filetypes=None):
    """Окно Проводника для выбора файлов (tkinter)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        ft = filetypes or [("Векторные данные", "*.gpkg *.geojson *.json *.shp"),
                           ("Все файлы", "*.*")]
        if multiple:
            res = list(filedialog.askopenfilenames(title=title, filetypes=ft))
        else:
            r = filedialog.askopenfilename(title=title, filetypes=ft)
            res = [r] if r else []
        root.destroy()
        return res
    except Exception as e:
        LOG.debug(f"Диалог выбора файлов недоступен: {e}")
        return []


def pick_folder_dialog(title: str) -> Optional[str]:
    """Окно Проводника для выбора папки (tkinter)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        r = filedialog.askdirectory(title=title, mustexist=False)
        root.destroy()
        return r or None
    except Exception as e:
        LOG.debug(f"Диалог выбора папки недоступен: {e}")
        return None


def in_notebook() -> bool:
    """Запуск из Jupyter/IPython-ядра."""
    return "ipykernel" in sys.modules


def _ask(prompt: str, default: str = "") -> str:
    try:
        v = input(prompt).strip()
    except EOFError:
        v = ""
    return v or default


def read_aoi(cfg: dict):
    import geopandas as gpd
    path = cfg["inputs"].get("aoi")
    if not path:
        sel = pick_files_dialog("Выберите полигон территории исследования")
        if not sel:
            raise ConfigError("inputs.aoi не задан и файл не выбран")
        path = sel[0]
        cfg["inputs"]["aoi"] = path
    p = Path(path)
    if not p.is_absolute():
        p = Path(cfg["project"]["root"]) / p
    if not p.exists():
        raise ConfigError(f"файл территории не найден: {p}. Положите полигон "
                          "туда или укажите путь в inputs.aoi (config.yaml)")
    gdf = gpd.read_file(p)
    if gdf.crs is None:
        raise ConfigError(f"AOI {p}: не задана система координат")
    if gdf.empty:
        raise ConfigError(f"AOI {p}: пустой слой")
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    invalid = ~gdf.geometry.is_valid
    fixes = []
    if invalid.any():
        fixes.append(f"исправлено невалидных геометрий: {int(invalid.sum())}")
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid].geometry.buffer(0)
    geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") \
        else gdf.geometry.unary_union
    out = gpd.GeoDataFrame({"aoi": [1]}, geometry=[geom], crs=gdf.crs)
    return out, str(p), fixes


def auto_albers(aoi_ll) -> str:
    minx, miny, maxx, maxy = aoi_ll.total_bounds
    dlat = maxy - miny
    lat1 = miny + dlat / 6.0
    lat2 = maxy - dlat / 6.0
    lat0 = (miny + maxy) / 2
    lon0 = round((minx + maxx) / 2, 1)
    return (f"+proj=aea +lat_1={lat1:.2f} +lat_2={lat2:.2f} +lat_0={lat0:.2f} "
            f"+lon_0={lon0} +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs")


@dataclass
class Grid:
    crs: str
    transform: Tuple[float, ...]
    width: int
    height: int
    res: float
    aoi_mask: np.ndarray          # True внутри AOI
    zones: Optional[np.ndarray]   # id зоны (int32), 0 = нет
    zone_names: Dict[int, str]

    @property
    def affine(self):
        from affine import Affine
        return Affine(*self.transform[:6])

    @property
    def cell_ha(self):
        return self.res * self.res / 10000.0

    def profile(self, dtype="float32", nodata=np.nan):
        return dict(driver="GTiff", dtype=dtype, count=1, width=self.width,
                    height=self.height, crs=self.crs, transform=self.affine,
                    nodata=nodata, tiled=True, blockxsize=256,
                    blockysize=256, compress="deflate", predictor=1)

    @staticmethod
    def load_or_create(prj: Project) -> "Grid":
        import geopandas as gpd
        import rasterio
        from rasterio import features
        from rasterio.transform import from_origin
        gpath = prj.work / "grid.json"
        mpath = prj.work / "aoi_mask.tif"
        zpath = prj.work / "zones.tif"
        cfg = prj.cfg
        aoi, aoi_path, _ = read_aoi(cfg)
        key = config_hash({"aoi": sha256_file(Path(aoi_path)),
                           "grid": cfg["grid"],
                           "buffer": cfg["inputs"].get("buffer_km", 0),
                           "zones": cfg["inputs"].get("zones"),
                           "zf": cfg["inputs"].get("zones_field")})
        if gpath.exists() and mpath.exists():
            meta = json.loads(gpath.read_text(encoding="utf-8"))
            if meta.get("key") == key:
                with rasterio.open(mpath) as src:
                    m = src.read(1).astype(bool)
                zones = None
                if zpath.exists():
                    with rasterio.open(zpath) as src:
                        zones = src.read(1)
                return Grid(meta["crs"], tuple(meta["transform"]),
                            meta["width"], meta["height"], meta["res"], m,
                            zones, {int(k): v for k, v in
                                    meta.get("zone_names", {}).items()})
        aoi_ll = aoi.to_crs(4326)
        crs = cfg["grid"]["crs"]
        if crs in (None, "auto_albers", "regional_equal_area"):
            crs = auto_albers(aoi_ll)
        res = float(cfg["grid"]["resolution_m"])
        a = aoi.to_crs(crs)
        buf = float(cfg["inputs"].get("buffer_km") or 0) * 1000
        minx, miny, maxx, maxy = a.buffer(buf).total_bounds if buf else \
            a.total_bounds
        minx = math.floor(minx / res) * res
        miny = math.floor(miny / res) * res
        maxx = math.ceil(maxx / res) * res
        maxy = math.ceil(maxy / res) * res
        w = int(round((maxx - minx) / res))
        h = int(round((maxy - miny) / res))
        tr = from_origin(minx, maxy, res, res)
        mask = features.geometry_mask(a.geometry, (h, w), tr, invert=True)
        zones, zone_names = None, {}
        if cfg["inputs"].get("zones"):
            zp = Path(cfg["inputs"]["zones"])
            if not zp.is_absolute():
                zp = Path(cfg["project"]["root"]) / zp
            zg = gpd.read_file(zp).to_crs(crs)
            fld = cfg["inputs"].get("zones_field")
            names = zg[fld].astype(str).tolist() if fld else \
                [f"zone_{i+1}" for i in range(len(zg))]
            uniq = sorted(set(names))
            zone_names = {i + 1: n for i, n in enumerate(uniq)}
            inv = {n: i for i, n in zone_names.items()}
            shapes = [(g, inv[n]) for g, n in zip(zg.geometry, names)]
            zones = features.rasterize(shapes, (h, w), transform=tr, fill=0,
                                       dtype="int32")
        g = Grid(crs, tuple(tr)[:6], w, h, res, mask, zones, zone_names)
        gpath.write_text(json.dumps({"key": key, "crs": crs,
                                     "transform": list(tuple(tr)[:6]),
                                     "width": w, "height": h, "res": res,
                                     "zone_names": zone_names}, indent=1),
                         encoding="utf-8")
        write_raster(mpath, mask.astype("uint8"), g, nodata=None,
                     dtype="uint8")
        if zones is not None:
            write_raster(zpath, zones, g, nodata=None, dtype="int32")
        LOG.info(f"Сетка: {w}×{h} ячеек по {res:.0f} м, CRS: {crs}")
        return g


def aoi_sinusoidal_tiles(cfg) -> List[str]:
    from shapely.geometry import box
    aoi, _, _ = read_aoi(cfg)
    buf = float(cfg["inputs"].get("buffer_km") or 0) * 1000
    ll = aoi.to_crs(4326)
    geom_ll = ll.geometry.iloc[0]
    if buf:
        a = aoi.to_crs(auto_albers(ll)).buffer(buf)
        geom_ll = a.to_crs(4326).iloc[0]
    import geopandas as gpd
    sg = gpd.GeoSeries([geom_ll.segmentize(0.25)], crs=4326).to_crs(SINU_WKT)
    g = sg.iloc[0]
    minx, miny, maxx, maxy = g.bounds
    h0 = int((minx - SINU_X0) // TILE_SIZE_M)
    h1 = int((maxx - SINU_X0) // TILE_SIZE_M)
    v0 = int((SINU_Y0 - maxy) // TILE_SIZE_M)
    v1 = int((SINU_Y0 - miny) // TILE_SIZE_M)
    tiles = []
    for hh in range(max(h0, 0), min(h1, 35) + 1):
        for vv in range(max(v0, 0), min(v1, 17) + 1):
            x0 = SINU_X0 + hh * TILE_SIZE_M
            y1 = SINU_Y0 - vv * TILE_SIZE_M
            if g.intersects(box(x0, y1 - TILE_SIZE_M, x0 + TILE_SIZE_M, y1)):
                tiles.append(f"h{hh:02d}v{vv:02d}")
    return tiles


def tile_transform(tile: str, npix: int):
    from affine import Affine
    hh, vv = int(tile[1:3]), int(tile[4:6])
    px = TILE_SIZE_M / npix
    return Affine(px, 0, SINU_X0 + hh * TILE_SIZE_M, 0, -px,
                  SINU_Y0 - vv * TILE_SIZE_M)


def write_raster(path: Path, arr: np.ndarray, grid: Grid, nodata=np.nan,
                 dtype=None, tags: Optional[dict] = None):
    import rasterio
    dtype = dtype or str(arr.dtype)
    prof = grid.profile(dtype=dtype, nodata=nodata)
    if nodata is None:
        prof.pop("nodata")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
        try:
            dst.build_overviews([2, 4, 8, 16])
        except Exception:
            pass
    os.replace(tmp, path)


def read_raster(path: Path, window=None) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as src:
        a = src.read(1, window=window)
        nd = src.nodata
    if a.dtype.kind == "f" and nd is not None and not np.isnan(nd):
        a = np.where(a == nd, np.nan, a)
    return a


# =============================================================================
# 6. Календари MODIS и сезоны
# =============================================================================
def year_len(y):
    return 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365


def md_to_doy(y, md: str) -> int:
    m, d = map(int, md.split("-"))
    return (dt.date(y, m, d) - dt.date(y, 1, 1)).days + 1


def composite_starts(prod: str, year: int) -> List[int]:
    spec = PRODUCTS[prod]
    cad = spec["cadence"]
    if cad == "annual":
        return [spec["start_doy"]]
    if cad == "monthly":
        return [(dt.date(year, m, 1) - dt.date(year, 1, 1)).days + 1
                for m in range(1, 13)]
    return list(range(spec["start_doy"], year_len(year) + 1, cad))


def composite_interval(prod: str, year: int, start: int) -> Tuple[int, int]:
    spec = PRODUCTS[prod]
    cad = spec["cadence"]
    yl = year_len(year)
    if cad == "annual":
        return 1, yl
    if cad == "monthly":
        d = dt.date(year, 1, 1) + dt.timedelta(start - 1)
        nxt = dt.date(year + (d.month == 12), d.month % 12 + 1, 1)
        return start, start + (nxt - d).days - 1
    return start, min(start + cad - 1, yl)


def needed_composites(cfg, prod: str, year: int) -> List[int]:
    """Только композиты, пересекающие нужные сезоны (экономия загрузки)."""
    cad = PRODUCTS[prod]["cadence"]
    if cad in ("annual", "monthly"):
        return composite_starts(prod, year)
    s0 = min(md_to_doy(year, cfg["seasons"][k][0]) for k in cfg["seasons"])
    s1 = max(md_to_doy(year, cfg["seasons"][k][1]) for k in cfg["seasons"])
    out = []
    for st in composite_starts(prod, year):
        a, b = composite_interval(prod, year, st)
        if b >= s0 and a <= s1:
            out.append(st)
    return out


def season_overlap(year, a, b, season):
    s0 = md_to_doy(year, season[0])
    s1 = md_to_doy(year, season[1])
    ov = max(0, min(b, s1) - max(a, s0) + 1)
    return ov, (b - a + 1)


# =============================================================================
# 7. Инвентаризация (NASA CMR) и загрузка (Earthdata HTTPS)
# =============================================================================
GRANULE_RE = re.compile(
    r"(?P<prod>M[OYC]D\w+?)\.A(?P<year>\d{4})(?P<doy>\d{3})\.(?P<tile>h\d{2}v\d{2})"
    r"\.(?P<ver>\d{3})\.(?P<ptime>\d{13})\.hdf$")


def parse_granule_name(name: str) -> Optional[dict]:
    m = GRANULE_RE.search(name)
    if not m:
        return None
    d = m.groupdict()
    d["year"] = int(d["year"])
    d["doy"] = int(d["doy"])
    return d


def cmr_search(cfg, prod, tiles, year_from, year_to, raw_dir: Path) -> List[dict]:
    import requests
    url = cfg["data"]["cmr_url"].rstrip("/") + "/granules.umm_json"
    ll_bbox = None
    try:
        aoi, _, _ = read_aoi(cfg)
        b = aoi.to_crs(4326).total_bounds
        ll_bbox = f"{b[0]:.4f},{b[1]:.4f},{b[2]:.4f},{b[3]:.4f}"
    except Exception:
        pass
    params = {"short_name": prod, "version": cfg["data"]["collection_version"],
              "page_size": 2000,
              "temporal": f"{year_from}-01-01T00:00:00Z,"
                          f"{year_to}-12-31T23:59:59Z"}
    if ll_bbox:
        params["bounding_box"] = ll_bbox
    # фильтр по тайлам на стороне CMR (шаблон имени гранулы)
    if tiles:
        params["readable_granule_name[]"] = [f"{prod}.A*.{t}.*" for t in tiles]
        params["options[readable_granule_name][pattern]"] = "true"
    headers = {}
    items, page = [], 0
    raw_dir.mkdir(parents=True, exist_ok=True)
    while True:
        for attempt in range(cfg["data"]["retries"]):
            try:
                r = requests.get(url, params=params, headers=headers,
                                 timeout=(15, 120))
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    raise RuntimeError(f"CMR {prod}: HTTP {r.status_code}: "
                                       f"{r.text[:500]}")
                r.raise_for_status()
                break
            except RuntimeError:
                raise
            except Exception as e:
                wait = 2 ** attempt
                LOG.warning(f"CMR {prod}: {e}; повтор через {wait} с")
                time.sleep(wait)
        else:
            raise RuntimeError(f"CMR недоступен для {prod}")
        js = r.json()
        import gzip
        with gzip.open(raw_dir / f"{prod}_{year_from}_{year_to}_p{page:03d}"
                       ".json.gz", "wt", encoding="utf-8") as fz:
            json.dump(js, fz)
        page += 1
        for it in js.get("items", []):
            umm = it.get("umm", {})
            meta = it.get("meta", {})
            urls = [u.get("URL") for u in umm.get("RelatedUrls", [])
                    if u.get("Type") in ("GET DATA", None) and
                    str(u.get("URL", "")).startswith("https") and
                    str(u.get("URL", "")).endswith(".hdf")]
            if not urls:
                continue
            name = urls[0].rsplit("/", 1)[-1]
            g = parse_granule_name(name)
            if not g or g["tile"] not in tiles:
                continue
            size, cks, alg = None, None, None
            adis = umm.get("DataGranule", {}).get(
                "ArchiveAndDistributionInformation", [])
            for ad in adis:
                # имя часто «Not provided»: берём запись .hdf или единственную
                if str(ad.get("Name", "")).endswith(".hdf") or len(adis) == 1:
                    size = ad.get("SizeInBytes") or (
                        ad.get("Size", 0) * {"KB": 2.0 ** 10, "MB": 2.0 ** 20,
                                             "GB": 2.0 ** 30}.get(
                            ad.get("SizeUnit", "MB"), 1e6))
                    ck = ad.get("Checksum") or {}
                    cks, alg = ck.get("Value"), ck.get("Algorithm")
            items.append(dict(product=prod, file=name, url=urls[0],
                              tile=g["tile"], year=g["year"], doy=g["doy"],
                              production_time=g["ptime"],
                              concept_id=meta.get("concept-id"),
                              collection_concept_id=meta.get(
                                  "collection-concept-id"),
                              revision=meta.get("revision-id"),
                              size_bytes=size, checksum=cks,
                              checksum_algorithm=alg))
        sa = r.headers.get("CMR-Search-After")
        if not sa or not js.get("items"):
            break
        headers["CMR-Search-After"] = sa
    return items


# ---- проверка сети и альтернативный источник LAADS DAAC ---------------------
LPDAAC_PROBE_URL = (
    "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-public/MOD13Q1.061/"
    "MOD13Q1.A2000049.h20v03.061.2020041155741/"
    "BROWSE.MOD13Q1.A2000049.h20v03.061.2020041155742.1.jpg")


_DOH_CACHE: Dict[str, List[str]] = {}
_DNS_PATCHED = False


def _doh_resolve(host: str) -> List[str]:
    """Адреса IPv4 через DNS-over-HTTPS (Cloudflare, затем Google)."""
    if host in _DOH_CACHE:
        return _DOH_CACHE[host]
    import requests
    for url in ("https://cloudflare-dns.com/dns-query",
                "https://dns.google/resolve"):
        try:
            j = requests.get(url, params={"name": host, "type": "A"},
                             headers={"accept": "application/dns-json"},
                             timeout=10).json()
            ips = [a["data"] for a in j.get("Answer", []) if a.get("type") == 1]
            if ips:
                _DOH_CACHE[host] = ips
                return ips
        except Exception:
            continue
    return []


def enable_dns_fallback():
    """Если системный DNS не находит хост (так бывает у DNS-серверов VPN,
    которые блокируют *.nasa.gov), адрес берётся через DNS-over-HTTPS.
    Подключение остаётся по HTTPS с проверкой сертификата по имени хоста."""
    global _DNS_PATCHED
    if _DNS_PATCHED:
        return
    import socket
    orig = socket.getaddrinfo

    def getaddrinfo(host, port, *args, **kw):
        if isinstance(host, str) and host in _DOH_CACHE:
            out = []
            for ip in _DOH_CACHE[host]:
                out += orig(ip, port, *args, **kw)
            return out
        try:
            return orig(host, port, *args, **kw)
        except socket.gaierror:
            if not isinstance(host, str) or host in (
                    "cloudflare-dns.com", "dns.google"):
                raise
            ips = _doh_resolve(host)
            if not ips:
                raise
            if host not in _DOH_LOGGED:
                _DOH_LOGGED.add(host)
                LOG.warning(f"Системный DNS не нашёл {host}; адрес {ips[0]} "
                            "получен через DNS-over-HTTPS")
            out = []
            for ip in ips:
                out += orig(ip, port, *args, **kw)
            return out

    socket.getaddrinfo = getaddrinfo
    _DNS_PATCHED = True


_DOH_LOGGED: set = set()


def _probe_url(url, params=None, min_bytes=40_000, timeout=(10, 15)) -> Tuple[bool, str]:
    """Читает ответ целиком. Возвращает (успех, описание). Ловит «зависание»
    соединения после первых ~16 КБ (характерно для фильтрации трафика)."""
    import requests
    n = 0
    try:
        with requests.get(url, params=params, stream=True, timeout=timeout,
                          headers={"Accept-Encoding": "identity"}) as r:
            for ch in r.iter_content(8192):
                n += len(ch)
                if n >= min_bytes:
                    break
            return True, f"HTTP {r.status_code}, получено {n} байт"
    except Exception as e:
        return False, f"{type(e).__name__} после {n} байт"


def choose_source(cfg) -> str:
    """auto → lpdaac, если CMR и хранилище LP DAAC отдают данные полностью;
    иначе laads. Явное значение data.source возвращается как есть."""
    enable_dns_fallback()
    src = str(cfg["data"].get("source", "auto")).lower()
    if src in ("lpdaac", "laads"):
        return src
    cmr = cfg["data"]["cmr_url"].rstrip("/") + "/granules.json"
    ok1, m1 = _probe_url(cmr, dict(short_name="MOD13Q1", version="061",
                                   page_size=20))
    ok2, m2 = (_probe_url(LPDAAC_PROBE_URL, min_bytes=30_000) if ok1
               else (False, "не проверялось"))
    LOG.info(f"Проверка сети: CMR — {m1}; LP DAAC — {m2}")
    if ok1 and ok2:
        return "lpdaac"
    laads = cfg["data"].get("laads_url", "https://ladsweb.modaps.eosdis.nasa.gov")
    ok3, m3 = _probe_url(laads.rstrip("/") + "/api/v2/content/details",
                         dict(products="MOD13Q1", collections="61",
                              temporalRanges="2003-01-01..2003-01-02"),
                         min_bytes=20_000)
    LOG.info(f"Проверка сети: LAADS DAAC — {m3}")
    if ok3:
        LOG.warning("Серверы CMR/LP DAAC обрывают передачу данных из вашей сети "
                    "(соединение «зависает» после первых килобайт). Используется "
                    "зеркало LAADS DAAC с теми же файлами MODIS C6.1. Чтобы "
                    "задать источник явно, укажите data.source в config.yaml.")
        return "laads"
    raise RuntimeError(
        "Серверы NASA недоступны из этой сети: и CMR/LP DAAC, и LAADS DAAC "
        "не отвечают или обрывают передачу (подробности в журнале выше). "
        "Включите VPN или прокси (переменные среды "
        "HTTPS_PROXY/HTTP_PROXY учитываются) либо скачайте HDF на другом "
        "компьютере и запустите с data.mode: offline.")


def laads_search(cfg, prod, tiles, years: Sequence[int], raw_dir: Path) -> List[dict]:
    """Поиск гранул через LAADS DAAC API v2 (одна страница = одна дата)."""
    import requests
    base = cfg["data"].get("laads_url",
                           "https://ladsweb.modaps.eosdis.nasa.gov").rstrip("/")
    url = base + "/api/v2/content/details"
    region = None
    try:
        aoi, _, _ = read_aoi(cfg)
        b = aoi.to_crs(4326).total_bounds
        region = (f"[BBOX]W{b[0]:.4f} N{b[3]:.4f} E{b[2]:.4f} S{b[1]:.4f}")
    except Exception:
        pass
    coll = str(int(cfg["data"]["collection_version"]))
    raw_dir.mkdir(parents=True, exist_ok=True)

    def get(params):
        last = None
        for attempt in range(cfg["data"]["retries"]):
            try:
                r = requests.get(url, params=params, timeout=(15, 90))
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    raise RuntimeError(f"LAADS {prod}: HTTP {r.status_code}: "
                                       f"{r.text[:300]}")
                r.raise_for_status()
                return r.json()
            except RuntimeError:
                raise
            except Exception as e:
                last = e
                time.sleep(min(30, 2 ** attempt))
        raise RuntimeError(f"LAADS недоступен для {prod}: {last}")

    queries = []
    for y in years:
        doys = needed_composites(cfg, prod, y)
        if not doys:
            continue
        d0 = dt.date(y, 1, 1) + dt.timedelta(days=min(doys) - 1)
        d1 = dt.date(y, 1, 1) + dt.timedelta(days=max(doys) - 1)
        q = dict(products=prod, collections=coll,
                 temporalRanges=f"{d0.isoformat()}..{d1.isoformat()}")
        if region:
            q["regions"] = region
        queries.append(q)
    pages = []
    with cf.ThreadPoolExecutor(cfg["data"].get("search_workers", 6)) as ex:
        firsts = list(ex.map(get, queries))
        more = []
        for q, js in zip(queries, firsts):
            pages.append(js)
            for k in range(2, int(js.get("n_pages") or 1) + 1):
                more.append(dict(q, page=k))
        pages += list(ex.map(get, more))
    import gzip
    with gzip.open(raw_dir / f"LAADS_{prod}_{min(years)}_{max(years)}.json.gz",
                   "wt", encoding="utf-8") as fz:
        json.dump(pages, fz)
    items, seen = [], set()
    for js in pages:
        for it in js.get("content", []) or []:
            name = it.get("name", "")
            g = parse_granule_name(name)
            if not g or g["tile"] not in tiles or name in seen:
                continue
            seen.add(name)
            items.append(dict(product=prod, file=name,
                              url=it.get("downloadsLink"),
                              tile=g["tile"], year=g["year"], doy=g["doy"],
                              production_time=g["ptime"],
                              concept_id=f"LAADS:{it.get('fileId')}",
                              collection_concept_id=f"LAADS:{prod}.{coll}",
                              revision=it.get("mtime"),
                              size_bytes=it.get("size"),
                              checksum=it.get("md5sum"),
                              checksum_algorithm="MD5" if it.get("md5sum")
                              else None))
    return items


def earthdata_token(cfg) -> str:
    """Токен Earthdata (Bearer) для LAADS: EARTHDATA_TOKEN, файл
    .earthdata_token (в папке проекта или домашней), data.earthdata_token,
    затем получение через earthaccess по логину/паролю, затем запрос."""
    tok = os.environ.get("EARTHDATA_TOKEN")
    if tok:
        return tok.strip()
    for p in (Path(cfg["project"]["root"]) / ".earthdata_token",
              Path.home() / ".earthdata_token"):
        if p.exists():
            t = p.read_text(encoding="utf-8").strip()
            if t:
                return t
    if cfg["data"].get("earthdata_token"):
        return str(cfg["data"]["earthdata_token"]).strip()
    try:
        import earthaccess
        for strategy in ("environment", "netrc"):
            try:
                auth = earthaccess.login(strategy=strategy, persist=False)
                t = (getattr(auth, "token", None) or {}).get("access_token")
                if t:
                    LOG.info(f"Earthdata: токен получен ({strategy})")
                    return t
            except Exception:
                continue
    except ImportError:
        pass
    msg = ("Для LAADS DAAC нужен токен NASA Earthdata. Откройте "
           "https://urs.earthdata.nasa.gov → Generate Token, скопируйте токен и "
           "сохраните его в файл .earthdata_token в папке проекта (или задайте "
           "переменную среды EARTHDATA_TOKEN).")
    if sys.stdin is not None and sys.stdin.isatty() or in_notebook():
        import getpass
        LOG.warning(msg)
        try:
            t = getpass.getpass("Вставьте токен Earthdata (ввод скрыт): ").strip()
        except (EOFError, KeyboardInterrupt):
            t = ""
        if t:
            p = Path(cfg["project"]["root"]) / ".earthdata_token"
            p.write_text(t, encoding="utf-8")
            LOG.info(f"Токен сохранён в {p}")
            return t
    raise RuntimeError(msg)


def laads_session(cfg):
    import requests
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {earthdata_token(cfg)}"
    return s


def local_granules(prj: Project) -> pd.DataFrame:
    rows = []
    for p in prj.raw.rglob("*.hdf"):
        g = parse_granule_name(p.name)
        if g:
            rows.append(dict(product=g["prod"], file=p.name, url=None,
                             tile=g["tile"], year=g["year"], doy=g["doy"],
                             production_time=g["ptime"], local_path=str(p)))
    return pd.DataFrame(rows)


def run_inventory(prj: Project, online: bool = True) -> pd.DataFrame:
    cfg = prj.cfg
    tiles = aoi_sinusoidal_tiles(cfg)
    LOG.info(f"Тайлы MODIS для AOI: {', '.join(tiles)} ({len(tiles)})")
    years = prj.years
    rows = []
    if online and cfg["data"]["mode"] != "offline":
        source = choose_source(cfg)
        LOG.info(f"Источник гранул: {source}")
        for prod in cfg["data"]["products"]:
            t0 = time.time()
            # годовой продукт MOD44B: гранула A{year}065
            y_from = years[0]
            y_to = years[-1] + (1 if PRODUCTS[prod]["start_doy"] > 1 and
                                PRODUCTS[prod]["cadence"] == "annual" else 0)
            if source == "laads":
                items = laads_search(cfg, prod, tiles,
                                     list(range(y_from, y_to + 1)),
                                     prj.inv / "raw")
            else:
                items = cmr_search(cfg, prod, tiles, y_from, y_to,
                                   prj.inv / "raw")
            rows += items
            LOG.info(f"{source.upper()} {prod}: найдено {len(items)} гранул "
                     f"({time.time()-t0:.0f} с)")
        cat = pd.DataFrame(rows)
    else:
        cat = local_granules(prj)
        LOG.info(f"Офлайн-режим: локальных гранул {len(cat)}")
    if cat.empty:
        cat = pd.DataFrame(columns=["product", "file", "url", "tile", "year",
                                    "doy", "production_time"])
    # одна редакция на (продукт, тайл, год, doy): самая поздняя production_time
    cat = (cat.sort_values("production_time")
              .drop_duplicates(["product", "tile", "year", "doy"], keep="last"))
    # только нужные композиты
    need = {(pr, y): set(needed_composites(cfg, pr, y))
            for pr in cfg["data"]["products"] if pr in PRODUCTS for y in years}
    if len(cat):
        keep = [d in need.get((pr, int(y)), ())
                for pr, y, d in zip(cat["product"], cat["year"], cat["doy"])]
        cat = cat[np.array(keep, bool)]
    cat.to_csv(prj.inv / "granule_catalog.csv", index=False)
    # полнота
    av = []
    loc = local_granules(prj)
    locset = set(loc["file"]) if len(loc) else set()
    for prod in cfg["data"]["products"]:
        # тайлы, для которых продукт не выпускается вовсе (например, MCD64A1
        # над Арктикой): исключаются из ожидаемого числа и документируются
        not_prod = []
        if online and cfg["data"]["mode"] != "offline" and len(cat):
            have = set(cat.loc[cat["product"] == prod, "tile"])
            not_prod = sorted(set(tiles) - have) if have else []
        if not_prod:
            LOG.info(f"{prod}: не выпускается для тайлов {', '.join(not_prod)}"
                     " (нет ни одной гранулы за период) — исключены из "
                     "ожидаемого числа")
        n_t = len(tiles) - len(not_prod)
        for y in years:
            exp = len(needed_composites(cfg, prod, y)) * n_t
            sub = cat[(cat["product"] == prod) & (cat["year"] == y)] \
                if len(cat) else cat
            found = len(sub)
            down = int(sum(f in locset for f in sub["file"])) if found else 0
            frac = found / exp if exp else 0
            status = "complete" if found >= exp else (
                "partial" if found else "missing")
            reason = ""
            if status != "complete":
                reason = ("гранулы отсутствуют в каталоге (возможна задержка "
                          "выпуска продукта)" if not found else
                          "часть композитов/тайлов отсутствует")
            av.append(dict(product=prod, version="061", year=y,
                           tiles=n_t,
                           tiles_not_produced=";".join(not_prod),
                           expected_granules=exp,
                           found_granules=found, downloaded_granules=down,
                           completeness_fraction=round(frac, 4),
                           status=status, reason=reason))
    avdf = pd.DataFrame(av)
    avdf.to_csv(prj.inv / "data_availability.csv", index=False)
    miss = avdf[avdf.status != "complete"]
    if len(miss):
        LOG.warning(f"Неполные продукт-годы: {len(miss)} (см. "
                    "data/inventory/data_availability.csv)")
    return cat


def run_plan(prj: Project):
    cfg = prj.cfg
    catp = prj.inv / "granule_catalog.csv"
    if not catp.exists():
        run_inventory(prj, online=True)
    cat = pd.read_csv(catp)
    g = prj.grid()
    n_cells = g.width * g.height
    n_years = len(prj.years)
    size_gb = (cat["size_bytes"].fillna(0).sum() / 1e9
               if "size_bytes" in cat else float("nan"))
    by_prod = cat.groupby("product").agg(
        granules=("file", "count"),
        size_gb=("size_bytes", lambda s: s.fillna(0).sum() / 1e9)) \
        if len(cat) and "size_bytes" in cat else pd.DataFrame()
    n_layers_year = 60
    cache_gb = n_cells * 4 * n_layers_year * n_years / 1e9
    windows = (g.width * g.res / (cfg["fragmentation"]["main_window_km"] * 1e3)) * \
              (g.height * g.res / (cfg["fragmentation"]["main_window_km"] * 1e3))
    n_scen = sum({"threshold": len(cfg["treecover"]["sensitivity_thresholds_pct"]),
                  "window": len(cfg["fragmentation"]["sensitivity_windows_km"]),
                  "edge_depth": len(cfg["fragmentation"]["sensitivity_edge_depths"]),
                  "persistence": len(cfg["sensitivity"]["persistence_values"]),
                  "baseline": 1, "aggregation": 1, "lst_role": 1}.get(k, 0)
                 for k in cfg["sensitivity"]["run"])
    plan = dict(run_id=prj.run_id, grid_cells=n_cells, grid_width=g.width,
                grid_height=g.height, aoi_cells=int(g.aoi_mask.sum()),
                aoi_area_km2=round(g.aoi_mask.sum() * g.res ** 2 / 1e6, 1),
                years=n_years, granules=int(len(cat)),
                download_gb=round(size_gb, 2),
                est_work_cache_gb=round(cache_gb, 2),
                est_main_windows=int(math.ceil(windows)),
                sensitivity_scenarios=n_scen,
                per_product=by_prod.reset_index().to_dict("records")
                if len(by_prod) else [])
    (prj.inv / "plan.json").write_text(json.dumps(plan, indent=1,
                                                  ensure_ascii=False,
                                                  default=float),
                                       encoding="utf-8")
    LOG.info("План: " + json.dumps({k: v for k, v in plan.items()
                                    if k != "per_product"},
                                   ensure_ascii=False, default=float))
    if len(by_prod):
        LOG.info("\n" + by_prod.to_string())
    return plan


def earthdata_session():
    """Сессия HTTPS с авторизацией Earthdata (netrc / переменные среды /
    интерактивно). Учётные данные не журналируются."""
    try:
        import earthaccess
    except ImportError:
        raise RuntimeError("Установите earthaccess: pip install earthaccess")
    auth = None
    for strategy in ("environment", "netrc", "interactive"):
        try:
            auth = earthaccess.login(strategy=strategy, persist=False)
            if auth and getattr(auth, "authenticated", False):
                LOG.info(f"Earthdata: авторизация ({strategy})")
                break
        except Exception:
            continue
    if not auth or not getattr(auth, "authenticated", False):
        raise RuntimeError("Не удалось авторизоваться в NASA Earthdata. "
                           "Задайте EARTHDATA_USERNAME/EARTHDATA_PASSWORD или "
                           "~/.netrc (urs.earthdata.nasa.gov).")
    return earthaccess.get_requests_https_session()


def hdf_readable(path: Path) -> bool:
    try:
        from pyhdf.SD import SD, SDC
        sd = SD(str(path), SDC.READ)
        n = len(sd.datasets())
        sd.end()
        return n > 0
    except Exception:
        try:
            import rasterio
            with rasterio.open(str(path)) as src:
                return len(src.subdatasets) > 0
        except Exception:
            return False


def verify_checksum(path: Path, value, alg) -> Optional[bool]:
    if not value or not alg:
        return None
    alg = str(alg).lower().replace("-", "")
    if alg not in ("md5", "sha256", "sha1", "sha512"):
        return None
    h = hashlib.new(alg)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().lower() == str(value).lower()


def run_download(prj: Project):
    cfg = prj.cfg
    catp = prj.inv / "granule_catalog.csv"
    if not catp.exists():
        run_inventory(prj, online=True)
    cat = pd.read_csv(catp)
    regp = prj.inv / "granule_registry.csv"
    reg = pd.read_csv(regp) if regp.exists() else pd.DataFrame(
        columns=["file", "local_path", "sha256", "checksum_ok", "status"])
    done = set(reg.loc[reg.status == "ok", "file"]) if len(reg) else set()
    todo = cat[~cat["file"].isin(done)]
    LOG.info(f"К загрузке: {len(todo)} из {len(cat)} гранул")
    if todo.empty:
        return reg
    enable_dns_fallback()
    hosts = todo["url"].astype(str)
    sess_laads = laads_session(cfg) if hosts.str.contains("ladsweb").any() \
        else None
    sess_lp = earthdata_session() if (~hosts.str.contains("ladsweb")).any() \
        else None
    bad = prj.raw / "_corrupt"
    bad.mkdir(exist_ok=True)

    def fetch(r):
        dest = prj.raw / r["product"] / str(r["year"]) / r["file"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and hdf_readable(dest):
            ck = verify_checksum(dest, r.get("checksum"),
                                 r.get("checksum_algorithm"))
            if ck is not False:
                return dict(file=r["file"], local_path=str(dest),
                            sha256=sha256_file(dest), checksum_ok=ck,
                            status="ok")
        part = dest.with_suffix(".part")
        for attempt in range(cfg["data"]["retries"]):
            try:
                sess = sess_laads if "ladsweb" in str(r["url"]) else sess_lp
                with sess.get(r["url"], stream=True,
                              timeout=cfg["data"]["timeout_s"]) as resp:
                    if resp.status_code in (401, 403) or (
                            "ladsweb" in str(r["url"]) and
                            "text/html" in resp.headers.get("Content-Type", "")):
                        raise PermissionError(
                            f"HTTP {resp.status_code}: проверьте токен/доступ "
                            "Earthdata (для LAADS — действующий токен "
                            "в .earthdata_token или EARTHDATA_TOKEN)")
                    resp.raise_for_status()
                    with open(part, "wb") as f:
                        for chunk in resp.iter_content(1 << 20):
                            f.write(chunk)
                exp = r.get("size_bytes")
                if exp and not pd.isna(exp) and abs(part.stat().st_size -
                                                    float(exp)) > max(
                        2048, 0.002 * float(exp)):
                    raise IOError("размер файла не совпадает с каталогом")
                ck = verify_checksum(part, r.get("checksum"),
                                     r.get("checksum_algorithm"))
                if ck is False:
                    raise IOError("контрольная сумма не совпадает")
                os.replace(part, dest)
                if not hdf_readable(dest):
                    shutil.move(str(dest), str(bad / dest.name))
                    raise IOError("файл не читается как HDF4")
                return dict(file=r["file"], local_path=str(dest),
                            sha256=sha256_file(dest), checksum_ok=ck,
                            status="ok")
            except PermissionError as e:
                return dict(file=r["file"], local_path=None, sha256=None,
                            checksum_ok=None, status=f"error: {e}")
            except Exception as e:
                wait = min(60, 2 ** attempt)
                LOG.debug(f"{r['file']}: {e}; повтор через {wait} с")
                time.sleep(wait)
        if part.exists():
            shutil.move(str(part), str(bad / part.name))
        return dict(file=r["file"], local_path=None, sha256=None,
                    checksum_ok=None, status="error: попытки исчерпаны")

    results = []
    n = len(todo)
    with cf.ThreadPoolExecutor(cfg["data"]["download_workers"]) as ex:
        futs = [ex.submit(fetch, r) for r in todo.to_dict("records")]
        for i, f in enumerate(cf.as_completed(futs), 1):
            results.append(f.result())
            if i % 25 == 0 or i == n:
                LOG.info(f"Загружено {i}/{n}")
                pd.concat([reg, pd.DataFrame(results)]).drop_duplicates(
                    "file", keep="last").to_csv(regp, index=False)
    reg = pd.concat([reg, pd.DataFrame(results)]).drop_duplicates(
        "file", keep="last")
    reg.to_csv(regp, index=False)
    nerr = int((~reg.status.eq("ok")).sum())
    if nerr:
        LOG.warning(f"Ошибок загрузки: {nerr} (см. granule_registry.csv)")
    run_inventory(prj, online=False if cfg["data"]["mode"] == "offline"
                  else False)
    return reg


# =============================================================================
# 8. Чтение HDF4 и приведение к сетке
# =============================================================================
def read_hdf_layers(path: Path, names: Sequence[str]) -> Dict[str, np.ndarray]:
    out = {}
    try:
        from pyhdf.SD import SD, SDC
        sd = SD(str(path), SDC.READ)
        avail = set(sd.datasets().keys())
        for n in names:
            if n in avail:
                out[n] = sd.select(n).get()
        sd.end()
        return out
    except ImportError:
        pass
    import rasterio
    with rasterio.open(str(path)) as src:
        subs = src.subdatasets
    for n in names:
        cand = [s for s in subs if s.split(":")[-1] == n or
                s.endswith(f":{n}") or s.endswith(f'"{n}"')]
        if cand:
            with rasterio.open(cand[0]) as s:
                out[n] = s.read(1)
    return out


def hdf_attr_scale(path: Path, name: str) -> Optional[float]:
    try:
        from pyhdf.SD import SD, SDC
        sd = SD(str(path), SDC.READ)
        a = sd.select(name).attributes()
        sd.end()
        return float(a.get("scale_factor")) if "scale_factor" in a else None
    except Exception:
        return None


_SCALE_CHECKED = set()


def check_scale(prod, path, layer, spec):
    key = (prod, layer)
    if key in _SCALE_CHECKED or spec.get("kind") not in (
            "continuous", "accumulated"):
        return
    _SCALE_CHECKED.add(key)
    s = hdf_attr_scale(path, layer)
    if s is None:
        return
    eff = 1.0 / s if prod in DIVIDE_CONVENTION else s
    if not math.isclose(eff, spec["scale"], rel_tol=1e-6):
        LOG.warning(f"{prod}/{layer}: scale_factor в файле ({s}) не "
                    f"соответствует реестру ({spec['scale']}); используется "
                    "значение реестра — проверьте документацию.")


def physical(raw: np.ndarray, spec: dict) -> np.ndarray:
    a = raw.astype("float32")
    lo, hi = spec.get("valid", (-np.inf, np.inf))
    bad = (a < lo) | (a > hi)
    if spec.get("fill") is not None:
        bad |= raw == spec["fill"]
    a = a * spec.get("scale", 1.0) + spec.get("offset", 0.0)
    a[bad] = np.nan
    return a


def reproject_to_grid(arrays: List[Tuple[np.ndarray, Any]], grid: Grid,
                      method="average") -> np.ndarray:
    """Мозаика тайлов в общую сетку. arrays: [(float32 c NaN, transform)]."""
    from rasterio.warp import reproject, Resampling
    rs = {"average": Resampling.average, "nearest": Resampling.nearest,
          "mode": Resampling.mode}[method]
    out = np.full((grid.height, grid.width), np.nan, "float32")
    for arr, tr in arrays:
        dst = np.full_like(out, np.nan)
        reproject(source=arr, destination=dst, src_transform=tr,
                  src_crs=SINU_WKT, dst_transform=grid.affine,
                  dst_crs=grid.crs, src_nodata=np.nan, dst_nodata=np.nan,
                  resampling=rs)
        m = np.isfinite(dst) & ~np.isfinite(out)
        out[m] = dst[m]
    return out


class GranuleIndex:
    def __init__(self, prj: Project):
        self.prj = prj
        loc = local_granules(prj)
        if len(loc):
            loc = (loc.sort_values("production_time")
                      .drop_duplicates(["product", "tile", "year", "doy"],
                                       keep="last"))
        self.df = loc

    def files(self, prod, year, doy) -> List[Path]:
        if self.df.empty:
            return []
        s = self.df[(self.df["product"] == prod) & (self.df["year"] == year) &
                    (self.df["doy"] == doy)]
        return [Path(p) for p in s["local_path"]]

    def has(self, prod) -> bool:
        return (not self.df.empty) and (self.df["product"] == prod).any()


def load_composite(prj: Project, gi: GranuleIndex, prod: str, year: int,
                   doy: int, layers: Sequence[str], method="average",
                   extra_raw: Sequence[str] = ()) -> Optional[Dict[str, np.ndarray]]:
    """Читает гранулы всех тайлов, применяет QA и масштаб, переносит на сетку.
    Возвращает {layer: grid_array}. Для QA-масок — до пространственной
    агрегации."""
    spec = PRODUCTS[prod]
    files = gi.files(prod, year, doy)
    if not files:
        return None
    grid = prj.grid()
    profile = prj.cfg["quality"]["profile"]
    per_layer = {l: [] for l in list(layers) + list(extra_raw)}
    qa_names = [n for n, s in spec["layers"].items() if s["kind"] == "qa"]
    for f in files:
        g = parse_granule_name(f.name)
        need = list(layers) + qa_names + list(extra_raw)
        raw = read_hdf_layers(f, need)
        missing = [l for l in layers if l not in raw]
        if missing:
            LOG.warning(f"{f.name}: нет слоёв {missing}")
            continue
        try:
            q = spec["qa"](raw, profile)
        except KeyError as e:
            LOG.warning(f"{f.name}: нет QA-слоя {e}; QA не применён")
            q = None
        for l in layers:
            ls = spec["layers"][l]
            check_scale(prod, f, l, ls)
            if ls["kind"] in ("categorical", "doy", "event"):
                a = raw[l].astype("float32")
                if ls["kind"] == "categorical":
                    lo, hi = ls["valid"]
                    a[(a < lo) | (a > hi)] = np.nan
            else:
                a = physical(raw[l], ls)
            qm = q.get(l) if isinstance(q, dict) else q
            if qm is not None:
                a[~qm] = np.nan
            per_layer[l].append((a, tile_transform(g["tile"],
                                                   raw[l].shape[1])))
        for l in extra_raw:
            if l in raw:
                a = raw[l].astype("float32")
                per_layer[l].append((a, tile_transform(g["tile"],
                                                       raw[l].shape[1])))
    out = {}
    for l, lst in per_layer.items():
        if not lst:
            continue
        kind = spec["layers"].get(l, {}).get("kind", "qa")
        m = "nearest" if kind in ("categorical", "doy", "event", "qa") \
            else method
        out[l] = reproject_to_grid(lst, grid, m)
    return out


# =============================================================================
# 9. Состояние этапов (возобновление и инвалидация)
# =============================================================================
STAGE_CONFIG_KEYS = {
    "prepare": ["period", "seasons", "grid", "quality", "data", "inputs"],
    "landcover": ["landcover", "reporting", "statistics"],
    "treecover": ["treecover", "statistics"],
    "fragmentation": ["fragmentation", "treecover", "landcover", "quality"],
    "functional": ["functional", "lst", "statistics", "period"],
    "fire": ["fire", "treecover", "landcover"],
    "associations": ["associations", "statistics", "lst"],
    "vulnerability": ["vulnerability", "lst", "statistics"],
    "export": ["reporting"],
}


def stage_hash(prj: Project, stage: str) -> str:
    idx = STAGES.index(stage)
    parts = {k: prj.cfg.get(k) for k in STAGE_CONFIG_KEYS.get(stage, [])}
    parts["_version"] = __version__
    if idx > 0:
        prev = prj.stage_state_path(STAGES[idx - 1])
        parts["_upstream"] = json.loads(prev.read_text())["hash"] \
            if prev.exists() else None
    if stage == "prepare":
        reg = prj.inv / "granule_registry.csv"
        loc = local_granules(prj)
        parts["_inputs"] = config_hash(sorted(loc["file"].tolist())) \
            if len(loc) else None
        parts["_registry"] = sha256_file(reg) if reg.exists() else None
    return config_hash(parts)


def stage_done(prj: Project, stage: str) -> bool:
    p = prj.stage_state_path(stage)
    if not p.exists():
        return False
    st = json.loads(p.read_text())
    return st.get("status") == "done" and st.get("hash") == stage_hash(prj,
                                                                       stage)


def mark_stage(prj: Project, stage: str, status: str, seconds: float,
               info: Optional[dict] = None):
    p = prj.stage_state_path(stage)
    p.write_text(json.dumps(dict(stage=stage, status=status,
                                 hash=stage_hash(prj, stage),
                                 run_id=prj.run_id,
                                 finished=dt.datetime.now().isoformat(),
                                 seconds=round(seconds, 1),
                                 info=info or {}),
                            ensure_ascii=False, indent=1, default=str))


# =============================================================================
# 10. Статистика: тренды, MK, BH, аномалии
# =============================================================================
def bh_qvalues(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    m = np.isfinite(p)
    if not m.any():
        return q
    pv = p[m]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[m] = out
    return q


def _norm_sf(z):
    from scipy.special import erfc
    return 0.5 * erfc(z / np.sqrt(2))


def mk_sen_series(years: Sequence[float], x: Sequence[float],
                  alpha=0.05, autocorr=True) -> dict:
    """Mann–Kendall (+ модификация Hamed–Rao при значимой автокорреляции),
    наклон Сенна и его доверительный интервал (Gilbert, 1987)."""
    from scipy import stats
    t = np.asarray(years, float)
    v = np.asarray(x, float)
    m = np.isfinite(v)
    t, v = t[m], v[m]
    n = len(v)
    res = dict(n=n, slope=np.nan, intercept=np.nan, slope_lo=np.nan,
               slope_hi=np.nan, tau=np.nan, p_value=np.nan, z=np.nan,
               lag1_r=np.nan, variance_correction="none")
    if n < 4:
        res["status"] = "insufficient_years"
        return res
    i, j = np.triu_indices(n, 1)
    slopes = (v[j] - v[i]) / (t[j] - t[i])
    slope = np.median(slopes)
    s = np.sum(np.sign(v[j] - v[i]))
    _, cnt = np.unique(v, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) -
           np.sum(cnt * (cnt - 1) * (2 * cnt + 5))) / 18.0
    corr = 1.0
    if autocorr and n >= 10:
        resid = v - (slope * t + np.median(v - slope * t))
        r1 = np.corrcoef(resid[:-1], resid[1:])[0, 1] if np.std(resid) > 0 \
            else 0.0
        res["lag1_r"] = r1
        rk = stats.rankdata(resid)
        acf = []
        for k in range(1, n // 3):
            a = np.corrcoef(rk[:-k], rk[k:])[0, 1]
            if np.isfinite(a) and abs(a) > 1.96 / np.sqrt(n):
                acf.append((k, a))
        if acf:
            ssum = sum((n - k) * (n - k - 1) * (n - k - 2) * a
                       for k, a in acf)
            corr = 1 + 2 / (n * (n - 1) * (n - 2)) * ssum
            corr = max(corr, 1e-3)
            res["variance_correction"] = "hamed_rao"
    var *= corr
    z = 0.0 if s == 0 else (s - np.sign(s)) / np.sqrt(var)
    p = 2 * _norm_sf(abs(z))
    zc = stats.norm.ppf(1 - alpha / 2)
    c = zc * np.sqrt(var)
    ns = len(slopes)
    m1 = int(np.floor((ns - c) / 2.0))
    m2 = int(np.ceil((ns + c) / 2.0))
    ss = np.sort(slopes)
    lo = ss[max(m1 - 1, 0)] if 0 < m1 <= ns else np.nan
    hi = ss[min(m2, ns) - 1] if 0 < m2 <= ns else np.nan
    res.update(slope=slope, intercept=np.median(v - slope * t),
               slope_lo=lo, slope_hi=hi, tau=s / (0.5 * n * (n - 1)),
               p_value=p, z=z, status="ok")
    return res


def pixel_sen_mk(stack: np.ndarray, years: Sequence[int], min_n: int):
    """Векторный наклон Сенна и MK по пикселям. stack: (T, N) с NaN.
    Возвращает slope, p, n (без поправки на автокорреляцию — отмечается
    в метаданных)."""
    T, N = stack.shape
    t = np.asarray(years, float)
    i, j = np.triu_indices(T, 1)
    n_valid = np.isfinite(stack).sum(0)
    slope = np.full(N, np.nan, "float32")
    pval = np.full(N, np.nan, "float32")
    ok = n_valid >= max(min_n, 4)
    if not ok.any():
        return slope, pval, n_valid
    idx = np.where(ok)[0]
    chunk = max(1, int(2e7 // max(len(i), 1)))
    for s0 in range(0, len(idx), chunk):
        cols = idx[s0:s0 + chunk]
        x = stack[:, cols]
        d = x[j] - x[i]
        sl = d / (t[j] - t[i])[:, None]
        slope[cols] = np.nanmedian(sl, axis=0)
        sgn = np.nansum(np.sign(d), axis=0)
        n = n_valid[cols].astype(float)
        var = n * (n - 1) * (2 * n + 5) / 18.0
        z = np.where(sgn == 0, 0, (sgn - np.sign(sgn)) / np.sqrt(var))
        pval[cols] = 2 * _norm_sf(np.abs(z))
    return slope, pval, n_valid


def iter_row_blocks(height: int, rows: int):
    for r0 in range(0, height, rows):
        yield r0, min(height, r0 + rows)


def read_stack(paths: List[Optional[Path]], r0, r1, width) -> np.ndarray:
    from rasterio.windows import Window
    out = np.full((len(paths), r1 - r0, width), np.nan, "float32")
    for k, p in enumerate(paths):
        if p is not None and Path(p).exists():
            out[k] = read_raster(Path(p), Window(0, r0, width, r1 - r0))
    return out


# =============================================================================
# 11. Этап prepare: ежегодные слои
# =============================================================================
def annual_path(prj: Project, var: str, year: int) -> Path:
    return prj.work / "annual" / var / f"{var}_{year}.tif"


def _stats_block(stack, weights=None):
    with np.errstate(all="ignore"):
        return dict(mean=np.nanmean(stack, 0), median=np.nanmedian(stack, 0),
                    max=np.nanmax(stack, 0),
                    p10=np.nanpercentile(stack, 10, axis=0),
                    p90=np.nanpercentile(stack, 90, axis=0),
                    n=np.isfinite(stack).sum(0).astype("float32"))


def prepare_state_product(prj, gi, prod, var_prefix, layer, year, season_key,
                          use_pixel_doy=None, qrows: List[dict] = None):
    """Сезонные статистики переменных состояния (EVI, LAI, FPAR, LST)."""
    cfg = prj.cfg
    grid = prj.grid()
    season = cfg["seasons"][season_key]
    comps = needed_composites(cfg, prod, year)
    arrays = []
    n_expected = 0
    for st in comps:
        a, b = composite_interval(prod, year, st)
        ov, ln = season_overlap(year, a, b, season)
        if ov == 0:
            continue
        if use_pixel_doy is None and ov < ln / 2:
            continue
        n_expected += 1
        layers = [layer] + ([use_pixel_doy] if use_pixel_doy else [])
        d = load_composite(prj, gi, prod, year, st, layers)
        if d is None or layer not in d:
            continue
        arr = d[layer]
        if use_pixel_doy and use_pixel_doy in d:
            doy = d[use_pixel_doy]
            s0, s1 = md_to_doy(year, season[0]), md_to_doy(year, season[1])
            arr = np.where((doy >= s0) & (doy <= s1), arr, np.nan)
        arrays.append(arr)
    if not arrays:
        return False
    stack = np.stack(arrays)
    st = _stats_block(stack)
    cov = st["n"] / max(n_expected, 1)
    insufficient = cov < cfg["quality"]["min_season_coverage"]
    tags = dict(source_product=f"{prod}.061", layer=layer, season=season_key,
                season_range="..".join(season), qa_profile=cfg["quality"]["profile"],
                n_expected=n_expected, run_id=prj.run_id)
    for k in ("mean", "median", "max", "p10", "p90"):
        v = st[k].copy()
        v[insufficient] = np.nan
        write_raster(annual_path(prj, f"{var_prefix}_{season_key}_{k}", year),
                     v, grid, tags={**tags, "statistic": k})
    write_raster(annual_path(prj, f"{var_prefix}_{season_key}_coverage", year),
                 cov.astype("float32"), grid, tags=tags)
    if qrows is not None:
        am = grid.aoi_mask
        qrows.append(dict(variable=f"{var_prefix}_{season_key}", year=year,
                          product=prod, n_expected=n_expected,
                          n_loaded=len(arrays),
                          mean_coverage=float(np.nanmean(cov[am])),
                          insufficient_fraction=float(np.mean(insufficient[am])),
                          qa_profile=cfg["quality"]["profile"]))
    return True


def prepare_accumulated(prj, gi, prod, layers, year, season_key, qrows):
    """Накопленные величины (GPP, ET, PET): сумма по валидным интервалам с
    пропорциональным распределением пограничных интервалов; ET/PET как
    отношение сумм по общему набору валидных интервалов."""
    cfg = prj.cfg
    grid = prj.grid()
    season = cfg["seasons"][season_key]
    s_days = md_to_doy(year, season[1]) - md_to_doy(year, season[0]) + 1
    sums = {l: np.zeros((grid.height, grid.width), "float64") for l in layers}
    days = {l: np.zeros((grid.height, grid.width), "float64") for l in layers}
    common_et = np.zeros((grid.height, grid.width), "float64")
    common_pet = np.zeros((grid.height, grid.width), "float64")
    n_loaded = 0
    for st in needed_composites(cfg, prod, year):
        a, b = composite_interval(prod, year, st)
        ov, ln = season_overlap(year, a, b, season)
        if ov == 0:
            continue
        d = load_composite(prj, gi, prod, year, st, layers)
        if d is None:
            continue
        n_loaded += 1
        w = ov / ln
        for l in layers:
            if l in d:
                v = d[l]
                m = np.isfinite(v)
                sums[l][m] += v[m] * w
                days[l][m] += ov
        if "ET_500m" in d and "PET_500m" in d:
            m = np.isfinite(d["ET_500m"]) & np.isfinite(d["PET_500m"])
            common_et[m] += d["ET_500m"][m] * w
            common_pet[m] += d["PET_500m"][m] * w
    if n_loaded == 0:
        return False
    names = {"Gpp_500m": "gpp", "ET_500m": "et", "PET_500m": "pet"}
    for l in layers:
        cov = days[l] / s_days
        bad = cov < cfg["quality"]["min_season_coverage"]
        # сезонная сумма, приведённая к полной длине сезона (явное приближение)
        total = np.where(days[l] > 0, sums[l] / np.maximum(days[l], 1) * s_days,
                         np.nan)
        total[bad] = np.nan
        tags = dict(source_product=f"{prod}.061", layer=l, season=season_key,
                    statistic="season_sum_scaled_to_full_season",
                    units=PRODUCTS[prod]["layers"][l]["units"].replace(
                        "/interval", "/season"), run_id=prj.run_id)
        write_raster(annual_path(prj, f"{names[l]}_{season_key}_sum", year),
                     total.astype("float32"), grid, tags=tags)
        write_raster(annual_path(prj, f"{names[l]}_{season_key}_coverage",
                                 year), cov.astype("float32"), grid, tags=tags)
        am = grid.aoi_mask
        qrows.append(dict(variable=f"{names[l]}_{season_key}", year=year,
                          product=prod, n_expected=None, n_loaded=n_loaded,
                          mean_coverage=float(np.nanmean(cov[am])),
                          insufficient_fraction=float(np.mean(bad[am])),
                          qa_profile=cfg["quality"]["profile"]))
    if "ET_500m" in layers and "PET_500m" in layers:
        ratio = np.where(common_pet > 0, common_et / common_pet, np.nan)
        cov = days["ET_500m"] / s_days
        ratio[cov < cfg["quality"]["min_season_coverage"]] = np.nan
        write_raster(annual_path(prj, f"etpet_{season_key}_ratio", year),
                     ratio.astype("float32"), grid,
                     tags=dict(statistic="ratio_of_sums_common_intervals",
                               source_product="MOD16A2GF.061",
                               run_id=prj.run_id))
    return True


def stage_prepare(prj: Project):
    cfg = prj.cfg
    grid = prj.grid()
    gi = GranuleIndex(prj)
    prods = cfg["data"]["products"]
    qrows: List[dict] = []
    missing = []
    for y in prj.years:
        t0 = time.time()
        # --- MCD12Q1
        if "MCD12Q1" in prods:
            d = load_composite(prj, gi, "MCD12Q1", y, 1, ["LC_Type1"],
                               method="nearest")
            if d:
                lc = d["LC_Type1"]
                lc8 = np.where(np.isfinite(lc), lc, NODATA_U8).astype("uint8")
                write_raster(annual_path(prj, "lc", y), lc8, grid,
                             nodata=NODATA_U8, dtype="uint8",
                             tags=dict(source_product="MCD12Q1.061",
                                       layer="LC_Type1", legend="IGBP",
                                       qa_profile=cfg["quality"]["profile"],
                                       resampling="nearest",
                                       caution_after_2020=str(y > 2020)))
                qrows.append(dict(variable="lc", year=y, product="MCD12Q1",
                                  n_expected=1, n_loaded=1,
                                  mean_coverage=float(np.mean(
                                      lc8[grid.aoi_mask] != NODATA_U8)),
                                  insufficient_fraction=None,
                                  qa_profile=cfg["quality"]["profile"]))
            else:
                missing.append(("MCD12Q1", y))
        # --- MOD44B
        if "MOD44B" in prods:
            d = load_composite(prj, gi, "MOD44B", y, 65,
                               ["Percent_Tree_Cover"], method="average")
            if d:
                ptc = d["Percent_Tree_Cover"]
                write_raster(annual_path(prj, "ptc", y), ptc, grid,
                             tags=dict(source_product="MOD44B.061",
                                       layer="Percent_Tree_Cover", units="%",
                                       product_year=f"DOY 65 {y} – DOY 64 {y+1}",
                                       resampling="average_250_to_500"))
                qrows.append(dict(variable="ptc", year=y, product="MOD44B",
                                  n_expected=1, n_loaded=1,
                                  mean_coverage=float(np.mean(np.isfinite(
                                      ptc[grid.aoi_mask]))),
                                  insufficient_fraction=None,
                                  qa_profile=cfg["quality"]["profile"]))
            else:
                missing.append(("MOD44B", y))
        # --- вегетационные индексы
        for prod, pref in (("MOD13Q1", "evi_terra"), ("MYD13Q1", "evi_aqua")):
            if prod in prods:
                for sk in ("growing",):
                    if not prepare_state_product(
                            prj, gi, prod, pref, "250m 16 days EVI", y, sk,
                            use_pixel_doy="250m 16 days composite day of the year",
                            qrows=qrows):
                        missing.append((prod, y))
                    prepare_state_product(
                        prj, gi, prod, pref.replace("evi", "ndvi"),
                        "250m 16 days NDVI", y, sk,
                        use_pixel_doy="250m 16 days composite day of the year",
                        qrows=qrows)
        if "MCD15A2H" in prods:
            if not prepare_state_product(prj, gi, "MCD15A2H", "lai",
                                         "Lai_500m", y, "growing",
                                         qrows=qrows):
                missing.append(("MCD15A2H", y))
            prepare_state_product(prj, gi, "MCD15A2H", "fpar", "Fpar_500m", y,
                                  "growing", qrows=qrows)
        if "MOD17A2HGF" in prods:
            if not prepare_accumulated(prj, gi, "MOD17A2HGF", ["Gpp_500m"], y,
                                       "growing", qrows):
                missing.append(("MOD17A2HGF", y))
        if "MOD16A2GF" in prods:
            if not prepare_accumulated(prj, gi, "MOD16A2GF",
                                       ["ET_500m", "PET_500m"], y, "growing",
                                       qrows):
                missing.append(("MOD16A2GF", y))
        for prod, plat in (("MOD11A2", "terra"), ("MYD11A2", "aqua")):
            if prod in prods:
                ok1 = prepare_state_product(prj, gi, prod, f"lstday_{plat}",
                                            "LST_Day_1km", y, "summer",
                                            qrows=qrows)
                prepare_state_product(prj, gi, prod, f"lstnight_{plat}",
                                      "LST_Night_1km", y, "summer",
                                      qrows=qrows)
                if not ok1:
                    missing.append((prod, y))
        # --- MCD64A1
        if "MCD64A1" in prods:
            burned = np.zeros((grid.height, grid.width), "uint8")
            valid_any = np.zeros_like(burned, bool)
            first_doy = np.full((grid.height, grid.width), np.nan, "float32")
            n_months = 0
            for st in composite_starts("MCD64A1", y):
                d = load_composite(prj, gi, "MCD64A1", y, st, ["Burn Date"],
                                   method="nearest")
                if not d:
                    continue
                n_months += 1
                bd = d["Burn Date"]
                v = np.isfinite(bd) & (bd >= 0)
                valid_any |= v
                b = v & (bd > 0)
                newb = b & ~np.isfinite(first_doy)
                first_doy[newb] = bd[newb]
                burned[b] = 1
            if n_months:
                out = np.where(valid_any, burned, NODATA_U8).astype("uint8")
                write_raster(annual_path(prj, "burned", y), out, grid,
                             nodata=NODATA_U8, dtype="uint8",
                             tags=dict(source_product="MCD64A1.061",
                                       months_available=n_months,
                                       meaning="1=зарегистрировано выгорание; "
                                               "0=не зарегистрировано; "
                                               "255=нет данных"))
                write_raster(annual_path(prj, "burn_doy", y), first_doy, grid)
                qrows.append(dict(variable="burned", year=y, product="MCD64A1",
                                  n_expected=12, n_loaded=n_months,
                                  mean_coverage=float(np.mean(
                                      valid_any[grid.aoi_mask])),
                                  insufficient_fraction=None,
                                  qa_profile=cfg["quality"]["profile"]))
            else:
                missing.append(("MCD64A1", y))
        LOG.info(f"prepare {y}: {time.time()-t0:.0f} с")
    q = pd.DataFrame(qrows)
    save_table(prj, q, "quality_annual_layers")
    if missing:
        m = pd.DataFrame(missing, columns=["product", "year"]).drop_duplicates()
        save_table(prj, m, "missing_product_years")
        req = [p for p, _ in missing if PRODUCTS[p]["required"]]
        LOG.warning(f"Отсутствуют данные для {len(m)} продукт-лет")
        if req and cfg["data"]["fail_on_missing_required"]:
            raise RuntimeError(
                "Отсутствуют обязательные продукты: " +
                ", ".join(sorted(set(f"{p}:{y}" for p, y in missing
                                     if PRODUCTS[p]["required"]))) +
                ". Установите data.fail_on_missing_required: false для явно "
                "обозначенного неполного сценария.")
    return {"missing_product_years": len(missing)}


# =============================================================================
# 11b. GEE-бэкенд: те же годовые слои из Google Earth Engine, без гранул
# =============================================================================
# Каталог GEE: https://developers.google.com/earth-engine/datasets
# MCD15A2H в каталоге нет — используется MCD15A3H (4-дневный, Terra+Aqua,
# с 2002-07). MOD17A2HGF (8-дневный GPP) в GEE есть только с 2021 г., поэтому
# по умолчанию годовой GPP берётся из MOD17A3HGF (2001+, gee_gpp: annual).
GEE_ASSETS: Dict[str, Dict[str, Any]] = {
    "MCD12Q1": dict(asset="MODIS/061/MCD12Q1",
                    bands={"LC_Type1": "LC_Type1", "QC": "QC"}),
    "MOD44B": dict(asset="MODIS/061/MOD44B",
                   bands={"Percent_Tree_Cover": "Percent_Tree_Cover",
                          "Quality": "Quality"}),
    "MOD13Q1": dict(asset="MODIS/061/MOD13Q1",
                    bands={"250m 16 days EVI": "EVI",
                           "250m 16 days NDVI": "NDVI",
                           "250m 16 days pixel reliability": "SummaryQA",
                           "250m 16 days composite day of the year":
                               "DayOfYear"}),
    "MYD13Q1": dict(asset="MODIS/061/MYD13Q1",
                    bands={"250m 16 days EVI": "EVI",
                           "250m 16 days NDVI": "NDVI",
                           "250m 16 days pixel reliability": "SummaryQA",
                           "250m 16 days composite day of the year":
                               "DayOfYear"}),
    "MCD15A2H": dict(asset="MODIS/061/MCD15A3H", cadence=4,
                     bands={"Lai_500m": "Lai", "Fpar_500m": "Fpar",
                            "FparLai_QC": "FparLai_QC"}),
    "MOD17A2HGF": dict(asset="MODIS/061/MOD17A2HGF",
                       bands={"Gpp_500m": "Gpp",
                              "Psn_QC_500m": "Psn_QC"}),
    "MOD17A3HGF": dict(asset="MODIS/061/MOD17A3HGF", bands={"Gpp": "Gpp"}),
    "MOD16A2GF": dict(asset="MODIS/061/MOD16A2GF",
                      bands={"ET_500m": "ET", "PET_500m": "PET",
                             "ET_QC_500m": "ET_QC"}),
    "MOD11A2": dict(asset="MODIS/061/MOD11A2",
                    bands={"LST_Day_1km": "LST_Day_1km",
                           "LST_Night_1km": "LST_Night_1km",
                           "QC_Day": "QC_Day", "QC_Night": "QC_Night"}),
    "MYD11A2": dict(asset="MODIS/061/MYD11A2",
                    bands={"LST_Day_1km": "LST_Day_1km",
                           "LST_Night_1km": "LST_Night_1km",
                           "QC_Day": "QC_Day", "QC_Night": "QC_Night"}),
    "MCD64A1": dict(asset="MODIS/061/MCD64A1",
                    bands={"Burn Date": "BurnDate", "QA": "QA"}),
}
GEE_250M = {"MOD13Q1", "MYD13Q1", "MOD44B"}   # усреднение 250 → 500 м
GEE_SENTINEL = -9999.0                         # пропуск при скачивании


def gee_init(cfg):
    """Подключение к Earth Engine. Проект: data.gee_project, EE_PROJECT или
    настройки earthengine. При первом запуске откроется вход Google."""
    try:
        import ee
    except ImportError:
        raise RuntimeError("Для backend=gee установите earthengine-api: "
                           "pip install earthengine-api")
    project = (cfg["data"].get("gee_project")
               or os.environ.get("EE_PROJECT")
               or os.environ.get("GOOGLE_CLOUD_PROJECT") or None)
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
        return ee
    except Exception:
        pass
    try:
        ee.Authenticate()
        ee.Initialize(project=project) if project else ee.Initialize()
        return ee
    except Exception as e:
        raise RuntimeError(
            f"Не удалось подключиться к Earth Engine: {e}. Выполните в "
            "терминале `earthengine authenticate` и укажите проект в "
            "config.yaml (data.gee_project).")


def _gee_in(img, values):
    import ee
    m = ee.Image(0)
    for v in values:
        m = m.Or(img.eq(v))
    return m


def _gee_scf_cloud(qc, profile, sets):
    scf = qc.rightShift(5).bitwiseAnd(7)
    m = _gee_in(scf, sets[profile])
    if profile == "strict":
        m = m.And(_gee_in(qc.rightShift(3).bitwiseAnd(3), [0, 3]))
    return m


def gee_qa_mask(prod, img, profile, layer=None):
    """QA-маска (1 = пиксель принимается) — те же правила, что при загрузке
    гранул. None — QA для продукта не применяется."""
    if prod == "MCD12Q1":
        allowed = {"strict": [0], "standard": [0, 8, 9],
                   "relaxed": [0, 5, 7, 8, 9]}[profile]
        return _gee_in(img.select("QC"), allowed)
    if prod in ("MOD13Q1", "MYD13Q1"):
        allowed = {"strict": [0], "standard": [0, 1],
                   "relaxed": [0, 1, 2]}[profile]
        return _gee_in(img.select("SummaryQA"), allowed)
    if prod == "MCD15A2H":
        return _gee_scf_cloud(img.select("FparLai_QC"), profile,
                              {"strict": [0], "standard": [0, 1],
                               "relaxed": [0, 1, 2, 3]})
    if prod == "MOD17A2HGF":
        if profile == "relaxed":
            return None
        return _gee_scf_cloud(img.select("Psn_QC"), profile,
                              {"strict": [0, 1], "standard": [0, 1, 2, 3]})
    if prod == "MOD16A2GF":
        if profile == "relaxed":
            return None
        return _gee_scf_cloud(img.select("ET_QC"), profile,
                              {"strict": [0, 1], "standard": [0, 1, 2, 3]})
    if prod in ("MOD11A2", "MYD11A2"):
        qc = img.select("QC_Day" if layer == "LST_Day_1km" else "QC_Night")
        mand = qc.bitwiseAnd(3)
        if profile == "strict":
            return mand.eq(0)
        m = _gee_in(mand, [0, 1])
        if profile == "standard":
            m = m.And(qc.rightShift(6).bitwiseAnd(3).lte(1))
        return m
    if prod == "MCD64A1":
        qa = img.select("QA")
        m = qa.bitwiseAnd(1).eq(1).And(
            qa.rightShift(1).bitwiseAnd(1).eq(1))
        if profile == "strict":
            m = m.And(qa.rightShift(2).bitwiseAnd(1).eq(0))
        return m
    if prod == "MOD44B":
        if profile == "strict":
            return img.select("Quality").eq(0)
        return None
    return None


def gee_cadence(prod: str):
    return GEE_ASSETS[prod].get("cadence", PRODUCTS[prod]["cadence"])


def gee_composite_starts(prod: str, year: int) -> List[int]:
    cad = gee_cadence(prod)
    if isinstance(cad, str):          # annual | monthly — как в реестре
        return composite_starts(prod, year)
    return list(range(PRODUCTS[prod]["start_doy"], year_len(year) + 1, cad))


def gee_composite_interval(prod: str, year: int, st: int) -> Tuple[int, int]:
    cad = gee_cadence(prod)
    if isinstance(cad, str):
        return composite_interval(prod, year, st)
    return st, min(st + cad - 1, year_len(year))


def gee_needed_composites(cfg, prod: str, year: int) -> List[int]:
    cad = gee_cadence(prod)
    if isinstance(cad, str):
        return composite_starts(prod, year)
    s0 = min(md_to_doy(year, cfg["seasons"][k][0]) for k in cfg["seasons"])
    s1 = max(md_to_doy(year, cfg["seasons"][k][1]) for k in cfg["seasons"])
    out = []
    for st in gee_composite_starts(prod, year):
        a, b = gee_composite_interval(prod, year, st)
        if b >= s0 and a <= s1:
            out.append(st)
    return out


def _gee_empty(bands):
    import ee
    im = None
    for b in bands:
        c = ee.Image(0).mask(0).rename(b)
        im = c if im is None else im.addBands(c)
    return im


def gee_composite(prod, year, a, b):
    """Изображение GEE на интервал дней [a, b]; пустое (замаскированное),
    если данных за интервал нет."""
    import ee
    g = GEE_ASSETS[prod]
    d0 = (dt.date(year, 1, 1) + dt.timedelta(days=a - 1)).isoformat()
    d1 = (dt.date(year, 1, 1) + dt.timedelta(days=b)).isoformat()
    col = ee.ImageCollection(g["asset"]).filterDate(d0, d1)
    empty = _gee_empty(list(g["bands"].values()))
    return ee.Image(ee.Algorithms.If(col.size().gt(0), col.first(), empty))


def gee_layer(prod, img, layer, profile):
    """Слой в физических единицах с QA-маской (брак скрыт маской)."""
    spec = PRODUCTS[prod]["layers"][layer]
    raw = img.select(GEE_ASSETS[prod]["bands"][layer])
    lo, hi = spec.get("valid", (-1e9, 1e9))
    m = raw.gte(lo).And(raw.lte(hi))
    if spec.get("fill") is not None:
        m = m.And(raw.neq(spec["fill"]))
    qm = gee_qa_mask(prod, img, profile, layer=layer)
    if qm is not None:
        m = m.And(qm)
    if spec["kind"] in ("categorical", "doy", "event", "qa"):
        return raw.updateMask(m)
    return raw.multiply(spec.get("scale", 1.0)).add(
        spec.get("offset", 0.0)).updateMask(m)


def gee_to_grid(prod, img, layer):
    """Приведение к сетке 500 м: mean для 250 м продуктов, bilinear для 1 км
    и непрерывных 500 м (соответствует average при загрузке гранул)."""
    import ee
    if PRODUCTS[prod]["layers"][layer]["kind"] in ("categorical", "doy",
                                                   "event"):
        return img                       # nearest (по умолчанию)
    if prod in GEE_250M:
        return img.reduceResolution(ee.Reducer.mean(), maxPixels=1024)
    return img.resample("bilinear")


def gee_season_composites(cfg, prod, year, season_key, use_pixel_doy=None):
    """Композиты сезона по правилам отбора needed_composites и с теми же
    весами ov/ln, что при загрузке гранул. [(img, w, ov, ln), ...]"""
    season = cfg["seasons"][season_key]
    out = []
    for st in gee_needed_composites(cfg, prod, year):
        a, b = gee_composite_interval(prod, year, st)
        ov, ln = season_overlap(year, a, b, season)
        if ov == 0:
            continue
        if use_pixel_doy is None and ov < ln / 2:
            continue
        out.append((gee_composite(prod, year, a, b), ov / ln, ov, ln))
    return out


def gee_state_stats(cfg, prod, year, season_key, layer_specs,
                    use_pixel_doy=None):
    """Сезонные статистики (mean/median/max/p10/p90/coverage) слоёв состояния
    (EVI, NDVI, LAI, FPAR, LST). layer_specs: [(local_layer, out_prefix)].
    Возвращает (ee.Image, [(band, layer, stat)], n_expected) или None."""
    import ee
    comps = gee_season_composites(cfg, prod, year, season_key, use_pixel_doy)
    if not comps:
        return None
    season = cfg["seasons"][season_key]
    s0, s1 = md_to_doy(year, season[0]), md_to_doy(year, season[1])
    profile = cfg["quality"]["profile"]
    stacks: Dict[str, list] = {p: [] for _, p in layer_specs}
    for img, w, ov, ln in comps:
        pix = None
        if use_pixel_doy:
            doy = img.select(GEE_ASSETS[prod]["bands"][use_pixel_doy])
            pix = doy.gte(s0).And(doy.lte(s1))
        for layer, pref in layer_specs:
            v = gee_layer(prod, img, layer, profile)
            if pix is not None:
                v = v.updateMask(pix)
            stacks[pref].append(gee_to_grid(prod, v, layer))
    mincov = cfg["quality"]["min_season_coverage"]
    out, specs = None, []
    for layer, pref in layer_specs:
        col = ee.ImageCollection.fromImages(stacks[pref])
        cov = col.count().divide(float(len(comps)))
        ok = cov.gte(mincov)
        for k, im in (("mean", col.mean()), ("median", col.median()),
                      ("max", col.max()),
                      ("p10", col.reduce(ee.Reducer.percentile([10]))),
                      ("p90", col.reduce(ee.Reducer.percentile([90])))):
            im = im.updateMask(ok).rename(f"{pref}_{season_key}_{k}")
            out = im if out is None else out.addBands(im)
            specs.append((f"{pref}_{season_key}_{k}", layer, k))
        covb = cov.rename(f"{pref}_{season_key}_coverage")
        out = covb if out is None else out.addBands(covb)
        specs.append((f"{pref}_{season_key}_coverage", layer, "coverage"))
    return out, specs, len(comps)


def gee_accumulated(cfg, prod, year, season_key, layers):
    """Сезонные суммы накопленных величин (GPP/ET/PET) с весами ov/ln и
    ET/PET по общим валидным интервалам (как prepare_accumulated).
    layers: [(local_layer, out_prefix)]."""
    import ee
    comps = gee_season_composites(cfg, prod, year, season_key)
    if not comps:
        return None
    season = cfg["seasons"][season_key]
    s_days = md_to_doy(year, season[1]) - md_to_doy(year, season[0]) + 1
    profile = cfg["quality"]["profile"]
    layer_map = dict(layers)
    sums = {p: [] for p in layer_map.values()}
    days = {p: [] for p in layer_map.values()}
    cet, cpet = [], []
    for img, w, ov, ln in comps:
        vals = {}
        for layer in layer_map:
            v = gee_to_grid(prod, gee_layer(prod, img, layer, profile), layer)
            vals[layer] = v
            p = layer_map[layer]
            sums[p].append(v.multiply(w))
            days[p].append(v.mask().multiply(ov))
        if "ET_500m" in vals and "PET_500m" in vals:
            both = vals["ET_500m"].mask().And(vals["PET_500m"].mask())
            cet.append(vals["ET_500m"].updateMask(both).multiply(w))
            cpet.append(vals["PET_500m"].updateMask(both).multiply(w))
    mincov = cfg["quality"]["min_season_coverage"]
    out, specs = None, []

    def add(im, name, layer, stat):
        nonlocal out
        im = im.rename(name)
        out = im if out is None else out.addBands(im)
        specs.append((name, layer, stat))

    for layer, p in layers:
        s = ee.ImageCollection.fromImages(sums[p]).sum()
        d = ee.ImageCollection.fromImages(days[p]).sum()
        cov = d.divide(float(s_days))
        total = (s.divide(d.max(1)).multiply(float(s_days))
                 .updateMask(d.gt(0)).updateMask(cov.gte(mincov)))
        add(total, f"{p}_{season_key}_sum", layer,
            "season_sum_scaled_to_full_season")
        add(cov, f"{p}_{season_key}_coverage", layer, "coverage")
    if cet:
        cs = ee.ImageCollection.fromImages(cet).sum()
        ps = ee.ImageCollection.fromImages(cpet).sum()
        cov = ee.ImageCollection.fromImages(days["et"]).sum().divide(
            float(s_days))
        ratio = cs.divide(ps).updateMask(ps.gt(0)).updateMask(
            cov.gte(mincov))
        add(ratio, f"etpet_{season_key}_ratio", "ET_500m",
            "ratio_of_sums_common_intervals")
    return out, specs, len(comps)


def gee_lc(cfg, year):
    """LC_Type1 (IGBP) за год: класс 1–17 с QA (MCD12Q1)."""
    img = gee_composite("MCD12Q1", year, 1, year_len(year))
    v = gee_layer("MCD12Q1", img, "LC_Type1", cfg["quality"]["profile"])
    return v.rename("lc"), [("lc", "LC_Type1", "class")], 1


def gee_ptc(cfg, year):
    """Percent_Tree_Cover за год (снимок ~5 марта), среднее 250 → 500 м."""
    img = gee_composite("MOD44B", year, 60, 81)
    v = gee_layer("MOD44B", img, "Percent_Tree_Cover",
                  cfg["quality"]["profile"])
    v = gee_to_grid("MOD44B", v, "Percent_Tree_Cover")
    return v.rename("ptc"), [("ptc", "Percent_Tree_Cover", "percent")], 1


def gee_gpp_annual(cfg, year):
    """Годовая сумма GPP (MOD17A3HGF, с 2001 г.) — заполнение периода до
    2020, когда 8-дневного MOD17A2HGF в GEE нет."""
    img = gee_composite("MOD17A3HGF", year, 1, year_len(year))
    raw = img.select("Gpp")
    v = raw.multiply(1e-4).updateMask(raw.gte(0).And(raw.lte(32700)))
    return v.rename("gpp_annual_sum"),         [("gpp_annual_sum", "Gpp", "annual_sum")], 1


def gee_burned(cfg, year):
    """Выгорание за год: burned (0/1, 255=нет данных) и DOY первого пожара
    (MCD64A1, месячные композиты, QA как qa_mcd64)."""
    import ee
    profile = cfg["quality"]["profile"]
    bds, oks = [], []
    for st in composite_starts("MCD64A1", year):
        a, b = composite_interval("MCD64A1", year, st)
        img = gee_composite("MCD64A1", year, a, b)
        raw = img.select("BurnDate")
        qa = img.select("QA")
        ok = qa.bitwiseAnd(1).eq(1).And(
            qa.rightShift(1).bitwiseAnd(1).eq(1))
        if profile == "strict":
            ok = ok.And(qa.rightShift(2).bitwiseAnd(1).eq(0))
        m = raw.gte(0).And(ok)
        bds.append(raw.updateMask(m))
        oks.append(m.unmask(0))
    bdcol = ee.ImageCollection.fromImages(bds)
    okcol = ee.ImageCollection.fromImages(oks)
    burned = (bdcol.map(lambda i: i.gt(0).unmask(0)).max()
              .updateMask(okcol.max().gt(0)).rename("burned"))
    doy = bdcol.map(lambda i: i.updateMask(i.gt(0))).min().rename("burn_doy")
    out = burned.addBands(doy)
    return out, [("burned", "Burn Date", "binary"),
                 ("burn_doy", "Burn Date", "first_doy")], 12


def gee_grid_crs(prj: Project) -> str:
    from rasterio.crs import CRS
    return CRS.from_user_input(prj.grid().crs).to_wkt()


def gee_blocks(grid: Grid, nbands: int, max_mb: float = 40.0):
    """Разбиение сетки на блоки под лимит загрузки GEE (~50 МБ/запрос)."""
    rows = int(max_mb * 1024 * 1024 // (grid.width * nbands * 4))
    rows = max(1, min(rows, grid.height, 10000))
    out, r0 = [], 0
    while r0 < grid.height:
        r1 = min(grid.height, r0 + rows)
        out.append((r0, r1, 0, grid.width))
        r0 = r1
    return out


def gee_download(prj: Project, group: str, year: int, image, band_specs):
    """Скачивает multiband-изображение блоками (кэш + продолжение после
    сбоя) и пишет годовые растры через annual_path. band_specs:
    [(band, dtype, nodata, tags, role)]; role: primary|coverage|None.
    Возвращает {band: статистика по AOI} или {} (всё уже скачано)."""
    import ee
    import requests
    import io
    import zipfile
    import rasterio
    grid = prj.grid()
    am = grid.aoi_mask
    wkt = gee_grid_crs(prj)
    a = grid.affine
    res, x0, y0 = a.a, a.c, a.f
    names = [b[0] for b in band_specs]
    paths = {b[0]: annual_path(prj, b[0], year) for b in band_specs}
    if all(p.exists() for p in paths.values()):
        LOG.info(f"GEE {group} {year}: уже готово — пропуск")
        return {}
    cache = prj.work / "gee_cache" / f"{group}_{year}"
    cache.mkdir(parents=True, exist_ok=True)
    blocks = gee_blocks(grid, len(names))
    img = image.select(names).unmask(GEE_SENTINEL)
    LOG.info(f"GEE {group} {year}: {len(blocks)} блоков × {len(names)} полос")

    def fetch(r0, r1, c0, c1):
        bf = cache / f"b_{r0:05d}_{c0:05d}.npy"
        if bf.exists():
            return
        bx0, by0 = x0 + c0 * res, y0 - r0 * res
        bx1, by1 = x0 + c1 * res, y0 - r1 * res
        rect = ee.Geometry.Rectangle([bx0, by1, bx1, by0],
                                     proj=ee.Projection(wkt), geodesic=False)
        # Многополосный GeoTIFF: filePerBand=False обязателен — при True
        # Earth Engine игнорирует общие crs/crs_transform и возвращает
        # каждый слой в нативной проекции (это давало ошибку
        # "Pixel grid dimensions must be <= 32768").
        params = dict(bands=[{"id": n} for n in names], region=rect,
                      crs=wkt, crsTransform=[res, 0, 0, -res, bx0, by0],
                      format="GEO_TIFF", filePerBand=False)
        last = None
        for attempt in range(5):
            try:
                url = img.getDownloadURL(params)
                r = requests.get(url, timeout=(30, 600))
                r.raise_for_status()
                data = r.content
                if data[:2] == b"PK":
                    zf = zipfile.ZipFile(io.BytesIO(data))
                    inner = [n for n in zf.namelist()
                             if n.lower().endswith(".tif")]
                    data = zf.read(inner[0])
                with rasterio.open(io.BytesIO(data)) as src:
                    arr = src.read()
                if arr.shape != (len(names), r1 - r0, c1 - c0):
                    raise IOError(f"блок {arr.shape} != "
                                  f"{(len(names), r1 - r0, c1 - c0)}")
                np.save(bf, arr.astype("float32"))
                return
            except Exception as e:
                last = e
                LOG.debug(f"GEE блок {r0}: {e}; повтор")
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"GEE: блок строк {r0}-{r1} не скачан: {last}")

    nw = int(prj.cfg["data"].get("gee_workers", 4))
    with cf.ThreadPoolExecutor(nw) as ex:
        list(ex.map(lambda b: fetch(*b), blocks))
    stats = {}
    mincov = prj.cfg["quality"]["min_season_coverage"]
    for bi, (band, dtype, nodata, tags, role) in enumerate(band_specs):
        arr = np.full((grid.height, grid.width), np.nan, "float32")
        for (r0, r1, c0, c1) in blocks:
            blk = np.load(cache / f"b_{r0:05d}_{c0:05d}.npy")
            arr[r0:r1, c0:c1] = blk[bi]
        arr[arr == GEE_SENTINEL] = np.nan
        if dtype == "uint8":
            out = np.where(np.isfinite(arr), arr, NODATA_U8).astype("uint8")
        else:
            out = arr
        write_raster(paths[band], out, grid, nodata=nodata, dtype=dtype,
                     tags=tags)
        if am.any():
            v = arr[am]
            stats[band] = dict(
                frac_valid=float(np.isfinite(v).mean()),
                mean=float(np.nanmean(v)) if np.isfinite(v).any() else None,
                frac_lowcov=float((v < mincov).mean())
                if role == "coverage" else None)
    shutil.rmtree(cache, ignore_errors=True)
    return stats


def stage_prepare_gee(prj: Project):
    """prepare при backend=gee: те же годовые растры work/annual, источник —
    Google Earth Engine. Дальнейшие этапы не меняются."""
    cfg = prj.cfg
    gee_init(cfg)
    grid = prj.grid()
    prods = cfg["data"]["products"]
    profile = cfg["quality"]["profile"]
    qrows: List[dict] = []
    missing: List[tuple] = []
    gpp_mode = str(cfg["data"].get("gee_gpp", "annual")).lower()

    def specs_to_bands(prod, specs, season_key=None, extra=None):
        out = []
        for band, layer, stat in specs:
            tags = dict(source_product=f"{prod}.061", layer=layer,
                        statistic=stat, qa_profile=profile, backend="gee",
                        run_id=prj.run_id)
            if season_key:
                tags.update(season=season_key, season_range="..".join(
                    cfg["seasons"][season_key]))
            if extra:
                tags.update(extra)
            u8 = band in ("lc", "burned")
            role = ("primary" if stat in ("mean", "sum", "class", "percent",
                                          "binary", "annual_sum")
                    else "coverage" if stat == "coverage" else None)
            out.append((band, "uint8" if u8 else "float32",
                        NODATA_U8 if u8 else np.nan, tags, role))
        return out

    def emit(group, year, image, specs, n_expected, product, variables,
             season_key=None, extra=None):
        st = gee_download(prj, group, year, image,
                          specs_to_bands(product, specs, season_key, extra))
        if not st:
            return True
        prim = next((b[0] for b in specs if b[2] in
                     ("mean", "sum", "class", "percent", "binary",
                      "annual_sum")), specs[0][0])
        if not st.get(prim, {}).get("frac_valid"):
            return False
        for v in variables:
            cs = st.get(f"{v}_coverage")
            qrows.append(dict(
                variable=v, year=year, product=product,
                n_expected=n_expected, n_loaded=None,
                mean_coverage=(cs["mean"] if cs and cs.get("mean") is not None
                               else st[prim]["frac_valid"]),
                insufficient_fraction=cs["frac_lowcov"] if cs else None,
                qa_profile=profile))
        return True

    for y in prj.years:
        t0 = time.time()
        if "MCD12Q1" in prods:
            if not emit("lc", y, *gee_lc(cfg, y), "MCD12Q1", ["lc"]):
                missing.append(("MCD12Q1", y))
        if "MOD44B" in prods:
            if not emit("ptc", y, *gee_ptc(cfg, y), "MOD44B", ["ptc"]):
                missing.append(("MOD44B", y))
        for prod, pref in (("MOD13Q1", "evi_terra"), ("MYD13Q1", "evi_aqua")):
            if prod in prods:
                ls = [("250m 16 days EVI", pref),
                      ("250m 16 days NDVI", pref.replace("evi", "ndvi"))]
                r = gee_state_stats(cfg, prod, y, "growing", ls,
                                    use_pixel_doy="250m 16 days composite "
                                    "day of the year")
                if r is None or not emit(f"{prod}_vi", y, r[0], r[1], r[2],
                                         prod,
                                         [f"{p}_growing" for _, p in ls],
                                         season_key="growing"):
                    missing.append((prod, y))
        if "MCD15A2H" in prods:
            ls = [("Lai_500m", "lai"), ("Fpar_500m", "fpar")]
            r = gee_state_stats(cfg, "MCD15A2H", y, "growing", ls)
            if r is None or not emit("mcd15", y, r[0], r[1], r[2],
                                     "MCD15A2H",
                                     [f"{p}_growing" for _, p in ls],
                                     season_key="growing"):
                missing.append(("MCD15A2H", y))
        if "MOD17A2HGF" in prods:
            if gpp_mode == "annual":
                if not emit("gpp_annual", y, *gee_gpp_annual(cfg, y),
                            "MOD17A3HGF", ["gpp_annual"]):
                    missing.append(("MOD17A2HGF", y))
            else:
                r = gee_accumulated(cfg, "MOD17A2HGF", y, "growing",
                                    [("Gpp_500m", "gpp")])
                if r is None or not emit("gpp", y, r[0], r[1], r[2],
                                         "MOD17A2HGF", ["gpp_growing"],
                                         season_key="growing"):
                    missing.append(("MOD17A2HGF", y))
        if "MOD16A2GF" in prods:
            r = gee_accumulated(cfg, "MOD16A2GF", y, "growing",
                                [("ET_500m", "et"), ("PET_500m", "pet")])
            if r is None or not emit("mod16", y, r[0], r[1], r[2],
                                     "MOD16A2GF", ["et_growing",
                                                   "pet_growing"],
                                     season_key="growing"):
                missing.append(("MOD16A2GF", y))
        for prod, plat in (("MOD11A2", "terra"), ("MYD11A2", "aqua")):
            if prod in prods:
                ok_day = False
                for var, layer in (("lstday", "LST_Day_1km"),
                                   ("lstnight", "LST_Night_1km")):
                    r = gee_state_stats(cfg, prod, y, "summer",
                                        [(layer, f"{var}_{plat}")])
                    if r and emit(f"{prod}_{var}", y, r[0], r[1], r[2],
                                  prod, [f"{var}_{plat}_summer"],
                                  season_key="summer") and var == "lstday":
                        ok_day = True
                if not ok_day:
                    missing.append((prod, y))
        if "MCD64A1" in prods:
            if not emit("burned", y, *gee_burned(cfg, y), "MCD64A1",
                        ["burned"]):
                missing.append(("MCD64A1", y))
        LOG.info(f"prepare(GEE) {y}: {time.time() - t0:.0f} с")
    q = pd.DataFrame(qrows)
    save_table(prj, q, "quality_annual_layers")
    if missing:
        m = pd.DataFrame(missing, columns=["product", "year"]).drop_duplicates()
        save_table(prj, m, "missing_product_years")
        LOG.warning(f"Отсутствуют данные для {len(m)} продукт-лет")
        req = [p for p, _ in missing if PRODUCTS[p]["required"]]
        if req and cfg["data"]["fail_on_missing_required"]:
            raise RuntimeError(
                "Отсутствуют обязательные продукты: " +
                ", ".join(sorted(set(f"{p}:{y}" for p, y in missing
                                     if PRODUCTS[p]["required"]))) +
                ". Установите data.fail_on_missing_required: false для "
                "явно обозначенного неполного сценария.")
    return {"missing_product_years": len(missing), "backend": "gee"}


def cmd_gee_info(prj: Project):
    """Проверка доступа к Earth Engine и покрытия продуктов."""
    cfg = prj.cfg
    ee = gee_init(cfg)
    y0, y1 = prj.years[0], prj.years[-1]
    LOG.info("Earth Engine: подключение установлено")
    for prod in cfg["data"]["products"]:
        g = GEE_ASSETS.get(prod)
        if not g:
            LOG.warning(f"{prod}: нет в каталоге GEE")
            continue
        col = (ee.ImageCollection(g["asset"])
               .filterDate(f"{y0}-01-01", f"{y1 + 1}-01-01"))
        n = col.size().getInfo()
        rng = col.reduceColumns(ee.Reducer.minMax(),
                                ["system:time_start"]).getInfo()
        d = lambda t: dt.datetime.fromtimestamp(
            t / 1000, dt.timezone.utc).date()
        LOG.info(f"{prod} → {g['asset']}: {n} изображений, "
                 f"{d(rng['min'])} … {d(rng['max'])}"
                 + (" (замена: MCD15A3H)" if prod == "MCD15A2H" else "")
                 + (" (только 2021+; годовой GPP — MOD17A3HGF)"
                    if prod == "MOD17A2HGF" else ""))
    return 0


# =============================================================================
# 12. Табличный вывод
# =============================================================================
COMMON_FIELDS = ["run_id", "result_id"]


def save_table(prj: Project, df: pd.DataFrame, name: str, subdir: str = ""):
    d = prj.tables / subdir if subdir else prj.tables
    d.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    if "run_id" not in df.columns:
        df.insert(0, "run_id", prj.run_id)
    if "result_id" not in df.columns:
        df.insert(1, "result_id", [f"{name}:{i}" for i in range(len(df))])
    df.to_csv(d / f"{name}.csv", index=False, encoding="utf-8")
    try:
        df.to_parquet(d / f"{name}.parquet", index=False)
    except Exception as e:
        LOG.debug(f"Parquet не записан ({name}): {e}")
    return d / f"{name}.csv"


def load_table(prj: Project, name: str, subdir="") -> Optional[pd.DataFrame]:
    d = prj.tables / subdir if subdir else prj.tables
    p = d / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def trend_table(df: pd.DataFrame, group_cols: List[str], value_col: str,
                year_col="year", alpha=0.05, min_n=15, autocorr=True,
                family=None) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        g = g.sort_values(year_col)
        r = mk_sen_series(g[year_col].values, g[value_col].values, alpha,
                          autocorr)
        r.update(dict(zip(group_cols, key)))
        r["metric"] = value_col
        r["first_year"] = g.loc[g[value_col].notna(), year_col].min()
        r["last_year"] = g.loc[g[value_col].notna(), year_col].max()
        if r["n"] < min_n and r.get("status") == "ok":
            r["status"] = "insufficient_years"
        rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        out["q_value"] = bh_qvalues(out["p_value"].values)
        out["significant_q"] = out["q_value"] < alpha
        out["test_family"] = family or value_col
    return out


# =============================================================================
# 13. Этап landcover
# =============================================================================
def stage_landcover(prj: Project):
    cfg = prj.cfg
    grid = prj.grid()
    yrs = [y for y in prj.years if annual_path(prj, "lc", y).exists()]
    if not yrs:
        LOG.warning("landcover: нет слоёв MCD12Q1 — этап пропущен")
        return {"skipped": True}
    lcc = cfg["landcover"]
    classes = lcc["classes"]
    am = grid.aoi_mask
    zones = grid.zones
    rows = []
    stack = np.stack([read_raster(annual_path(prj, "lc", y)) for y in yrs])
    T = len(yrs)
    # --- ежегодные площади (исходные и сглаженные модой трёх лет)
    smooth = stack.copy()
    ambig = np.zeros_like(stack, bool)
    for k in range(1, T - 1):
        a, b, c = stack[k - 1], stack[k], stack[k + 1]
        va = (a != NODATA_U8) & (c != NODATA_U8)
        eq = va & (a == c)
        smooth[k] = np.where(eq, a, b)
        ambig[k] = va & (a != b) & (b != c) & (a != c)
    units = [("aoi", "AOI", am)]
    if zones is not None:
        for zid, zn in grid.zone_names.items():
            units.append(("zone", zn, am & (zones == zid)))
    for k, y in enumerate(yrs):
        for utype, uid, um in units:
            ua = um.sum() * grid.cell_ha
            for version, arr in (("raw", stack[k]), ("mode3", smooth[k])):
                v = arr[um]
                valid = v != NODATA_U8
                valid_ha = valid.sum() * grid.cell_ha
                woody = np.isin(v, classes).sum() * grid.cell_ha
                cnt = np.bincount(v[valid].astype(int), minlength=256)
                for c in sorted(set(range(1, 18))):
                    area = cnt[c] * grid.cell_ha
                    rows.append(dict(year=y, unit_type=utype, unit_id=uid,
                                     version=version, cover_class=c,
                                     class_name=IGBP_NAMES_RU.get(c),
                                     selected=c in classes, area_ha=area,
                                     share_of_unit_pct=100 * area / ua if ua else np.nan,
                                     share_of_woody_pct=100 * area / woody
                                     if (woody and c in classes) else np.nan,
                                     valid_area_ha=valid_ha, unit_area_ha=ua,
                                     ambiguous_mode_ha=float(ambig[k][um].sum()
                                                             * grid.cell_ha)
                                     if version == "mode3" else np.nan,
                                     caution_post_2020=y > 2020))
    areas = pd.DataFrame(rows)
    areas = areas.sort_values(["unit_type", "unit_id", "version",
                               "cover_class", "year"])
    g = areas.groupby(["unit_type", "unit_id", "version", "cover_class"])
    areas["change_prev_year_ha"] = g["area_ha"].diff()
    b0, b1 = cfg["period"]["comparison_baseline"]
    base = areas[(areas.year >= b0) & (areas.year <= b1)].groupby(
        ["unit_type", "unit_id", "version", "cover_class"])["area_ha"].mean() \
        .rename("baseline_area_ha")
    areas = areas.merge(base, on=["unit_type", "unit_id", "version",
                                  "cover_class"], how="left")
    areas["change_vs_baseline_ha"] = areas["area_ha"] - areas["baseline_area_ha"]
    areas["units"] = "ha"
    save_table(prj, areas, "annual_class_areas")
    # --- пятилетние сводки
    prow = []
    for p0, p1 in cfg["reporting"]["periods"]:
        sub = areas[(areas.year >= p0) & (areas.year <= p1)]
        if sub.empty:
            continue
        agg = sub.groupby(["unit_type", "unit_id", "version", "cover_class",
                           "class_name", "selected"]).agg(
            mean_annual_area_ha=("area_ha", "mean"),
            n_years=("year", "nunique")).reset_index()
        agg["period_start"], agg["period_end"] = p0, p1
        agg["period_length_years"] = p1 - p0 + 1
        agg["incomplete_period"] = agg["n_years"] < agg["period_length_years"]
        prow.append(agg)
    if prow:
        save_table(prj, pd.concat(prow), "period_class_areas")
    ref = [y for y in cfg["reporting"]["reference_years"] if y in yrs]
    if ref:
        save_table(prj, areas[areas.year.isin(ref)], "reference_year_class_areas")
    # --- тренды площадей
    st = cfg["statistics"]
    tr = trend_table(areas[(areas.version == "raw") & areas.selected],
                     ["unit_type", "unit_id", "cover_class"], "area_ha",
                     alpha=st["alpha"], min_n=st["min_valid_years"],
                     autocorr=st["autocorrelation_check"],
                     family="class_area_trends")
    save_table(prj, tr, "trends_class_area")
    # --- исходные матрицы переключения меток (год к году)
    sw = []
    for k in range(1, T):
        a, b = stack[k - 1][am], stack[k][am]
        v = (a != NODATA_U8) & (b != NODATA_U8)
        pair = a[v].astype(int) * 256 + b[v].astype(int)
        u, c = np.unique(pair, return_counts=True)
        for pu, cu in zip(u, c):
            fa, tb = pu // 256, pu % 256
            if fa in classes or tb in classes:
                sw.append(dict(year_from=yrs[k - 1], year_to=yrs[k],
                               from_class=fa, to_class=tb,
                               area_ha=cu * grid.cell_ha))
    save_table(prj, pd.DataFrame(sw), "label_switches_annual")
    # --- кандидатные устойчивые переходы
    ev, mat = detect_transitions(prj, stack, yrs, lcc["persistence_before"],
                                 lcc["persistence_after"], classes)
    save_table(prj, ev, "class_transitions")
    save_table(prj, mat, "transition_matrix_persistent")
    # карта: год подтверждённого перехода
    tmap = np.full((grid.height, grid.width), np.nan, "float32")
    if len(ev):
        c = ev[ev.status == "confirmed"]
        tmap[c["row"].values, c["col"].values] = c["first_new_class_year"].values
    write_raster(prj.rasters / "landcover" / "transition_year_confirmed.tif",
                 tmap, grid, tags=dict(meaning="год первого появления нового "
                                               "класса для подтверждённых "
                                               "кандидатных переходов"))
    for y in ref:
        k = yrs.index(y)
        write_raster(prj.rasters / "landcover" / f"lc_igbp_{y}.tif", stack[k],
                     grid, nodata=NODATA_U8, dtype="uint8")
        write_raster(prj.rasters / "landcover" / f"lc_igbp_mode3_{y}.tif",
                     smooth[k], grid, nodata=NODATA_U8, dtype="uint8")
    return {"years": len(yrs), "transition_events": int(len(ev))}


def detect_transitions(prj, stack, yrs, pb, pa, classes,
                       ptc_stack=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Кандидатные устойчивые переходы A→B: A ≥ pb валидных лет подряд до,
    B ≥ pa лет подряд после; пропуски не засчитываются; цензурирование на
    краях ряда фиксируется."""
    grid = prj.grid()
    cfg = prj.cfg
    T = len(yrs)
    am = grid.aoi_mask
    rr, cc = np.where(am)
    S = stack[:, rr, cc]
    if ptc_stack is None:
        ptc_paths = [annual_path(prj, "ptc", y) for y in yrs]
        if all(p.exists() for p in ptc_paths):
            P = np.stack([read_raster(p)[rr, cc] for p in ptc_paths])
        else:
            P = None
    else:
        P = ptc_stack[:, rr, cc]
    delta = cfg["landcover"]["ptc_consistency_delta_pp"]
    events = []
    for t in range(1, T):
        A, B = S[t - 1], S[t]
        cand = (A != NODATA_U8) & (B != NODATA_U8) & (A != B) & \
               (np.isin(A, classes) | np.isin(B, classes))
        if not cand.any():
            continue
        idx = np.where(cand)[0]
        a, b = A[idx], B[idx]
        pre = np.zeros(len(idx), int)
        k = t - 1
        alive = np.ones(len(idx), bool)
        while k >= 0 and alive.any():
            alive &= S[k, idx] == a
            pre += alive
            k -= 1
        post = np.zeros(len(idx), int)
        k = t
        alive = np.ones(len(idx), bool)
        while k < T and alive.any():
            alive &= S[k, idx] == b
            post += alive
            k += 1
        # цензурирование: ряд закончился/начался до набора длины
        left_c = (pre < pb) & (pre == t)            # вся история до t = A
        right_c = (post < pa) & (post == T - t)     # всё после t = B
        status = np.where((pre >= pb) & (post >= pa), "confirmed",
                          np.where(right_c & (pre >= pb),
                                   "unconfirmed_end_of_series",
                                   np.where(left_c & (post >= pa),
                                            "unconfirmed_start_of_series",
                                            "rejected_unstable")))
        keep = status != "rejected_unstable"
        if not keep.any():
            continue
        idx, a, b, pre, post, status = (idx[keep], a[keep], b[keep], pre[keep],
                                        post[keep], status[keep])
        lc_, rc_ = left_c[keep], right_c[keep]
        cons = np.full(len(idx), "insufficient_data", object)
        if P is not None:
            pre_m = np.full(len(idx), np.nan)
            post_m = np.full(len(idx), np.nan)
            for i_, (ix, pr, po) in enumerate(zip(idx, pre, post)):
                s0 = max(0, t - pr)
                pre_m[i_] = np.nanmean(P[s0:t, ix]) if t > s0 else np.nan
                post_m[i_] = np.nanmean(P[t:t + po, ix]) if po else np.nan
            ra = np.array([WOODINESS_RANK.get(int(x), 0) for x in a])
            rb = np.array([WOODINESS_RANK.get(int(x), 0) for x in b])
            dif = post_m - pre_m
            fin = np.isfinite(dif)
            cons = np.where(~fin, "insufficient_data",
                            np.where(ra == rb, "not_informative",
                                     np.where(((rb < ra) & (dif <= -delta)) |
                                              ((rb > ra) & (dif >= delta)),
                                              "consistent",
                                              np.where(((rb < ra) & (dif >= delta)) |
                                                       ((rb > ra) & (dif <= -delta)),
                                                       "contradictory",
                                                       "not_informative"))))
        x0, _, _, y0 = grid.transform[2], 0, 0, grid.transform[5]
        r_, c_ = rr[idx], cc[idx]
        events.append(pd.DataFrame(dict(
            pixel_id=r_ * grid.width + c_, row=r_, col=c_,
            x=x0 + (c_ + 0.5) * grid.res, y=y0 - (r_ + 0.5) * grid.res,
            from_class=a.astype(int), to_class=b.astype(int),
            first_new_class_year=yrs[t],
            confirmation_year=[yrs[min(t + pa - 1, T - 1)] if s == "confirmed"
                               else np.nan for s in status],
            pre_run_length=pre, post_run_length=post, qa_status="passed",
            ptc_consistency=cons, validation_status="not_validated",
            left_censored=lc_, right_censored=rc_, status=status)))
    ev = pd.concat(events, ignore_index=True) if events else pd.DataFrame(
        columns=["pixel_id", "row", "col", "from_class", "to_class",
                 "first_new_class_year", "status"])
    mx = cfg["landcover"]["max_transition_events_export"]
    if len(ev) > mx:
        LOG.warning(f"Событий переходов {len(ev)} > {mx}: в таблицу событий "
                    "попадут подтверждённые, матрицы считаются по всем.")
    mat = ev.groupby(["status", "from_class", "to_class",
                      "ptc_consistency"]).size().rename("n_pixels").reset_index() \
        if len(ev) else pd.DataFrame(columns=["status", "from_class",
                                              "to_class", "n_pixels"])
    if len(mat):
        mat["area_ha"] = mat["n_pixels"] * grid.cell_ha
        mat["persistence_before"], mat["persistence_after"] = pb, pa
    if len(ev) > mx:
        ev = ev[ev.status == "confirmed"].head(mx)
    return ev, mat


# =============================================================================
# 14. Этап treecover
# =============================================================================
def event_detection(stack: np.ndarray, years: List[int], thr: np.ndarray,
                    pre_years: int, persist: int, tol: np.ndarray):
    """Первое устойчивое снижение и восстановление по пикселям.
    stack (T,N); thr, tol — (N,). Возвращает словарь массивов (N,)."""
    T, N = stack.shape
    ev_year = np.full(N, np.nan, "float32")
    depth = np.full(N, np.nan, "float32")
    pre_lvl = np.full(N, np.nan, "float32")
    rec_time = np.full(N, np.nan, "float32")
    status = np.zeros(N, "uint8")  # 0 нет, 1 recovered, 2 not_recovered(цензур), 3 insufficient_followup
    n_events = np.zeros(N, "uint16")
    found = np.zeros(N, bool)
    for t in range(pre_years, T - persist + 1):
        pre = np.nanmedian(stack[t - pre_years:t], 0)
        cond = np.isfinite(pre)
        for k in range(persist):
            cond &= (stack[t + k] - pre) <= -thr
        new = cond & ~found
        n_events += cond & ((t == pre_years) | ~(
            (stack[t - 1] - pre) <= -thr))
        if new.any():
            ev_year[new] = years[t]
            pre_lvl[new] = pre[new]
            depth[new] = (pre - np.nanmin(stack[t:t + persist], 0))[new]
            found |= new
            # восстановление: два года подряд ≥ pre - tol
            rec = np.full(N, np.nan, "float32")
            for s in range(t + persist, T - 1):
                ok = new & np.isnan(rec) & (stack[s] >= pre - tol) & \
                     (stack[s + 1] >= pre - tol)
                rec[ok] = years[s] - years[t]
            rec_time[new] = rec[new]
            follow = T - (t + persist)
            st = np.where(np.isfinite(rec), 1, np.where(follow >= 2, 2, 3))
            status[new] = st[new]
    return dict(event_year=ev_year, depth=depth, pre_level=pre_lvl,
                recovery_years=rec_time, recovery_status=status,
                n_events=n_events.astype("float32"))


REC_STATUS = {0: "no_event", 1: "recovered", 2: "not_recovered_censored",
              3: "insufficient_followup"}


def stage_treecover(prj: Project):
    cfg = prj.cfg
    grid = prj.grid()
    yrs = [y for y in prj.years if annual_path(prj, "ptc", y).exists()]
    if not yrs:
        LOG.warning("treecover: нет MOD44B — этап пропущен")
        return {"skipped": True}
    tc = cfg["treecover"]
    st = cfg["statistics"]
    b0, b1 = cfg["period"]["comparison_baseline"]
    names = ["mean", "median", "change_vs_baseline_pp", "relative_change_pct",
             "sen_slope_pp_per_year", "mk_p", "n_valid", "min", "min_year",
             "max_annual_drop_pp", "n_negative_anomaly_years",
             "decline_year", "decline_depth_pp", "recovery_years",
             "recovery_status", "n_decline_events", "fire_coincident"]
    out = {n: np.full((grid.height, grid.width), np.nan, "float32")
           for n in names}
    paths = [annual_path(prj, "ptc", y) for y in yrs]
    bpaths = [annual_path(prj, "burned", y) for y in yrs]
    has_fire = all(p.exists() for p in bpaths)
    for r0, r1 in iter_row_blocks(grid.height, cfg["execution"]["block_rows"]):
        S = read_stack(paths, r0, r1, grid.width)
        T, h, w = S.shape
        X = S.reshape(T, -1)
        msk = grid.aoi_mask[r0:r1].ravel()
        X[:, ~msk] = np.nan
        with np.errstate(all="ignore"):
            mean = np.nanmean(X, 0)
            med = np.nanmedian(X, 0)
            bi = [i for i, y in enumerate(yrs) if b0 <= y <= b1]
            base = np.nanmean(X[bi], 0) if bi else np.full(X.shape[1], np.nan)
            last = np.nanmean(X[-3:], 0)
            chg = last - base
            rel = np.where(base >= 10, 100 * chg / base, np.nan)
            sl, p, nv = pixel_sen_mk(X, yrs, st["min_valid_years"])
            allnan = np.all(~np.isfinite(X), 0)
            mn = np.nanmin(X, 0)
            amin = np.where(allnan, -1, np.nanargmin(np.where(np.isfinite(X), X,
                                                              np.inf), 0))
            minyr = np.where(amin >= 0, np.array(yrs)[np.clip(amin, 0, None)],
                             np.nan)
            d = np.diff(X, axis=0)
            maxdrop = -np.nanmin(d, 0) if T > 1 else np.full(X.shape[1], np.nan)
            mad = 1.4826 * np.nanmedian(np.abs(d - np.nanmedian(d, 0)), 0) \
                if T > 2 else np.zeros(X.shape[1])
            resid = X - (sl * (np.array(yrs)[:, None] - yrs[0]) +
                         np.nanmedian(X - sl * (np.array(yrs)[:, None] - yrs[0]),
                                      0))
            sdr = np.nanstd(resid, 0)
            nneg = np.nansum(resid < -sdr, 0).astype("float32")
        thr = np.maximum(tc["decline_min_pp"], tc["decline_k_mad"] *
                         np.nan_to_num(mad))
        tol = np.maximum(tc["recovery_tolerance_pp"], np.nan_to_num(mad))
        ev = event_detection(X, yrs, thr, tc["pre_event_years"],
                             tc["decline_persistence_years"], tol)
        firec = np.full(X.shape[1], np.nan, "float32")
        if has_fire:
            B = read_stack(bpaths, r0, r1, grid.width).reshape(T, -1)
            B = np.where(B == NODATA_U8, np.nan, B)
            has_ev = np.isfinite(ev["event_year"])
            firec[has_ev] = 0
            for k, y in enumerate(yrs):
                sel = has_ev & (ev["event_year"] == y)
                if sel.any():
                    win = B[max(0, k - 1):k + 1][:, sel]
                    firec[sel] = (np.nansum(win, 0) > 0).astype("float32")
        vals = dict(mean=mean, median=med, change_vs_baseline_pp=chg,
                    relative_change_pct=rel, sen_slope_pp_per_year=sl, mk_p=p,
                    n_valid=nv.astype("float32"), min=mn, min_year=minyr,
                    max_annual_drop_pp=maxdrop, n_negative_anomaly_years=nneg,
                    decline_year=ev["event_year"],
                    decline_depth_pp=ev["depth"],
                    recovery_years=ev["recovery_years"],
                    recovery_status=np.where(msk, ev["recovery_status"],
                                             np.nan),
                    n_decline_events=ev["n_events"], fire_coincident=firec)
        for n in names:
            v = np.asarray(vals[n], "float32").reshape(h, w)
            v[~grid.aoi_mask[r0:r1]] = np.nan
            out[n][r0:r1] = v
    for n in names:
        write_raster(prj.rasters / "treecover" / f"ptc_{n}.tif", out[n], grid,
                     tags=dict(source="MOD44B.061 Percent_Tree_Cover",
                               years=f"{yrs[0]}-{yrs[-1]}",
                               comparison_baseline=f"{b0}-{b1}",
                               trend_note="MK без поправки на автокорреляцию "
                                          "на уровне пикселя",
                               run_id=prj.run_id))
    # события в таблицу
    am = grid.aoi_mask
    m = am & np.isfinite(out["decline_year"])
    r_, c_ = np.where(m)
    evdf = pd.DataFrame(dict(
        pixel_id=r_ * grid.width + c_, row=r_, col=c_,
        decline_year=out["decline_year"][m].astype(int),
        depth_pp=out["decline_depth_pp"][m],
        recovery_years=out["recovery_years"][m],
        recovery_status=[REC_STATUS.get(int(s), "?") for s in
                         out["recovery_status"][m]],
        origin=np.where(out["fire_coincident"][m] == 1, "fire_coincident",
                        np.where(np.isfinite(out["fire_coincident"][m]),
                                 "unknown_origin", "fire_data_unavailable")),
        right_censored=out["recovery_status"][m] == 2))
    save_table(prj, evdf, "treecover_events")
    # региональные ряды
    rows = []
    thrs = sorted(set(tc["sensitivity_thresholds_pct"] +
                      [tc["primary_threshold_pct"]]))
    units = [("aoi", "AOI", am)]
    if grid.zones is not None:
        for zid, zn in grid.zone_names.items():
            units.append(("zone", zn, am & (grid.zones == zid)))
    for y, pth in zip(yrs, paths):
        a = read_raster(pth)
        for utype, uid, um in units:
            v = a[um]
            fin = np.isfinite(v)
            row = dict(year=y, unit_type=utype, unit_id=uid,
                       mean_ptc_pct=float(np.nanmean(v)) if fin.any() else np.nan,
                       median_ptc_pct=float(np.nanmedian(v)) if fin.any() else np.nan,
                       tree_cover_equivalent_area_ha=float(np.nansum(v) / 100 *
                                                           grid.cell_ha),
                       valid_fraction=float(fin.mean()),
                       unit_area_ha=float(um.sum() * grid.cell_ha))
            for t_ in thrs:
                row[f"mask_area_ge{t_}_ha"] = float((v >= t_).sum() *
                                                    grid.cell_ha)
            rows.append(row)
    ts = pd.DataFrame(rows)
    save_table(prj, ts, "treecover_timeseries")
    tr = []
    for col in ["mean_ptc_pct", "tree_cover_equivalent_area_ha"] + \
               [f"mask_area_ge{t_}_ha" for t_ in thrs]:
        tr.append(trend_table(ts, ["unit_type", "unit_id"], col,
                              alpha=st["alpha"], min_n=st["min_valid_years"],
                              autocorr=st["autocorrelation_check"],
                              family="treecover_trends"))
    trd = pd.concat(tr, ignore_index=True)
    trd["q_value"] = bh_qvalues(trd["p_value"].values)
    save_table(prj, trd, "trends_treecover")
    return {"years": len(yrs), "decline_events": int(len(evdf))}


# =============================================================================
# 15. Фрагментация
# =============================================================================
_S8 = np.ones((3, 3), bool)
_S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


def frag_metrics(m: np.ndarray, res: float, edge_depth=1, neighborhood=8,
                 count_outer=False, enn=True) -> dict:
    """Метрики класса в окне. m: int8 (1 — класс, 0 — фон, -1 — NoData).
    NoData не считается фоном: не создаёт границ и не уменьшает CORE."""
    from scipy import ndimage
    cell_ha = res * res / 1e4
    valid = m >= 0
    Z = int(valid.sum())
    cls = m == 1
    out = dict(valid_cells=Z, CA=np.nan, PLAND=np.nan, NP=np.nan, PD=np.nan,
               AREA_MN=np.nan, AREA_MD=np.nan, LPI=np.nan, ED=np.nan,
               TCA=np.nan, CPLAND=np.nan, CORE_SHARE=np.nan, COHESION=np.nan,
               MESH=np.nan, ENN_MN=np.nan)
    if Z == 0:
        return out
    A_ha = Z * cell_ha
    lab, NP = ndimage.label(cls, structure=_S8 if neighborhood == 8 else _S4)
    areas = np.bincount(lab.ravel())[1:].astype(float)
    CA = cls.sum() * cell_ha
    out.update(CA=CA, PLAND=100 * CA / A_ha, NP=NP, PD=NP / A_ha * 1e4)
    # границы класс/фон (горизонтальные и вертикальные стороны ячеек)
    bg = m == 0
    e = (np.sum(cls[:, 1:] & bg[:, :-1]) + np.sum(cls[:, :-1] & bg[:, 1:]) +
         np.sum(cls[1:, :] & bg[:-1, :]) + np.sum(cls[:-1, :] & bg[1:, :]))
    if count_outer:
        p = np.pad(cls, 1, constant_values=False)
        pv = np.pad(valid, 1, constant_values=False)
        outer = (np.sum(p[1:-1, 1:-1] & ~pv[1:-1, :-2]) +
                 np.sum(p[1:-1, 1:-1] & ~pv[1:-1, 2:]) +
                 np.sum(p[1:-1, 1:-1] & ~pv[:-2, 1:-1]) +
                 np.sum(p[1:-1, 1:-1] & ~pv[2:, 1:-1]))
        e += outer
    out["ED"] = e * res / A_ha
    # CORE: удаление edge_depth слоёв ячеек, соседствующих с фоном
    if edge_depth > 0:
        bgd = ndimage.binary_dilation(bg, structure=_S8,
                                      iterations=int(edge_depth))
        core = cls & ~bgd
    else:
        core = cls
    TCA = core.sum() * cell_ha
    out.update(TCA=TCA, CPLAND=100 * TCA / A_ha,
               CORE_SHARE=100 * TCA / CA if CA > 0 else np.nan)
    if NP == 0:
        out.update(AREA_MN=np.nan, AREA_MD=np.nan, LPI=0.0, MESH=0.0)
        return out
    a_ha = areas * cell_ha
    out.update(AREA_MN=a_ha.mean(), AREA_MD=float(np.median(a_ha)),
               LPI=100 * a_ha.max() / A_ha, MESH=float((a_ha ** 2).sum() / A_ha))
    # COHESION (FRAGSTATS): периметр в сторонах ячеек, площадь в ячейках
    p = np.pad(cls, 1, constant_values=False)
    same = (p[:-2, 1:-1].astype(int) + p[2:, 1:-1] + p[1:-1, :-2] +
            p[1:-1, 2:])
    perim_cell = np.where(cls, 4 - same, 0)
    per = np.bincount(lab.ravel(), weights=perim_cell.ravel())[1:]
    if Z > 1 and (per * np.sqrt(areas)).sum() > 0:
        coh = (1 - per.sum() / (per * np.sqrt(areas)).sum()) / \
              (1 - 1 / np.sqrt(Z)) * 100
        out["COHESION"] = float(coh)
    if enn and NP >= 2:
        out["ENN_MN"] = enn_mean(lab, NP, res)
    return out


def enn_mean(lab: np.ndarray, NP: int, res: float) -> float:
    """Среднее расстояние (центр–центр ячеек) до ближайшего однотипного пятна."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    cls = lab > 0
    er = ndimage.binary_erosion(cls, structure=_S4, border_value=0)
    edge = cls & ~er
    r, c = np.where(edge)
    L = lab[r, c]
    pts = np.column_stack([r, c]).astype(float) * res
    tree = cKDTree(pts)
    best = np.full(NP + 1, np.inf)
    k = min(len(pts), 16)
    while True:
        d, i = tree.query(pts, k=k)
        if k == 1:
            d, i = d[:, None], i[:, None]
        other = L[i] != L[:, None]
        dd = np.where(other, d, np.inf).min(1)
        np.minimum.at(best, L, dd)
        unresolved = [p for p in range(1, NP + 1) if not np.isfinite(best[p])]
        if not unresolved or k >= len(pts):
            break
        k = min(len(pts), k * 4)
    v = best[1:]
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else np.nan


def make_masks(prj: Project, year: int, ptc_threshold: Optional[float] = None,
               masks: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """Бинарные маски (int8: 1 класс, 0 фон, -1 NoData)."""
    cfg = prj.cfg
    grid = prj.grid()
    masks = masks or cfg["fragmentation"]["masks"]
    thr = ptc_threshold if ptc_threshold is not None else \
        cfg["treecover"]["primary_threshold_pct"]
    out = {}
    pp = annual_path(prj, "ptc", year)
    lp = annual_path(prj, "lc", year)
    outside = ~grid.aoi_mask
    if "ptc" in masks and pp.exists():
        a = read_raster(pp)
        m = np.where(np.isfinite(a), (a >= thr).astype("int8"), -1).astype("int8")
        m[outside] = -1
        out[f"ptc{int(thr)}"] = m
    if lp.exists():
        lc = read_raster(lp)
        nd = lc == NODATA_U8
        lcc = cfg["landcover"]
        groups = {"closed": lcc["closed_classes"],
                  "sparse": lcc["sparse_classes"], "woody": lcc["classes"]}
        for gname, gcls in groups.items():
            if gname in masks:
                m = np.isin(lc, gcls).astype("int8")
                m[nd | outside] = -1
                out[gname] = m
        if cfg["fragmentation"].get("per_class"):
            for c in lcc["classes"]:
                m = (lc == c).astype("int8")
                m[nd | outside] = -1
                out[f"lc{c}"] = m
    return out


def window_grid(grid: Grid, size_km: float):
    n = int(round(size_km * 1000 / grid.res))
    rows = []
    for i in range(0, grid.height, n):
        for j in range(0, grid.width, n):
            rows.append((f"W{int(size_km)}km_r{i//n:03d}_c{j//n:03d}", i,
                         min(i + n, grid.height), j, min(j + n, grid.width)))
    return rows, n


def windows_gdf(grid: Grid, size_km: float):
    import geopandas as gpd
    from shapely.geometry import box
    wins, n = window_grid(grid, size_km)
    x0, y0 = grid.transform[2], grid.transform[5]
    recs = []
    for wid, r0, r1, c0, c1 in wins:
        aoi_frac = grid.aoi_mask[r0:r1, c0:c1].sum() / float(n * n)
        if aoi_frac <= 0:
            continue
        recs.append(dict(window_id=wid, size_km=size_km, row0=r0, col0=c0,
                         aoi_fraction=aoi_frac, complete=aoi_frac >= 0.999,
                         geometry=box(x0 + c0 * grid.res, y0 - r1 * grid.res,
                                      x0 + c1 * grid.res, y0 - r0 * grid.res)))
    return gpd.GeoDataFrame(recs, crs=grid.crs)


def compute_fragmentation(prj: Project, years: List[int], size_km: float,
                          edge_depth: int, ptc_threshold: float,
                          masks: Optional[List[str]] = None,
                          enn: Optional[bool] = None) -> pd.DataFrame:
    cfg = prj.cfg
    fc = cfg["fragmentation"]
    grid = prj.grid()
    wins, n = window_grid(grid, size_km)
    enn = fc["enn"] if enn is None else enn
    rows = []
    minvf = cfg["quality"]["min_window_valid_fraction"]
    for y in years:
        ms = make_masks(prj, y, ptc_threshold, masks)
        for mname, m in ms.items():
            # ландшафт целиком (AOI)
            r = frag_metrics(m, grid.res, edge_depth, fc["neighborhood"],
                             fc["count_outer_boundary_in_ed"], enn=False)
            r.update(year=y, mask=mname, unit_type="aoi", unit_id="AOI",
                     window_km=np.nan, edge_depth_cells=edge_depth,
                     aoi_fraction=1.0, valid_fraction=r["valid_cells"] /
                     max(1, int(grid.aoi_mask.sum())))
            rows.append(r)
            for wid, r0, r1, c0, c1 in wins:
                sub = m[r0:r1, c0:c1]
                aoi_cells = grid.aoi_mask[r0:r1, c0:c1].sum()
                if aoi_cells == 0:
                    continue
                r = frag_metrics(sub, grid.res, edge_depth, fc["neighborhood"],
                                 fc["count_outer_boundary_in_ed"], enn)
                vf = r["valid_cells"] / float(aoi_cells)
                r.update(year=y, mask=mname, unit_type="window", unit_id=wid,
                         window_km=size_km, edge_depth_cells=edge_depth,
                         aoi_fraction=aoi_cells / float(n * n),
                         valid_fraction=vf)
                rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df["status"] = np.where(df["valid_fraction"] < minvf,
                                "insufficient_valid_data",
                                np.where(df["aoi_fraction"] < 0.999,
                                         "edge_window", "ok"))
        df["ptc_threshold_pct"] = ptc_threshold
    return df


FRAG_UNITS = {"CA": "ha", "PLAND": "%", "NP": "count", "PD": "per 100 km2",
              "AREA_MN": "ha", "AREA_MD": "ha", "LPI": "%", "ED": "m/ha",
              "TCA": "ha", "CPLAND": "%", "CORE_SHARE": "%",
              "COHESION": "0-100", "MESH": "ha", "ENN_MN": "m"}


def stage_fragmentation(prj: Project):
    cfg = prj.cfg
    fc = cfg["fragmentation"]
    yrs = [y for y in prj.years if annual_path(prj, "ptc", y).exists() or
           annual_path(prj, "lc", y).exists()]
    if not yrs:
        return {"skipped": True}
    grid = prj.grid()
    df = compute_fragmentation(prj, yrs, fc["main_window_km"],
                               fc["edge_depth_cells"],
                               cfg["treecover"]["primary_threshold_pct"])
    save_table(prj, df, "fragmentation_annual")
    wg = windows_gdf(grid, fc["main_window_km"])
    wg.to_file(prj.vectors / f"windows_{int(fc['main_window_km'])}km.gpkg",
               driver="GPKG")
    st = cfg["statistics"]
    w = df[(df.unit_type == "window") & (df.status != "insufficient_valid_data")]
    trs = []
    for met in ["PLAND", "ED", "PD", "CPLAND", "CORE_SHARE", "COHESION",
                "MESH", "LPI", "AREA_MN", "ENN_MN"]:
        trs.append(trend_table(w, ["mask", "unit_id"], met, alpha=st["alpha"],
                               min_n=st["min_valid_years"],
                               autocorr=False, family=f"frag_window_{met}"))
        trs.append(trend_table(df[df.unit_type == "aoi"], ["mask", "unit_id"],
                               met, alpha=st["alpha"],
                               min_n=st["min_valid_years"],
                               autocorr=st["autocorrelation_check"],
                               family=f"frag_aoi_{met}"))
    trd = pd.concat(trs, ignore_index=True)
    save_table(prj, trd, "trends_fragmentation")
    # пятилетние сводки
    pr = []
    for p0, p1 in cfg["reporting"]["periods"]:
        s = df[(df.year >= p0) & (df.year <= p1) & (df.unit_type == "aoi")]
        if len(s):
            a = s.groupby("mask")[list(FRAG_UNITS)].mean().reset_index()
            a["period_start"], a["period_end"] = p0, p1
            a["n_years"] = s.year.nunique()
            pr.append(a)
    if pr:
        save_table(prj, pd.concat(pr), "period_fragmentation_aoi")
    return {"rows": int(len(df))}


# =============================================================================
# 16. Функциональное состояние и LST
# =============================================================================
FUNC_VARS = {
    "evi_terra": "evi_terra_growing_mean", "evi_aqua": "evi_aqua_growing_mean",
    "ndvi_terra": "ndvi_terra_growing_mean", "lai": "lai_growing_mean",
    "fpar": "fpar_growing_mean", "gpp": "gpp_growing_sum",
    "etpet": "etpet_growing_ratio",
}
LST_VARS = {
    "lstday_terra": "lstday_terra_summer_mean",
    "lstnight_terra": "lstnight_terra_summer_mean",
    "lstday_aqua": "lstday_aqua_summer_mean",
    "lstnight_aqua": "lstnight_aqua_summer_mean",
}


def functional_pixel_stats(prj: Project, var: str, layer: str,
                           anomaly_baseline=None, out_dir="functional",
                           high_is_adverse=False):
    cfg = prj.cfg
    fcfg = cfg["functional"]
    st = cfg["statistics"]
    grid = prj.grid()
    yrs = [y for y in prj.years if annual_path(prj, layer, y).exists()]
    if len(yrs) < 3:
        return None
    a0, a1 = anomaly_baseline or cfg["period"]["anomaly_baseline"]
    bi = [i for i, y in enumerate(yrs) if a0 <= y <= a1]
    paths = [annual_path(prj, layer, y) for y in yrs]
    names = ["mean", "sen_slope", "mk_p", "n_valid", "adverse_years_own_p10",
             "adverse_years_z", "first_decline_year", "decline_depth_z",
             "recovery_years", "recovery_status", "longterm_change"]
    out = {n: np.full((grid.height, grid.width), np.nan, "float32")
           for n in names}
    zdir = prj.work / "anomaly" / var
    zdir.mkdir(parents=True, exist_ok=True)
    zfull = {y: np.full((grid.height, grid.width), np.nan, "float32")
             for y in yrs}
    sign = 1.0 if high_is_adverse else -1.0
    for r0, r1 in iter_row_blocks(grid.height, cfg["execution"]["block_rows"]):
        S = read_stack(paths, r0, r1, grid.width)
        T, h, w = S.shape
        X = S.reshape(T, -1)
        msk = grid.aoi_mask[r0:r1].ravel()
        X[:, ~msk] = np.nan
        with np.errstate(all="ignore"):
            mu = np.nanmean(X[bi], 0) if bi else np.nanmean(X, 0)
            sd = np.nanstd(X[bi], 0, ddof=1) if len(bi) > 2 else np.nanstd(X, 0, ddof=1)
            nb = np.isfinite(X[bi]).sum(0) if bi else np.isfinite(X).sum(0)
            sd = np.where((sd > 0) & (nb >= 3), sd, np.nan)
            Z = (X - mu) / sd
            p10 = np.nanpercentile(X, 100 - fcfg["adverse_percentile"]
                                   if high_is_adverse else
                                   fcfg["adverse_percentile"], axis=0)
            adv_own = np.nansum((X > p10) if high_is_adverse else (X < p10),
                                0).astype("float32")
            adv_z = np.nansum(sign * Z >= abs(fcfg["decline_z"]),
                              0).astype("float32")
            sl, p, nv = pixel_sen_mk(X, yrs, st["min_valid_years"])
            med = np.nanmedian(X[bi], 0) if bi else np.nanmedian(X, 0)
            first3 = np.nanmean(X[:3], 0)
            last3 = np.nanmean(X[-3:], 0)
        # функциональный спад: z ≤ decline_z (для LST: ≥ |z|) устойчиво
        Zs = sign * -Z  # теперь неблагоприятное = отрицательное
        ev = event_detection(Zs, yrs, np.full(X.shape[1], abs(fcfg["decline_z"])),
                             0 if False else 1, fcfg["sustained_years"],
                             np.full(X.shape[1], fcfg["recovery_sd"]))
        for k, y in enumerate(yrs):
            zfull[y][r0:r1] = Z[k].reshape(h, w)
        vals = dict(mean=np.nanmean(X, 0), sen_slope=sl, mk_p=p,
                    n_valid=nv.astype("float32"),
                    adverse_years_own_p10=adv_own, adverse_years_z=adv_z,
                    first_decline_year=ev["event_year"],
                    decline_depth_z=ev["depth"],
                    recovery_years=ev["recovery_years"],
                    recovery_status=np.where(msk, ev["recovery_status"], np.nan),
                    longterm_change=last3 - first3)
        for n in names:
            v = np.asarray(vals[n], "float32").reshape(h, w)
            v[~grid.aoi_mask[r0:r1]] = np.nan
            out[n][r0:r1] = v
    od = prj.rasters / out_dir
    for n in names:
        write_raster(od / f"{var}_{n}.tif", out[n], grid,
                     tags=dict(variable=var, source_layer=layer,
                               anomaly_baseline=f"{a0}-{a1}",
                               years=f"{yrs[0]}-{yrs[-1]}",
                               note="статистическая характеристика; не прямой "
                                    "диагноз состояния насаждения",
                               run_id=prj.run_id))
    for y in yrs:
        write_raster(zdir / f"z_{y}.tif", zfull[y], grid)
    return out, yrs


def func_layer(prj: Project, var: str):
    """Слой функциональной переменной. Для GPP — запасной вариант
    gpp_annual_sum, если сезонного GPP нет (GEE-бэкенд до 2020 г.)."""
    layer = FUNC_VARS.get(var) or LST_VARS.get(var)
    if var == "gpp" and layer and not any(
            annual_path(prj, layer, y).exists() for y in prj.years):
        return "gpp_annual_sum"
    return layer


def stage_functional(prj: Project):
    cfg = prj.cfg
    grid = prj.grid()
    am = grid.aoi_mask
    done = []
    regional = []
    st = cfg["statistics"]
    for var in cfg["functional"]["variables"]:
        layer = func_layer(prj, var)
        if not layer:
            continue
        r = functional_pixel_stats(prj, var, layer)
        if r is None:
            LOG.warning(f"functional: нет данных {var}")
            continue
        done.append(var)
    for var, layer in LST_VARS.items():
        r = functional_pixel_stats(prj, var, layer, out_dir="lst",
                                   high_is_adverse=True)
        if r is not None:
            done.append(var)
    # суточная амплитуда LST
    for plat in ("terra", "aqua"):
        for y in prj.years:
            dp = annual_path(prj, f"lstday_{plat}_summer_mean", y)
            npth = annual_path(prj, f"lstnight_{plat}_summer_mean", y)
            if dp.exists() and npth.exists():
                write_raster(annual_path(prj, f"lstdtr_{plat}_summer_mean", y),
                             read_raster(dp) - read_raster(npth), grid,
                             tags=dict(effective_resolution_m=1000))
    # региональные ряды по типам покрова
    for var in done:
        layer = func_layer(prj, var)
        for y in prj.years:
            p = annual_path(prj, layer, y)
            if not p.exists():
                continue
            a = read_raster(p)
            lp = annual_path(prj, "lc", y)
            lc = read_raster(lp) if lp.exists() else None
            groups = [("all_aoi", am)]
            if lc is not None:
                for c in cfg["landcover"]["classes"]:
                    groups.append((f"lc{c}", am & (lc == c)))
            for gname, gm in groups:
                v = a[gm]
                fin = np.isfinite(v)
                regional.append(dict(variable=var, year=y, stratum=gname,
                                     mean=float(np.nanmean(v)) if fin.any() else np.nan,
                                     median=float(np.nanmedian(v)) if fin.any() else np.nan,
                                     n_valid=int(fin.sum()),
                                     coverage=float(fin.mean()) if len(v) else np.nan,
                                     effective_resolution_m=1000 if
                                     var.startswith("lst") else 500))
    reg = pd.DataFrame(regional)
    save_table(prj, reg, "functional_annual")
    if len(reg):
        tr = trend_table(reg, ["variable", "stratum"], "mean",
                         alpha=st["alpha"], min_n=st["min_valid_years"],
                         autocorr=st["autocorrelation_check"],
                         family="functional_regional_trends")
        save_table(prj, tr, "trends_functional")
    # таблица функциональных событий (агрегировано по классам базового года)
    evrows = []
    b0 = cfg["period"]["comparison_baseline"][0]
    lcb = annual_path(prj, "lc", b0)
    lcbase = read_raster(lcb) if lcb.exists() else None
    for var in done:
        od = "lst" if var.startswith("lst") else "functional"
        ey = read_raster(prj.rasters / od / f"{var}_first_decline_year.tif")
        rs = read_raster(prj.rasters / od / f"{var}_recovery_status.tif")
        rt = read_raster(prj.rasters / od / f"{var}_recovery_years.tif")
        strata = [("all_aoi", am)]
        if lcbase is not None:
            strata += [(f"lc{c}", am & (lcbase == c))
                       for c in cfg["landcover"]["classes"]]
        for sname, sm in strata:
            m = sm & np.isfinite(ey)
            n_ev = int(m.sum())
            evrows.append(dict(variable=var, stratum=sname,
                               n_pixels=int(sm.sum()), n_event_pixels=n_ev,
                               event_fraction=n_ev / max(1, int(sm.sum())),
                               recovered=int((rs[m] == 1).sum()),
                               not_recovered_censored=int((rs[m] == 2).sum()),
                               insufficient_followup=int((rs[m] == 3).sum()),
                               median_recovery_years=float(np.nanmedian(rt[m]))
                               if np.isfinite(rt[m]).any() else np.nan))
    save_table(prj, pd.DataFrame(evrows), "functional_events")
    return {"variables": done}


# =============================================================================
# 17. Пожары
# =============================================================================
def stage_fire(prj: Project):
    cfg = prj.cfg
    grid = prj.grid()
    am = grid.aoi_mask
    yrs = [y for y in prj.years if annual_path(prj, "burned", y).exists()]
    if not yrs:
        LOG.warning("fire: нет MCD64A1 — этап пропущен")
        return {"skipped": True}
    B = np.stack([read_raster(annual_path(prj, "burned", y)) for y in yrs]
                 ).astype("float32")
    B[B == NODATA_U8] = np.nan
    nvalid = np.isfinite(B).sum(0)
    nburn = np.nansum(B, 0)
    freq = np.where(nvalid > 0, nburn / nvalid, np.nan).astype("float32")
    last = np.full(B.shape[1:], np.nan, "float32")
    for k, y in enumerate(yrs):
        last[B[k] == 1] = y
    since = np.where(np.isfinite(last), yrs[-1] - last, np.nan).astype("float32")
    for n, a in (("burn_frequency_registered", freq),
                 ("burn_years_count", nburn.astype("float32")),
                 ("last_registered_burn_year", last),
                 ("years_since_last_registered_burn", since)):
        a = a.copy()
        a[~am] = np.nan
        write_raster(prj.rasters / "fire" / f"{n}.tif", a, grid,
                     tags=dict(source="MCD64A1.061",
                               note="зарегистрированные выгорания; небольшие "
                                    "пожары могут быть пропущены; это не "
                                    "прогноз пожарной опасности"))
    # предпожарная лесная маска и отклик
    thr = cfg["treecover"]["primary_threshold_pct"]
    classes = cfg["landcover"]["classes"]
    rows, resp = [], []
    evi_layer = FUNC_VARS["evi_terra"]
    T = len(yrs)
    for k, y in enumerate(yrs):
        b = B[k]
        pre_y = y - 1
        lp = annual_path(prj, "lc", pre_y)
        pp = annual_path(prj, "ptc", pre_y)
        pre = np.zeros(b.shape, bool)
        src = []
        if lp.exists():
            pre |= np.isin(read_raster(lp), classes)
            src.append("lc")
        if pp.exists() and cfg["fire"]["prefire_mask"] in ("woody_or_ptc",
                                                           "ptc"):
            pre |= read_raster(pp) >= thr
            src.append("ptc")
        burned_forest = am & pre & (b == 1)
        rows.append(dict(year=y, unit_type="aoi", unit_id="AOI",
                         burned_area_ha=float(np.nansum(b[am]) * grid.cell_ha),
                         burned_prefire_forest_ha=float(burned_forest.sum() *
                                                        grid.cell_ha),
                         prefire_forest_ha=float((am & pre).sum() * grid.cell_ha),
                         observed_fraction=float(np.isfinite(b[am]).mean()),
                         prefire_mask_source="+".join(src) or "none"))
        # отклик EVI: pre = средн. (y-3..y-1), post = y+1
        if not burned_forest.any():
            continue
        prep = [annual_path(prj, evi_layer, yy) for yy in range(y - 3, y)]
        prev = [read_raster(p) for p in prep if p.exists()]
        if not prev:
            continue
        pre_evi = np.nanmean(np.stack(prev), 0)
        r_, c_ = np.where(burned_forest)
        pe = pre_evi[r_, c_]
        rec_year = np.full(len(r_), np.nan)
        drop = np.full(len(r_), np.nan)
        tol = cfg["fire"]["recovery_tolerance"]
        maxy = cfg["fire"]["recovery_years_max"]
        followup = 0
        for j, yy in enumerate(range(y + 1, y + 1 + maxy)):
            p = annual_path(prj, evi_layer, yy)
            if not p.exists():
                break
            followup += 1
            v = read_raster(p)[r_, c_]
            if j == 0:
                drop = (pe - v) / pe
            ok = np.isnan(rec_year) & (v >= pe * (1 - tol))
            rec_year[ok] = yy - y
        lc0 = read_raster(lp)[r_, c_] if lp.exists() else np.full(len(r_), 0)
        resp.append(pd.DataFrame(dict(
            fire_year=y, pixel_id=r_ * grid.width + c_, prefire_class=lc0,
            prefire_evi=pe, relative_evi_drop_y1=drop,
            recovery_years=rec_year, followup_years=followup,
            recovery_status=np.where(np.isfinite(rec_year), "recovered",
                                     np.where(followup >= maxy,
                                              "not_recovered_within_max",
                                              "not_recovered_censored")))))
    fa = pd.DataFrame(rows)
    fa["burned_fraction_of_prefire_forest"] = fa["burned_prefire_forest_ha"] / \
        fa["prefire_forest_ha"].replace(0, np.nan)
    save_table(prj, fa, "fire_annual")
    fr = pd.concat(resp, ignore_index=True) if resp else pd.DataFrame()
    save_table(prj, fr, "fire_response_pixels")
    if len(fr):
        summ = fr.groupby(["fire_year", "prefire_class"]).agg(
            n_pixels=("pixel_id", "count"),
            median_relative_evi_drop=("relative_evi_drop_y1", "median"),
            median_recovery_years=("recovery_years", "median"),
            share_recovered=("recovery_status",
                             lambda s: (s == "recovered").mean())).reset_index()
        save_table(prj, summ, "fire_response")
    return {"years": len(yrs), "fire_pixels": int(len(fr))}


# =============================================================================
# 18. Панель окон и связи показателей
# =============================================================================
def window_panel(prj: Project, size_km: Optional[float] = None) -> pd.DataFrame:
    """Средние по окнам и годам для всех ежегодных переменных."""
    cfg = prj.cfg
    grid = prj.grid()
    size_km = size_km or cfg["fragmentation"]["main_window_km"]
    wins, n = window_grid(grid, size_km)
    am = grid.aoi_mask
    lab = np.full((grid.height, grid.width), -1, "int32")
    ids = []
    for k, (wid, r0, r1, c0, c1) in enumerate(wins):
        lab[r0:r1, c0:c1] = k
        ids.append(wid)
    lab[~am] = -1
    flat = lab.ravel()
    ok = flat >= 0
    nW = len(wins)
    cnt = np.bincount(flat[ok], minlength=nW)
    variables = {"ptc": "ptc", **FUNC_VARS, **LST_VARS}
    rows = []
    for y in prj.years:
        rec = {"year": np.full(nW, y), "window_id": np.array(ids),
               "aoi_cells": cnt}
        for v, layer in variables.items():
            p = annual_path(prj, layer, y)
            if not p.exists():
                continue
            a = read_raster(p).ravel()
            m = ok & np.isfinite(a)
            s = np.bincount(flat[m], weights=a[m], minlength=nW)
            c = np.bincount(flat[m], minlength=nW)
            with np.errstate(all="ignore"):
                rec[v] = np.where(c > 0, s / c, np.nan)
                rec[f"{v}_valid_fraction"] = np.where(cnt > 0, c / cnt, np.nan)
        bp = annual_path(prj, "burned", y)
        if bp.exists():
            b = read_raster(bp).astype("float32").ravel()
            b[b == NODATA_U8] = np.nan
            m = ok & np.isfinite(b)
            s = np.bincount(flat[m], weights=b[m], minlength=nW)
            c = np.bincount(flat[m], minlength=nW)
            rec["burned_fraction"] = np.where(c > 0, s / c, np.nan)
        rows.append(pd.DataFrame(rec))
    df = pd.concat(rows, ignore_index=True)
    df = df[df.aoi_cells > 0]
    fr = load_table(prj, "fragmentation_annual")
    if fr is not None and size_km == cfg["fragmentation"]["main_window_km"]:
        mk = f"ptc{int(cfg['treecover']['primary_threshold_pct'])}"
        f = fr[(fr.unit_type == "window") & (fr["mask"] == mk)][
            ["unit_id", "year", "PLAND", "ED", "CPLAND", "COHESION", "MESH",
             "PD", "status"]].rename(columns={"unit_id": "window_id",
                                              "status": "frag_status"})
        df = df.merge(f, on=["window_id", "year"], how="left")
    return df


def stage_associations(prj: Project):
    from scipy import stats
    cfg = prj.cfg
    ac = cfg["associations"]
    st = cfg["statistics"]
    panel = window_panel(prj)
    save_table(prj, panel, "panel_window_year")
    responses = [v for v in ("evi_terra", "lai", "gpp", "etpet")
                 if v in panel.columns]
    lst = "lstday_terra" if "lstday_terra" in panel.columns else None
    preds = [p for p in (lst, "ptc", "PLAND", "CPLAND", "COHESION",
                         "burned_fraction") if p and p in panel.columns]
    rows = []
    # (а) временные связи внутри окон: аномалии, лаги 0..2, исходные и детрендированные
    for resp_v in responses:
        for pred in preds:
            if pred == resp_v:
                continue
            for lag in ac["lags"]:
                for mode in ("raw_anomaly", "detrended"):
                    rs, ps = [], []
                    for wid, g in panel.groupby("window_id"):
                        g = g.sort_values("year")
                        x = g[pred].values.astype(float)
                        yv = g[resp_v].values.astype(float)
                        if lag:
                            x = x[:-lag]
                            yv = yv[lag:]
                        yrs_ = g["year"].values[lag:] if lag else g["year"].values
                        m = np.isfinite(x) & np.isfinite(yv)
                        if m.sum() < ac["min_years"]:
                            continue
                        xx, yy = x[m], yv[m]
                        if mode == "detrended":
                            tt = yrs_[m]
                            xx = xx - np.polyval(np.polyfit(tt, xx, 1), tt)
                            yy = yy - np.polyval(np.polyfit(tt, yy, 1), tt)
                        else:
                            xx, yy = xx - xx.mean(), yy - yy.mean()
                        if np.std(xx) == 0 or np.std(yy) == 0:
                            continue
                        r, p = stats.spearmanr(xx, yy)
                        rs.append(r)
                        ps.append(p)
                    if not rs:
                        continue
                    q = bh_qvalues(np.array(ps))
                    rows.append(dict(analysis="temporal_within_window",
                                     response=resp_v, predictor=pred, lag=lag,
                                     mode=mode, n_windows=len(rs),
                                     median_r=float(np.median(rs)),
                                     q25_r=float(np.percentile(rs, 25)),
                                     q75_r=float(np.percentile(rs, 75)),
                                     share_p_lt_alpha=float(np.mean(np.array(ps) < st["alpha"])),
                                     share_q_lt_alpha=float(np.mean(q < st["alpha"])),
                                     interpretation="статистическая связь, не "
                                                    "причинный вклад"))
    # (б) пространственные связи средних и трендов по окнам
    agg = panel.groupby("window_id")[responses + preds].mean()
    for resp_v in responses:
        for pred in preds:
            if pred == resp_v:
                continue
            x, yv = agg[pred].values, agg[resp_v].values
            m = np.isfinite(x) & np.isfinite(yv)
            if m.sum() < 10:
                continue
            r, p = stats.spearmanr(x[m], yv[m])
            rows.append(dict(analysis="spatial_period_mean", response=resp_v,
                             predictor=pred, lag=0, mode="period_mean",
                             n_windows=int(m.sum()), median_r=float(r),
                             p_value_naive=float(p),
                             interpretation="p без учёта пространственной "
                                            "автокорреляции (завышенная "
                                            "значимость возможна)"))
    df = pd.DataFrame(rows)
    if "p_value_naive" in df:
        m = df["analysis"] == "spatial_period_mean"
        df.loc[m, "q_value_naive"] = bh_qvalues(df.loc[m, "p_value_naive"].values)
    save_table(prj, df, "associations")
    return {"rows": int(len(df))}


# =============================================================================
# 19. Относительная уязвимость (уровень аналитических окон)
# =============================================================================
def zero_preserving_rank(adverse: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """0 → 0; положительные → ранг в (0,1] внутри группы; NaN → NaN."""
    out = np.full(len(adverse), np.nan)
    for g in pd.unique(groups):
        m = groups == g
        a = adverse[m]
        r = np.full(len(a), np.nan)
        fin = np.isfinite(a)
        r[fin & (a <= 0)] = 0.0
        pos = fin & (a > 0)
        if pos.any():
            from scipy.stats import rankdata
            r[pos] = rankdata(a[pos], method="average") / pos.sum()
        out[m] = r
    return out


def geo_mean(cols: List[np.ndarray], weights: List[float]) -> np.ndarray:
    M = np.vstack(cols)
    w = np.asarray(weights, float)[:, None]
    with np.errstate(all="ignore"):
        lg = np.where(M > 0, np.log(np.where(M > 0, M, 1)), -np.inf)
        g = np.exp((w * lg).sum(0) / w.sum())
    g[np.any(M == 0, 0)] = 0.0
    g[np.any(~np.isfinite(M), 0)] = np.nan
    return g


def window_trends(panel: pd.DataFrame, cols: List[str], min_n: int,
                  alpha: float) -> pd.DataFrame:
    rows = []
    for wid, g in panel.groupby("window_id"):
        g = g.sort_values("year")
        rec = {"window_id": wid}
        for c in cols:
            if c not in g:
                continue
            r = mk_sen_series(g["year"].values, g[c].values, alpha,
                              autocorr=False)
            rec[f"{c}_slope"] = r["slope"] if r["n"] >= min_n else np.nan
            rec[f"{c}_p"] = r["p_value"] if r["n"] >= min_n else np.nan
            rec[f"{c}_n"] = r["n"]
        rows.append(rec)
    return pd.DataFrame(rows)


def compute_vulnerability(prj: Project, panel: pd.DataFrame,
                          vcfg: dict, lst_role: str,
                          size_km: float) -> pd.DataFrame:
    cfg = prj.cfg
    grid = prj.grid()
    st = cfg["statistics"]
    min_n = st["min_valid_years"]
    cols_cover = {"ptc_mean": "ptc", "pland": "PLAND"}
    cols_conf = {"cpland": "CPLAND", "cohesion": "COHESION", "mesh": "MESH"}
    cols_f = {"evi_terra": "evi_terra", "lai": "lai", "gpp": "gpp",
              "etpet": "etpet"}
    need = [cols_cover.get(c) for c in vcfg["s_cover_metrics"]] + \
           [cols_conf.get(c) for c in vcfg["s_config_metrics"]] + \
           [cols_f.get(c) for c in vcfg["f_metrics"] if c in cols_f] + \
           ["lstday_terra", "burned_fraction"]
    need = [c for c in need if c and c in panel.columns]
    tr = window_trends(panel, need, min_n, st["alpha"])
    base = panel.groupby("window_id").agg(
        aoi_cells=("aoi_cells", "first"),
        ptc_mean_period=("ptc", "mean") if "ptc" in panel else ("aoi_cells", "first"),
        burned_mean=("burned_fraction", "mean") if "burned_fraction" in panel
        else ("aoi_cells", lambda s: np.nan)).reset_index()
    qcols = [c for c in panel.columns if c.endswith("_valid_fraction")]
    q = panel.groupby("window_id")[qcols].mean().min(axis=1).rename("Q_min_valid_fraction") \
        if qcols else None
    df = base.merge(tr, on="window_id", how="left")
    if q is not None:
        df = df.merge(q.reset_index(), on="window_id", how="left")
    # доминирующий класс базового периода
    b0, b1 = cfg["period"]["comparison_baseline"]
    wins, n = window_grid(grid, size_km)
    lcs = [read_raster(annual_path(prj, "lc", y)) for y in range(b0, b1 + 1)
           if annual_path(prj, "lc", y).exists()]
    dom, zone_dom, woody_ha = {}, {}, {}
    if lcs:
        lcm = lcs[0]
        classes = cfg["landcover"]["classes"]
        for wid, r0, r1, c0, c1 in wins:
            sub = lcm[r0:r1, c0:c1][grid.aoi_mask[r0:r1, c0:c1]]
            cnts = {c: int((sub == c).sum()) for c in classes}
            tot = sum(cnts.values())
            dom[wid] = f"lc{max(cnts, key=cnts.get)}" if tot else "no_woody"
            woody_ha[wid] = tot * grid.cell_ha
            if grid.zones is not None:
                zs = grid.zones[r0:r1, c0:c1][grid.aoi_mask[r0:r1, c0:c1]]
                zs = zs[zs > 0]
                zone_dom[wid] = grid.zone_names.get(
                    int(np.bincount(zs).argmax()), "none") if len(zs) else "none"
    df["dominant_class"] = df["window_id"].map(dom).astype(object).fillna("unknown")
    df["zone"] = df["window_id"].map(zone_dom).astype(object).fillna("all")
    df["woody_area_baseline_ha"] = df["window_id"].map(woody_ha)
    # страты
    strat = []
    for _, r in df.iterrows():
        parts = []
        if "dominant_class" in vcfg["stratify_by"]:
            parts.append(r["dominant_class"])
        if "zone" in vcfg["stratify_by"]:
            parts.append(r["zone"])
        strat.append("|".join(parts) or "all")
    df["stratum"] = strat
    vc = df["stratum"].value_counts()
    small = df["stratum"].map(vc) < vcfg["min_stratum_windows"]
    df["stratum_merged"] = small
    df.loc[small, "stratum"] = "merged_small_strata"
    groups = df["stratum"].values

    def adverse(col, direction=-1):
        s = df.get(f"{col}_slope")
        if s is None:
            return None
        a = np.maximum(0, direction * s.values.astype(float))
        if vcfg["only_significant_trends"]:
            p = df[f"{col}_p"].values
            a = np.where(p < st["alpha"], a, np.where(np.isfinite(a), 0, np.nan))
        return a

    comp = {}
    for name, mapping, dirn, mets in (
            ("S_cover", cols_cover, -1, vcfg["s_cover_metrics"]),
            ("S_config", cols_conf, -1, vcfg["s_config_metrics"])):
        rs = []
        for m_ in mets:
            a = adverse(mapping[m_], dirn) if mapping.get(m_) else None
            if a is not None:
                r = zero_preserving_rank(a, groups)
                df[f"r_{m_}"] = r
                rs.append(r)
        comp[name] = geo_mean(rs, [1] * len(rs)) if rs else None
    rs = []
    for m_ in vcfg["f_metrics"]:
        if m_ == "functional_nonrecovery":
            continue
        col = cols_f.get(m_)
        a = adverse(col, -1) if col else None
        if a is not None:
            r = zero_preserving_rank(a, groups)
            df[f"r_{m_}"] = r
            rs.append(r)
    if lst_role == "stress_component" and "lstday_terra_slope" in df:
        a = adverse("lstday_terra", +1)
        r = zero_preserving_rank(a, groups)
        df["r_lst_warming"] = r
        rs.append(r)
    comp["F"] = geo_mean(rs, [1] * len(rs)) if rs else None
    for k, v in comp.items():
        df[k] = v if v is not None else np.nan
    w = vcfg["weights"]
    if comp.get("S_cover") is not None and comp.get("S_config") is not None:
        df["S"] = geo_mean([df["S_cover"].values, df["S_config"].values],
                           [w["S_cover"], w["S_config"]])
    else:
        df["S"] = df.get("S_cover", np.nan)
    # экспозиция и историческая пожарная экспозиция (глобальное ранжирование)
    E_raw = df["woody_area_baseline_ha"].fillna(0).values * \
        df["ptc_mean_period"].fillna(0).values / 100.0
    df["E_raw_tree_cover_equiv_ha"] = E_raw
    df["E"] = zero_preserving_rank(E_raw, np.zeros(len(df)))
    df["H_raw_mean_annual_burned_fraction"] = df["burned_mean"]
    df["H"] = zero_preserving_rank(df["burned_mean"].values.astype(float),
                                   np.zeros(len(df)))
    df["max_adverse_component"] = df[["S_cover", "S_config", "F"]].max(axis=1)
    status = vcfg.get("status")
    if status in (None, "", "undefined"):
        df["P_SF"] = np.nan
        df["P_fire"] = np.nan
        df["index_status"] = "not_computed_parameters_undefined"
    else:
        df["P_SF"] = geo_mean([df["S"].values, df["F"].values], [1, 1])
        df["P_SF_arithmetic"] = (df["S"] + df["F"]) / 2
        df["P_fire"] = geo_mean([df["H"].values, df["E"].values,
                                 df["P_SF"].values], [1, 1, 1])
        df["index_status"] = status
    df["scenario_id"] = vcfg.get("scenario_id")
    df["lst_role"] = lst_role
    df["window_km"] = size_km
    df["support"] = "analytical_window"
    return df


def stage_vulnerability(prj: Project):
    cfg = prj.cfg
    grid = prj.grid()
    panel = load_table(prj, "panel_window_year")
    if panel is None:
        panel = window_panel(prj)
    v = compute_vulnerability(prj, panel, cfg["vulnerability"],
                              cfg["lst"]["role"],
                              cfg["fragmentation"]["main_window_km"])
    comp_cols = [c for c in v.columns if c.startswith("r_") or c in (
        "S_cover", "S_config", "S", "F", "E", "H", "Q_min_valid_fraction",
        "E_raw_tree_cover_equiv_ha", "H_raw_mean_annual_burned_fraction",
        "max_adverse_component", "stratum", "stratum_merged",
        "dominant_class", "zone")] + [c for c in v.columns if c.endswith("_slope")
                                      or c.endswith("_p") or c.endswith("_n")]
    save_table(prj, v[["window_id"] + comp_cols], "vulnerability_components")
    idx_cols = ["window_id", "S", "F", "H", "E", "P_SF", "P_SF_arithmetic",
                "P_fire", "max_adverse_component", "Q_min_valid_fraction",
                "index_status", "scenario_id", "lst_role", "window_km",
                "support", "stratum"]
    idx = v[[c for c in idx_cols if c in v.columns]].copy()
    tf = cfg["vulnerability"]["top_fraction"]
    if idx["P_SF"].notna().any():
        thr = idx["P_SF"].quantile(1 - tf)
        idx["top_priority_P_SF"] = idx["P_SF"] >= thr
    save_table(prj, idx, "priority_indices")
    wg = windows_gdf(grid, cfg["fragmentation"]["main_window_km"])
    gv = wg.merge(v, on="window_id", how="left")
    gv.to_file(prj.vectors / "vulnerability_windows.gpkg", driver="GPKG")
    # контекстные растры окон (для отображения)
    wins, n = window_grid(grid, cfg["fragmentation"]["main_window_km"])
    vi = v.set_index("window_id")
    for col in ("S", "F", "H", "E", "P_SF", "P_fire"):
        if col not in vi:
            continue
        a = np.full((grid.height, grid.width), np.nan, "float32")
        for wid, r0, r1, c0, c1 in wins:
            if wid in vi.index:
                a[r0:r1, c0:c1] = vi.at[wid, col]
        a[~grid.aoi_mask] = np.nan
        write_raster(prj.rasters / "vulnerability" / f"window_context_{col}.tif",
                     a, grid, tags=dict(meaning="контекстный показатель "
                                                "аналитического окна, не "
                                                "независимая пиксельная оценка",
                                        status=str(cfg["vulnerability"]["status"])))
    return {"windows": int(len(v))}


# =============================================================================
# 20. Анализ чувствительности
# =============================================================================
def run_sensitivity(prj: Project):
    from scipy.stats import spearmanr
    cfg = prj.cfg
    sc = cfg["sensitivity"]
    fc = cfg["fragmentation"]
    st = cfg["statistics"]
    main_v = load_table(prj, "priority_indices")
    yrs = [y for y in prj.years if annual_path(prj, "ptc", y).exists()]
    rows = []
    base_panel = load_table(prj, "panel_window_year")
    sub = "sensitivity"

    def summarize(sid, param, value, frag_df=None, vul=None, extra=None):
        rec = dict(scenario_id=sid, parameter=param, value=str(value))
        if frag_df is not None and len(frag_df):
            a = frag_df[frag_df.unit_type == "aoi"]
            for mk, g in a.groupby("mask"):
                r = mk_sen_series(g.year.values, g.PLAND.values,
                                  autocorr=False)
                rec[f"aoi_PLAND_slope_{mk}"] = r["slope"]
                r = mk_sen_series(g.year.values, g.CPLAND.values,
                                  autocorr=False)
                rec[f"aoi_CPLAND_slope_{mk}"] = r["slope"]
        if vul is not None and main_v is not None and "P_SF" in vul:
            m = main_v.merge(vul[["window_id", "P_SF"]], on="window_id",
                             suffixes=("_main", "_scen"))
            m = m.dropna(subset=["P_SF_main", "P_SF_scen"])
            if len(m) > 5:
                rec["rank_spearman_P_SF"] = spearmanr(m.P_SF_main,
                                                      m.P_SF_scen)[0]
                tf = cfg["vulnerability"]["top_fraction"]
                a = set(m.window_id[m.P_SF_main >= m.P_SF_main.quantile(1 - tf)])
                b = set(m.window_id[m.P_SF_scen >= m.P_SF_scen.quantile(1 - tf)])
                rec["top_overlap_jaccard"] = len(a & b) / max(1, len(a | b))
                rec["n_windows_compared"] = len(m)
        if extra:
            rec.update(extra)
        rows.append(rec)

    if "threshold" in sc["run"]:
        for t in cfg["treecover"]["sensitivity_thresholds_pct"]:
            f = compute_fragmentation(prj, yrs, fc["main_window_km"],
                                      fc["edge_depth_cells"], t, ["ptc"],
                                      enn=False)
            save_table(prj, f[f.unit_type == "aoi"], f"frag_threshold_{t}", sub)
            vul = None
            if base_panel is not None:
                p2 = base_panel.drop(columns=[c for c in ("PLAND", "ED",
                                                          "CPLAND", "COHESION",
                                                          "MESH", "PD",
                                                          "frag_status")
                                              if c in base_panel])
                ff = f[f.unit_type == "window"][["unit_id", "year", "PLAND",
                                                 "ED", "CPLAND", "COHESION",
                                                 "MESH", "PD"]].rename(
                    columns={"unit_id": "window_id"})
                p2 = p2.merge(ff, on=["window_id", "year"], how="left")
                vul = compute_vulnerability(prj, p2, cfg["vulnerability"],
                                            cfg["lst"]["role"],
                                            fc["main_window_km"])
            summarize(f"thr{t}", "ptc_threshold_pct", t, f, vul)
    if "edge_depth" in sc["run"]:
        for d in [fc["edge_depth_cells"]] + fc["sensitivity_edge_depths"]:
            f = compute_fragmentation(prj, yrs, fc["main_window_km"], d,
                                      cfg["treecover"]["primary_threshold_pct"],
                                      ["ptc"], enn=False)
            summarize(f"edge{d}", "edge_depth_cells", d, f)
    if "window" in sc["run"]:
        biggest = max([fc["main_window_km"]] + fc["sensitivity_windows_km"])
        for wkm in fc["sensitivity_windows_km"]:
            f = compute_fragmentation(prj, yrs, wkm, fc["edge_depth_cells"],
                                      cfg["treecover"]["primary_threshold_pct"],
                                      ["ptc"], enn=False)
            save_table(prj, f[f.unit_type == "window"], f"frag_window_{wkm}km",
                       sub)
            # согласованная поддержка: среднее по вложенным окнам в окне biggest
            w = f[(f.unit_type == "window")].copy()
            ratio = biggest / wkm
            if float(ratio).is_integer():
                rc = w.unit_id.str.extract(r"_r(\d+)_c(\d+)").astype(int)
                w["parent"] = [f"W{int(biggest)}km_r{int(r // ratio):03d}_c"
                               f"{int(c // ratio):03d}" for r, c in rc.values]
                agg = w.groupby(["parent", "year"])[["PLAND", "CPLAND"]].mean()
                save_table(prj, agg.reset_index(),
                           f"frag_window_{wkm}km_on_{int(biggest)}km_support", sub)
            summarize(f"win{wkm}", "window_km", wkm, f)
    if "persistence" in sc["run"]:
        lyrs = [y for y in prj.years if annual_path(prj, "lc", y).exists()]
        if lyrs:
            stack = np.stack([read_raster(annual_path(prj, "lc", y))
                              for y in lyrs])
            for pv in sc["persistence_values"]:
                ev, mat = detect_transitions(prj, stack, lyrs, pv, pv,
                                             cfg["landcover"]["classes"])
                save_table(prj, mat, f"transition_matrix_persist{pv}", sub)
                n_conf = int((ev.status == "confirmed").sum()) if len(ev) else 0
                summarize(f"persist{pv}", "persistence_years", pv,
                          extra=dict(confirmed_transition_pixels=n_conf))
    if "baseline" in sc["run"]:
        b_alt = cfg["period"]["comparison_baseline"]
        for var in ("evi_terra",):
            r = functional_pixel_stats(prj, var + "_altbase", FUNC_VARS[var],
                                       anomaly_baseline=b_alt,
                                       out_dir="sensitivity_functional")
            if r is not None:
                out, _ = r
                main = read_raster(prj.rasters / "functional" /
                                   f"{var}_adverse_years_z.tif")
                alt = out["adverse_years_z"]
                m = np.isfinite(main) & np.isfinite(alt)
                summarize(f"baseline_{b_alt[0]}_{b_alt[1]}", "anomaly_baseline",
                          b_alt, extra=dict(
                              adverse_years_z_mean_main=float(main[m].mean())
                              if m.any() else np.nan,
                              adverse_years_z_mean_alt=float(alt[m].mean())
                              if m.any() else np.nan))
    if "aggregation" in sc["run"] and main_v is not None and \
            "P_SF_arithmetic" in main_v:
        m = main_v.dropna(subset=["P_SF", "P_SF_arithmetic"])
        if len(m) > 5:
            tf = cfg["vulnerability"]["top_fraction"]
            a = set(m.window_id[m.P_SF >= m.P_SF.quantile(1 - tf)])
            b = set(m.window_id[m.P_SF_arithmetic >=
                                m.P_SF_arithmetic.quantile(1 - tf)])
            summarize("agg_arith", "aggregation", "arithmetic_mean",
                      extra=dict(rank_spearman_P_SF=spearmanr(
                          m.P_SF, m.P_SF_arithmetic)[0],
                          top_overlap_jaccard=len(a & b) / max(1, len(a | b))))
    if "lst_role" in sc["run"] and base_panel is not None:
        alt = "stress_component" if cfg["lst"]["role"] == "explanatory" \
            else "explanatory"
        vul = compute_vulnerability(prj, base_panel, cfg["vulnerability"], alt,
                                    fc["main_window_km"])
        summarize(f"lst_{alt}", "lst_role", alt, vul=vul)
    if sc.get("include_qa_profiles"):
        LOG.warning("QA-сценарии требуют повторного prepare в отдельном "
                    "проекте: скопируйте config.yaml, задайте quality.profile "
                    "и project.root, затем выполните run.")
    df = pd.DataFrame(rows)
    save_table(prj, df, "sensitivity_results")
    # согласованность знаков
    sign_cols = [c for c in df.columns if c.startswith("aoi_") and
                 c.endswith(tuple(f"_{m}" for m in ["ptc30", "woody"]))]
    summ = []
    for c in [c for c in df.columns if c.startswith("aoi_")]:
        v = df[c].dropna()
        if len(v):
            summ.append(dict(metric=c, n_scenarios=len(v),
                             share_negative=float((v < 0).mean()),
                             share_positive=float((v > 0).mean()),
                             sign_consistent=bool((v < 0).all() or
                                                  (v > 0).all()),
                             min=float(v.min()), max=float(v.max()),
                             mean=float(v.mean())))
    for c in ("rank_spearman_P_SF", "top_overlap_jaccard"):
        if c in df:
            v = df[c].dropna()
            if len(v):
                summ.append(dict(metric=c, n_scenarios=len(v),
                                 min=float(v.min()), max=float(v.max()),
                                 mean=float(v.mean())))
    save_table(prj, pd.DataFrame(summ), "sensitivity_summary")
    return {"scenarios": int(len(df))}


# =============================================================================
# 21. Рисунки
# =============================================================================
def _savefig(prj: Project, fig, name: str, data: Optional[pd.DataFrame],
             captions: list, cap_ru: str, cap_en: str):
    import matplotlib.pyplot as plt
    for ext in ("png", "svg"):
        fig.savefig(prj.figures / f"{name}.{ext}", dpi=200 if ext == "png"
                    else None, bbox_inches="tight")
    plt.close(fig)
    if data is not None:
        data.to_csv(prj.figures / f"{name}_data.csv", index=False,
                    encoding="utf-8")
    captions.append(dict(figure=name, caption_ru=cap_ru, caption_en=cap_en,
                         data_file=f"{name}_data.csv" if data is not None
                         else ""))


def make_figures(prj: Project) -> List[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = prj.cfg
    caps: List[dict] = []
    ref = cfg["reporting"]["reference_years"]

    a = load_table(prj, "annual_class_areas")
    if a is not None and len(a):
        s = a[(a.unit_type == "aoi") & (a.version == "raw")]
        if len(s):
            # только изучаемые древесные классы, присутствующие в AOI
            keep_cls = set(cfg["landcover"]["classes"])
            s = s[s.cover_class.isin(keep_cls)]
            s = s[s.groupby("cover_class").area_ha.transform("max") > 0]
            a = a[a.cover_class.isin(set(s.cover_class))]
            fig, ax = plt.subplots(figsize=(10, 4.5))
            for c, g in s.groupby("cover_class"):
                if c in IGBP_NAMES_RU:
                    ax.plot(g.year, g.area_ha / 1000, marker="o", ms=3,
                            label=f"{c}: {IGBP_NAMES_RU[c]}")
            sm = a[(a.unit_type == "aoi") & (a.version == "mode3")]
            for c, g in sm.groupby("cover_class"):
                if c in IGBP_NAMES_RU:
                    ax.plot(g.year, g.area_ha / 1000, ls="--", lw=0.8,
                            color="grey")
            for y in ref:
                ax.axvline(y, color="k", lw=0.3, alpha=0.4)
            ax.set_xlabel("Год")
            ax.set_ylabel("Площадь, тыс. га")
            ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1),
                      frameon=False)
            fig.subplots_adjust(right=0.62)
            ax.set_title("Площади классов MCD12Q1 (сплошные — исходные; "
                         "пунктир — мода 3 лет)", fontsize=9)
            _savefig(prj, fig, "fig01_class_areas", s, caps,
                     "Ежегодные площади древесных классов IGBP MCD12Q1 в "
                     "пределах территории; вертикальные линии — пятилетние "
                     "опорные годы. Межгодовые колебания включают "
                     "классификационный шум продукта.",
                     "Annual areas of MCD12Q1 IGBP woody classes; vertical "
                     "lines mark five-year reference years. Interannual "
                     "variation includes product classification noise.")

    m = load_table(prj, "transition_matrix_persistent")
    if m is not None and len(m) and {"from_class", "to_class"} <= set(m):
        col = "area_ha" if "area_ha" in m else m.columns[-1]
        pv = m.pivot_table(index="from_class", columns="to_class",
                           values=col, aggfunc="sum", fill_value=0)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(pv.values, cmap="viridis")
        ax.set_xticks(range(len(pv.columns)), [str(c) for c in pv.columns])
        ax.set_yticks(range(len(pv.index)), [str(c) for c in pv.index])
        ax.set_xlabel("Новый класс")
        ax.set_ylabel("Исходный класс")
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                ax.text(j, i, f"{pv.values[i, j]:.0f}", ha="center",
                        va="center", fontsize=6, color="w")
        fig.colorbar(im, ax=ax, label="га")
        ax.set_title("Кандидатные устойчивые переходы", fontsize=9)
        _savefig(prj, fig, "fig02_transition_matrix", pv.reset_index(), caps,
                 "Матрица кандидатных устойчивых переходов меток MCD12Q1 "
                 "(правило устойчивости до/после перехода). Не является "
                 "доказательством смены лесных сообществ.",
                 "Matrix of candidate persistent MCD12Q1 label transitions; "
                 "not proof of community change.")

    t = load_table(prj, "treecover_timeseries")
    if t is not None and len(t):
        s = t[t.unit_type == "aoi"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(s.year, s.mean_ptc_pct, marker="o")
        ax.set_xlabel("Год")
        ax.set_ylabel("Средняя доля древесного покрытия, %")
        ax.set_title("MOD44B Percent Tree Cover", fontsize=9)
        _savefig(prj, fig, "fig03_treecover", s, caps,
                 "Средняя доля древесного покрытия MOD44B в пределах "
                 "территории.", "Mean MOD44B percent tree cover over the AOI.")

    f = load_table(prj, "fragmentation_annual")
    if f is not None and len(f):
        s = f[f.unit_type == "aoi"]
        mets = ["PLAND", "ED", "CPLAND", "COHESION", "MESH", "PD"]
        fig, axs = plt.subplots(2, 3, figsize=(11, 6))
        for ax, met in zip(axs.ravel(), mets):
            for mk, g in s.groupby("mask"):
                ax.plot(g.year, g[met], marker=".", label=mk)
            ax.set_title(f"{met}, {FRAG_UNITS.get(met, '')}", fontsize=9)
        axs[0, 0].legend(fontsize=7)
        fig.tight_layout()
        _savefig(prj, fig, "fig04_fragmentation_aoi", s, caps,
                 "Метрики фрагментации для территории в целом по маскам "
                 "(ptc — порог MOD44B; closed/sparse/woody — классы MCD12Q1). "
                 "Изменения конфигурации следует читать вместе с PLAND.",
                 "AOI-level fragmentation metrics by mask; configuration "
                 "changes must be read together with PLAND.")

    fa = load_table(prj, "functional_annual")
    if fa is not None and len(fa):
        s = fa[fa.stratum == "all_aoi"]
        vs = sorted(s.variable.unique())
        n = len(vs)
        cols = 3
        rows = int(math.ceil(n / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(11, 3 * rows),
                                squeeze=False)
        for ax, v in zip(axs.ravel(), vs):
            g = s[s.variable == v]
            ax.plot(g.year, g["mean"], marker=".")
            ax.set_title(v, fontsize=9)
        for ax in axs.ravel()[n:]:
            ax.axis("off")
        fig.tight_layout()
        _savefig(prj, fig, "fig05_functional_series", s, caps,
                 "Средние сезонные функциональные показатели и LST по "
                 "территории (Terra и Aqua раздельно).",
                 "AOI-mean seasonal functional variables and LST "
                 "(Terra and Aqua separate).")

    fi = load_table(prj, "fire_annual")
    if fi is not None and len(fi) and "burned_prefire_forest_ha" in fi:
        s = fi[fi.unit_type == "aoi"] if "unit_type" in fi else fi
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(s.year, s.burned_prefire_forest_ha / 1000)
        ax.set_xlabel("Год")
        ax.set_ylabel("Тыс. га")
        ax.set_title("Зарегистрированные выгорания в предпожарном лесу "
                     "(MCD64A1)", fontsize=9)
        _savefig(prj, fig, "fig06_fire", s, caps,
                 "Площадь зарегистрированных MCD64A1 выгораний в пределах "
                 "предпожарной древесной маски. Небольшие пожары продуктом "
                 "систематически недоучитываются.",
                 "Area of MCD64A1-registered burns within the pre-fire "
                 "woody mask; small fires are under-detected.")

    gp = prj.vectors / "vulnerability_windows.gpkg"
    if gp.exists():
        try:
            import geopandas as gpd
            g = gpd.read_file(gp)
            comps = [c for c in ("S_cover", "S_config", "F", "H", "E", "P_SF",
                                 "P_fire", "Q_min_valid_fraction") if c in g]
            n = len(comps)
            cols = 4
            rows = int(math.ceil(n / cols))
            fig, axs = plt.subplots(rows, cols, figsize=(14, 3.6 * rows),
                                    squeeze=False)
            for ax, c in zip(axs.ravel(), comps):
                g.plot(column=c, ax=ax, cmap="magma_r" if c != "Q_min_valid_fraction"
                       else "Greens", vmin=0, vmax=1, legend=True,
                       missing_kwds=dict(color="lightgrey"))
                ax.set_title(c, fontsize=9)
                ax.set_axis_off()
            for ax in axs.ravel()[n:]:
                ax.axis("off")
            fig.tight_layout()
            _savefig(prj, fig, "fig07_vulnerability_components",
                     pd.DataFrame(g.drop(columns="geometry")), caps,
                     "Компоненты относительной уязвимости и индексы "
                     "приоритета по аналитическим окнам (шкала 0–1, "
                     "процентили внутри страт). Серый — нет данных; низкое "
                     "качество данных не означает низкой уязвимости.",
                     "Relative vulnerability components and priority indices "
                     "by analysis window (0–1 within-stratum ranks). Grey: "
                     "no data.")
        except Exception as e:
            LOG.warning(f"Карта уязвимости не построена: {e}")

    tif = prj.rasters / "treecover" / "ptc_sen_slope.tif"
    if tif.exists():
        a = read_raster(tif)
        pth = prj.rasters / "treecover" / "ptc_mk_p.tif"
        p = read_raster(pth) if pth.exists() else None
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
        lim = np.nanpercentile(np.abs(a), 98) if np.isfinite(a).any() else 1
        im = axs[0].imshow(a, cmap="RdBu", vmin=-lim, vmax=lim)
        fig.colorbar(im, ax=axs[0], label="п.п./год")
        axs[0].set_title("Наклон Сенна PTC", fontsize=9)
        if p is not None:
            im2 = axs[1].imshow(np.where(np.isfinite(p), p < cfg["statistics"]
                                         ["alpha"], np.nan), cmap="Greys",
                                vmin=0, vmax=1)
            axs[1].set_title(f"p < {cfg['statistics']['alpha']} "
                             "(без поправки)", fontsize=9)
        for ax in axs:
            ax.set_axis_off()
        _savefig(prj, fig, "fig08_ptc_trend_map", None, caps,
                 "Пиксельный тренд доли древесного покрытия (слева) и "
                 "отдельно — статистическая поддержка (справа).",
                 "Per-pixel tree cover trend (left) and statistical support "
                 "shown separately (right).")
    return caps


# =============================================================================
# 22. Пакет для исследователя и ИИ
# =============================================================================
DATA_DICTIONARY = [
    ("annual_class_areas", "Ежегодные площади классов IGBP (raw и mode3)",
     "area_ha, share_of_unit_pct, share_of_woody_pct, valid_area_ha"),
    ("period_class_areas", "Пятилетние сводки площадей", "mean_area_ha"),
    ("class_transitions", "Кандидатные переходы меток MCD12Q1",
     "from_class, to_class, first_new_class_year, status, ptc_consistency"),
    ("transition_matrix_persistent", "Матрица кандидатных устойчивых переходов",
     "area_ha"),
    ("label_switches_annual", "Исходные межгодовые переключения меток",
     "area_ha"),
    ("treecover_timeseries", "Ряды MOD44B по единицам", "mean_ptc_pct, ..."),
    ("treecover_events", "События снижения PTC и восстановление",
     "event_year, depth_pp, recovery_status, origin"),
    ("fragmentation_annual", "Метрики фрагментации по окнам и годам",
     ", ".join(f"{k} [{v}]" for k, v in FRAG_UNITS.items())),
    ("functional_annual", "Сезонные функциональные показатели по стратам",
     "mean, median, n_valid, coverage, effective_resolution_m"),
    ("functional_events", "Функциональные спады и восстановление", ""),
    ("fire_annual", "Зарегистрированные выгорания", "burned_prefire_forest_ha"),
    ("fire_response", "Отклик EVI после пожаров", "median_relative_evi_drop"),
    ("trends_*", "Тренды: наклон Сенна, MK p, q (BH), n, статус",
     "slope, ci_low, ci_high, p_value, q_value, n"),
    ("associations", "Статистические связи (не причинные)", "r, p, q, lag"),
    ("panel_window_year", "Панель окно × год для самостоятельного анализа", ""),
    ("vulnerability_components", "Исходные и нормированные компоненты", ""),
    ("priority_indices", "Интегральные относительные индексы", "P_SF, P_fire"),
    ("sensitivity_results", "Сценарии чувствительности", ""),
    ("quality_annual_layers", "Полнота и качество годовых слоёв", ""),
]

AI_INSTRUCTIONS_RU = """# Инструкции для ИИ-анализа пакета ForestMODIS

Используйте только данные этого пакета. Каждое числовое утверждение
сопровождайте `result_id` или именем таблицы и фильтром строки
(см. `claims_registry.csv`).

Запрещено:
1. Придумывать отсутствующие измерения или годы.
2. Превращать переключение класса MCD12Q1 в доказанную смену сообщества;
   переходы — «кандидатные изменения меток».
3. Называть относительный индекс вероятностью гибели леса.
4. Делать причинные выводы из корреляций (`associations` — статистические
   связи).
5. Игнорировать NoData, неполные периоды (2023–2024) и статусы
   `insufficient_*`.
6. Считать LST, перенесённую на 500 м, более детальной, чем 1 км.
7. Подменять отсутствие зарегистрированного пожара отсутствием пожарного
   воздействия.
8. Выдавать согласованность сценариев чувствительности за точность.
9. Трактовать рост ED/PD без учёта PLAND как доказательство ухудшения.
10. Сравнивать данные MCD12Q1 после 2020 г. без оговорки о повышенной
    неопределённости.

Если независимая проверка точности не выполнена (см. START_HERE.md),
это нужно явно указать в тексте.
"""


def build_claims(prj: Project) -> pd.DataFrame:
    rows = []

    def add(rid, tmpl, value, units, period, unit, table, flt, unc="",
            q="", lim=""):
        rows.append(dict(claim_id=f"C{len(rows) + 1:04d}", result_id=rid,
                         statement_template=tmpl, value=value, units=units,
                         comparison_period=period, spatial_unit=unit,
                         source_table=table, source_row_filter=flt,
                         uncertainty=unc, quality_flag=q,
                         interpretation_limit=lim))

    for name, lim in (("trends_class_area", "изменение меток MCD12Q1, не "
                       "доказанная смена сообществ"),
                      ("trends_treecover", "MOD44B, мезомасштаб"),
                      ("trends_fragmentation", "конфигурация при 500 м; "
                       "читать вместе с PLAND"),
                      ("trends_functional", "дистанционные показатели "
                       "состояния, не диагноз")):
        t = load_table(prj, name)
        if t is None or not len(t):
            continue
        if "unit_type" in t:
            t = t[t.unit_type.isin(["aoi"])] if (t.unit_type == "aoi").any() \
                else t
        for _, r in t.head(400).iterrows():
            label = " / ".join(str(r[c]) for c in t.columns if c in (
                "cover_class", "variable", "metric", "mask", "stratum",
                "unit_id", "value_col") and pd.notna(r[c]))
            add(r.get("result_id"), f"Тренд {label}: {{value}} {{units}}/год",
                r.get("slope"), r.get("units", "") or "",
                f"{r.get('year_start', '')}-{r.get('year_end', '')}",
                r.get("unit_id", "AOI"), name, f"result_id=={r.get('result_id')}",
                f"CI [{r.get('ci_low')}, {r.get('ci_high')}], p={r.get('p_value')}"
                f", q={r.get('q_value')}", r.get("status", ""), lim)
    return pd.DataFrame(rows)


def methods_text(prj: Project, lang="ru") -> str:
    cfg = prj.cfg
    st = {}
    for s in STAGES:
        p = prj.stage_state_path(s)
        if p.exists():
            st[s] = json.loads(p.read_text())
    done = [s for s, v in st.items() if v.get("status") == "done"]
    p = cfg["period"]
    fc = cfg["fragmentation"]
    tc = cfg["treecover"]
    lc = cfg["landcover"]
    g = prj.grid()
    prods = ", ".join(f"{k}.061 (https://doi.org/10.5067/MODIS/{k}.061)"
                      for k in cfg["data"]["products"])
    if lang == "ru":
        return f"""# Методы (сформировано автоматически по фактическому запуску)

Запуск: `{prj.run_id}`; выполненные этапы: {', '.join(done) or 'нет'}.

**Данные.** Использованы готовые продукты MODIS Collection 6.1: {prods}.
Период {p['start_year']}–{p['end_year']} гг.; базовый период сравнения
{p['comparison_baseline'][0]}–{p['comparison_baseline'][1]} гг.; базовый
период аномалий {p['anomaly_baseline'][0]}–{p['anomaly_baseline'][1]} гг.
Вегетационный сезон {cfg['seasons']['growing'][0]}…{cfg['seasons']['growing'][1]},
летний {cfg['seasons']['summer'][0]}…{cfg['seasons']['summer'][1]}.

**Сетка.** Все слои приведены к сетке {g.res:.0f} м в проекции `{g.crs}`
({g.width}×{g.height} ячеек). Непрерывные данные 250 м агрегированы
усреднением валидных значений (resampling average), категориальные —
ближайшим соседом, QA декодирован до пространственных преобразований.
LST сохраняет эффективное разрешение 1 км.

**Качество.** Профиль QA `{cfg['quality']['profile']}`; минимальное
сезонное покрытие {cfg['quality']['min_season_coverage']}; минимальная доля
валидных ячеек окна {cfg['quality']['min_window_valid_fraction']}.
Интерполяция пропусков: {'да' if cfg['quality']['interpolate'] else 'нет'}.

**Типы покрова.** Классы IGBP {lc['classes']} (сомкнутые {lc['closed_classes']},
разреженные {lc['sparse_classes']}). Кандидатный переход регистрировался,
если исходный класс наблюдался ≥{lc['persistence_before']} валидных лет до
и новый ≥{lc['persistence_after']} лет после смены; согласованность с
MOD44B оценивалась при пороге {lc['ptc_consistency_delta_pp']} п.п.

**Древесное покрытие.** MOD44B Percent Tree Cover; рабочий порог маски
{tc['primary_threshold_pct']}%, сценарии {tc['sensitivity_thresholds_pct']}.
Событие снижения: падение ≥ max({tc['decline_min_pp']} п.п.;
{tc['decline_k_mad']}·σ_MAD) относительно медианы {tc['pre_event_years']}
предшествующих лет, сохраняющееся ≥{tc['decline_persistence_years']} лет.

**Фрагментация.** Окна {fc['main_window_km']}×{fc['main_window_km']} км
(сценарии {fc['sensitivity_windows_km']} км), {fc['neighborhood']}-связное
соседство, глубина края {fc['edge_depth_cells']} ячейка(и); NoData не
считается фоном; внешняя граница в ED: {fc['count_outer_boundary_in_ed']}.
Метрики: {', '.join(FRAG_UNITS)} (определения по FRAGSTATS 4.2;
MESH = Σa²/A).

**Функциональное состояние.** Переменные {cfg['functional']['variables']};
стандартизированные и робастные аномалии; неблагоприятный год — ниже
{cfg['functional']['adverse_percentile']}-го процентиля собственного ряда
или z ≤ {cfg['functional']['decline_z']}; устойчивое снижение
≥{cfg['functional']['sustained_years']} лет; восстановление — возврат в
пределы {cfg['functional']['recovery_sd']} SD от базовой медианы. Роль LST:
`{cfg['lst']['role']}`.

**Пожары.** MCD64A1 с учётом QA; предпожарная маска
`{cfg['fire']['prefire_mask']}`; восстановление EVI в пределах
{cfg['fire']['recovery_tolerance']:.0%} от предпожарного уровня за
≤{cfg['fire']['recovery_years_max']} лет.

**Статистика.** Наклон Сенна, критерий Манна–Кендалла (модификация
Hamed–Rao при значимой автокорреляции lag-1 для региональных рядов),
поправка Benjamini–Hochberg; минимум {cfg['statistics']['min_valid_years']}
валидных лет; α = {cfg['statistics']['alpha']}.

**Относительная уязвимость** (статус `{cfg['vulnerability']['status']}`,
сценарий `{cfg['vulnerability']['scenario_id']}`). Компоненты S_cover,
S_config, F, H, E нормированы процентильными рангами с сохранением нуля
внутри страт {cfg['vulnerability']['stratify_by']}; P_SF = √(S·F),
P_fire = (H·E·P_SF)^(1/3). Индексы — относительная приоритетность окон,
не вероятность гибели леса.

**Независимая проверка точности:** {'выполнена' if cfg['inputs'].get('validation') else 'не выполнена'}.
"""
    return f"""# Methods (auto-generated from the actual run)

Run `{prj.run_id}`; completed stages: {', '.join(done) or 'none'}.
MODIS C6.1 products: {prods}; period {p['start_year']}–{p['end_year']};
grid {g.res:.0f} m, CRS `{g.crs}`; QA profile `{cfg['quality']['profile']}`.
IGBP classes {lc['classes']}; candidate transitions require
{lc['persistence_before']}/{lc['persistence_after']} persistent valid years.
MOD44B threshold {tc['primary_threshold_pct']}% (scenarios
{tc['sensitivity_thresholds_pct']}). Fragmentation in
{fc['main_window_km']} km windows, {fc['neighborhood']}-neighbourhood, edge depth
{fc['edge_depth_cells']} cell(s); FRAGSTATS 4.2 definitions. Functional
variables {cfg['functional']['variables']}; LST role `{cfg['lst']['role']}`.
Trends: Sen slope, Mann–Kendall (Hamed–Rao when lag-1 autocorrelation is
significant), Benjamini–Hochberg. Relative vulnerability (status
`{cfg['vulnerability']['status']}`): zero-preserving within-stratum ranks,
P_SF = sqrt(S·F), P_fire = (H·E·P_SF)^(1/3) — relative priority, not
probability of forest loss. Independent accuracy assessment:
{'performed' if cfg['inputs'].get('validation') else 'not performed'}.
"""


def stage_export(prj: Project):
    cfg = prj.cfg
    pkg = prj.pkg
    caps = make_figures(prj) if cfg["reporting"]["figures"] else []
    # каталоги
    catalog = []
    for p in sorted(prj.tables.rglob("*.csv")):
        try:
            n = sum(1 for _ in open(p, encoding="utf-8")) - 1
        except Exception:
            n = None
        catalog.append(dict(kind="table", path=str(p.relative_to(prj.out)),
                            rows=n, sha256=sha256_file(p)))
    for p in sorted(prj.rasters.rglob("*.tif")):
        catalog.append(dict(kind="raster", path=str(p.relative_to(prj.out)),
                            rows=None, sha256=sha256_file(p)))
    for p in sorted(prj.vectors.glob("*.gpkg")):
        catalog.append(dict(kind="vector", path=str(p.relative_to(prj.out)),
                            rows=None, sha256=sha256_file(p)))
    for p in sorted(prj.figures.glob("*.png")):
        catalog.append(dict(kind="figure", path=str(p.relative_to(prj.out)),
                            rows=None, sha256=sha256_file(p)))
    pd.DataFrame(catalog).to_csv(pkg / "results_catalog.csv", index=False,
                                 encoding="utf-8")
    pd.DataFrame(DATA_DICTIONARY, columns=["table", "description_ru",
                                           "key_fields"]).to_csv(
        pkg / "data_dictionary.csv", index=False, encoding="utf-8")
    # сводки
    a = load_table(prj, "annual_class_areas")
    if a is not None:
        a[(a.unit_type == "aoi")].to_csv(pkg / "annual_summary.csv",
                                         index=False, encoding="utf-8")
    pc = load_table(prj, "period_class_areas")
    if pc is not None:
        pc.to_csv(pkg / "period_summary.csv", index=False, encoding="utf-8")
    trs = []
    for n in ("trends_class_area", "trends_treecover", "trends_fragmentation",
              "trends_functional"):
        t = load_table(prj, n)
        if t is not None and len(t):
            t = t.copy()
            t.insert(0, "source_table", n)
            if "unit_type" in t:
                t = t[t.unit_type != "window"]
            trs.append(t)
    if trs:
        pd.concat(trs, ignore_index=True).to_csv(pkg / "trends_summary.csv",
                                                 index=False, encoding="utf-8")
    ss = load_table(prj, "sensitivity_summary")
    if ss is not None:
        ss.to_csv(pkg / "sensitivity_summary.csv", index=False,
                  encoding="utf-8")
    claims = build_claims(prj)
    claims.to_csv(pkg / "claims_registry.csv", index=False, encoding="utf-8")
    key = claims[claims.quality_flag.astype(str).isin(["ok", ""])].head(60) \
        if len(claims) else claims
    key.to_csv(pkg / "key_results.csv", index=False, encoding="utf-8")
    (pkg / "methods_ru.md").write_text(methods_text(prj, "ru"),
                                       encoding="utf-8")
    (pkg / "methods_en.md").write_text(methods_text(prj, "en"),
                                       encoding="utf-8")
    (pkg / "ai_analysis_instructions.md").write_text(AI_INSTRUCTIONS_RU,
                                                     encoding="utf-8")
    (pkg / "figure_captions_ru.md").write_text(
        "# Подписи к рисункам\n\n" + "\n".join(
            f"**{c['figure']}.** {c['caption_ru']} Данные: `{c['data_file']}`\n"
            for c in caps), encoding="utf-8")
    (pkg / "figure_captions_en.md").write_text(
        "# Figure captions\n\n" + "\n".join(
            f"**{c['figure']}.** {c['caption_en']}\n" for c in caps),
        encoding="utf-8")
    # качество и ограничения
    q = load_table(prj, "quality_annual_layers")
    miss = load_table(prj, "missing_product_years")
    lines = ["# Качество данных и ограничения", ""]
    if miss is not None and len(miss):
        lines += ["## Отсутствующие продукт-годы", "",
                  miss.drop(columns=["run_id", "result_id"], errors="ignore")
                  .to_markdown(index=False) if hasattr(miss, "to_markdown")
                  else miss.to_string(index=False), ""]
    if q is not None and len(q) and "valid_fraction" in q:
        lo = q[q.valid_fraction < 0.5]
        lines += [f"Годовых слоёв с долей валидных ячеек < 0.5: {len(lo)} из "
                  f"{len(q)}.", ""]
    lines += [
        "## Методические ограничения", "",
        "- Разрешение 250–1000 м: мелкие вырубки, дороги, просеки не "
        "выявляются; фрагментация мезомасштабная.",
        "- MCD12Q1 не рекомендован разработчиками для прямого картирования "
        "изменений; переходы — кандидатные изменения меток.",
        "- После 2020 г. неопределённость MCD12Q1 повышена.",
        "- MOD44B и IGBP используют разные определения древесной "
        "растительности; значения не взаимозаменяемы.",
        "- Оптические индексы насыщаются в сомкнутых лесах.",
        "- MCD64A1 недоучитывает небольшие выгорания.",
        "- Индексы уязвимости — относительная приоритетность, не "
        "вероятность.",
        "- Независимая оценка точности: " + (
            "выполнена" if cfg["inputs"].get("validation") else
            "НЕ выполнена."), ""]
    (pkg / "quality_and_limitations.md").write_text("\n".join(lines),
                                                    encoding="utf-8")
    # START_HERE
    st = {}
    for s in STAGES:
        p = prj.stage_state_path(s)
        if p.exists():
            st[s] = json.loads(p.read_text())
    stage_lines = "\n".join(f"| {s} | {v.get('status')} | {v.get('seconds')} | "
                            f"{json.dumps(v.get('info', {}), ensure_ascii=False)} |"
                            for s, v in st.items())
    p = cfg["period"]
    (pkg / "START_HERE.md").write_text(f"""# ForestMODIS — пакет результатов

**Проект:** {cfg['project']['name']}  
**Запуск:** `{prj.run_id}`  
**Территория:** `{cfg['inputs']['aoi']}`  
**Период:** {p['start_year']}–{p['end_year']}  
**Версия программы:** {__version__}

## Что исследовано
Ежегодная динамика древесных классов MCD12Q1 и доли древесного покрытия
MOD44B, мезомасштабная фрагментация, функциональное состояние (EVI, LAI,
FPAR, GPP, ET/PET), тепловые условия (LST), зарегистрированные выгорания
(MCD64A1) и относительная уязвимость по аналитическим окнам.

## Статус этапов
| Этап | Статус | Секунд | Сводка |
|---|---|---|---|
{stage_lines}

## Невыполненные проверки
- Независимая оценка точности: {'выполнена' if cfg['inputs'].get('validation') else 'не выполнена'}.
- Формулы интегральных индексов имеют статус `{cfg['vulnerability']['status']}`.

## Где что лежит
- `methods_ru.md`, `methods_en.md` — описание фактически выполненных процедур.
- `trends_summary.csv`, `annual_summary.csv`, `period_summary.csv` — ключевые ряды.
- `claims_registry.csv` — числовые утверждения с адресом исходной строки.
- `quality_and_limitations.md` — качество и ограничения.
- `ai_analysis_instructions.md` — правила для ИИ-помощника.
- Полные таблицы: `../tables/`, растры: `../rasters/`, окна: `../vectors/`,
  рисунки: `../figures/`.
""", encoding="utf-8")
    (pkg / "study_design.md").write_text(
        "# Дизайн исследования\n\nСм. methods_ru.md. Основная единица "
        "фрагментации и уязвимости — регулярное окно "
        f"{cfg['fragmentation']['main_window_km']} км; динамика считается "
        "ежегодно, представление — по пятилетним опорным годам "
        f"{cfg['reporting']['reference_years']} и периодам "
        f"{cfg['reporting']['periods']}.\n", encoding="utf-8")
    # манифест
    reg = prj.inv / "granule_registry.csv"
    manifest = dict(run_id=prj.run_id, created=dt.datetime.now().isoformat(),
                    versions=env_versions(), config=public_config(cfg),
                    stages=st,
                    granule_registry=str(reg) if reg.exists() else None,
                    granule_registry_sha256=sha256_file(reg)
                    if reg.exists() else None,
                    grid=dict(crs=prj.grid().crs, res=prj.grid().res,
                              width=prj.grid().width,
                              height=prj.grid().height,
                              transform=list(prj.grid().transform)[:6]),
                    files=catalog)
    (pkg / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                  indent=1, default=str),
                                       encoding="utf-8")
    return {"figures": len(caps), "catalog_items": len(catalog)}


# =============================================================================
# 23. Исполнение этапов
# =============================================================================
STAGE_FUNCS: Dict[str, Callable[[Project], Any]] = {}


def _register_stages():
    STAGE_FUNCS.update({
        "prepare": stage_prepare, "landcover": stage_landcover,
        "treecover": stage_treecover, "fragmentation": stage_fragmentation,
        "functional": stage_functional, "fire": stage_fire,
        "associations": stage_associations,
        "vulnerability": stage_vulnerability, "export": stage_export})


def run_stages(prj: Project, from_stage=None, to_stage=None, resume=True):
    _register_stages()
    if prj.cfg["data"].get("backend") == "gee":
        STAGE_FUNCS["prepare"] = stage_prepare_gee
    i0 = STAGES.index(from_stage) if from_stage else 0
    i1 = STAGES.index(to_stage) if to_stage else len(STAGES) - 1
    (prj.logs / f"config_{prj.run_id}.json").write_text(
        json.dumps(public_config(prj.cfg), ensure_ascii=False, indent=1,
                   default=str), encoding="utf-8")
    (prj.logs / f"versions_{prj.run_id}.json").write_text(
        json.dumps(env_versions(), indent=1), encoding="utf-8")
    for s in STAGES[i0:i1 + 1]:
        if resume and s != "export" and stage_done(prj, s):
            LOG.info(f"[{s}] уже выполнен с теми же параметрами — пропуск")
            continue
        if s == "vulnerability" and prj.cfg["vulnerability"]["status"] in (
                None, "", "undefined"):
            LOG.warning("[vulnerability] status=undefined: считаются только "
                        "компоненты")
        LOG.info(f"[{s}] старт")
        t0 = time.time()
        try:
            info = STAGE_FUNCS[s](prj)
        except Exception as e:
            mark_stage(prj, s, "failed", time.time() - t0,
                       {"error": repr(e)})
            LOG.error(f"[{s}] ошибка: {e}")
            LOG.debug(traceback.format_exc())
            raise
        mark_stage(prj, s, "done", time.time() - t0, info)
        LOG.info(f"[{s}] готово за {time.time() - t0:.1f} с: {info}")


# =============================================================================
# 24. Интерфейс командной строки
# =============================================================================
EXAMPLE_CONFIG = """# ForestMODIS — конфигурация проекта. Все пути — относительно этого файла.
project:
  name: central_russia_modis
  root: .
  random_seed: 42

inputs:
  aoi: inputs/aoi.gpkg          # null → окно выбора файла
  zones: null                   # природные зоны (необязательно)
  zones_field: null
  validation: null              # независимые эталонные данные
  buffer_km: 0

period:
  start_year: 2003
  end_year: 2024
  comparison_baseline: [2003, 2005]
  anomaly_baseline: [2003, 2012]

seasons:
  growing: ["05-01", "09-30"]
  summer: ["06-01", "08-31"]

grid:
  resolution_m: 500
  crs: auto_albers              # или строка PROJ / EPSG

data:
  collection_version: "061"
  mode: online                  # online | offline
  backend: earthdata_https      # earthdata_https | gee (Google Earth Engine)
  fail_on_missing_required: true
  download_workers: 4

quality:
  profile: standard             # strict | standard | relaxed
  min_season_coverage: 0.5
  min_window_valid_fraction: 0.95
  interpolate: false

landcover:
  classes: [1, 3, 4, 5, 8, 9]
  persistence_before: 3
  persistence_after: 3

treecover:
  primary_threshold_pct: 30
  sensitivity_thresholds_pct: [10, 20, 30, 40, 50, 60]

fragmentation:
  main_window_km: 25
  sensitivity_windows_km: [10, 50]
  neighborhood: 8
  edge_depth_cells: 1
  count_outer_boundary_in_ed: false

statistics:
  min_valid_years: 15
  alpha: 0.05

lst:
  role: explanatory             # explanatory | stress_component

vulnerability:
  status: provisional           # undefined → интегральные индексы не считаются
  scenario_id: provisional_v1

reporting:
  reference_years: [2004, 2009, 2014, 2019, 2024]

execution:
  workers: 4
  block_rows: 256
"""


def cmd_init(args):
    root = Path(args.project).resolve()
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    cp = root / "config.yaml"
    if cp.exists() and not args.force:
        print(f"{cp} уже существует (используйте --force)")
    else:
        cp.write_text(EXAMPLE_CONFIG, encoding="utf-8")
        print(f"Создан {cp}. Положите полигон территории в "
              f"{root / 'inputs' / 'aoi.gpkg'} или укажите путь в config.yaml.")


def _project(args) -> Project:
    cfg = load_config(args.config)
    if getattr(args, "offline", False):
        cfg["data"]["mode"] = "offline"
    for w in validate_config(cfg):
        print("ПРЕДУПРЕЖДЕНИЕ:", w)
    prj = Project(cfg)
    setup_logging(prj, getattr(args, "verbose", False))
    return prj


def cmd_interactive(args=None):
    """Пошаговый режим: окна выбора файлов/папок и меню действий.
    Запускается командой `interactive` или при запуске без аргументов
    (в том числе из JupyterLab: `%run forestmodis.py`)."""
    print("ForestMODIS — пошаговый режим")
    cfgp = pick_files_dialog("Выберите config.yaml проекта (Отмена — создать "
                             "новый проект)", False, [("YAML", "*.yaml *.yml")])
    if cfgp:
        cfgp = cfgp[0] if isinstance(cfgp, (list, tuple)) else cfgp
    else:
        d = pick_folder_dialog("Выберите или создайте папку нового проекта")
        if not d:
            d = _ask("Папка нового проекта [./forest_project]: ",
                     "./forest_project")
        cmd_init(argparse.Namespace(project=d, force=False))
        cfgp = str(Path(d).resolve() / "config.yaml")
        import yaml
        with open(cfgp, encoding="utf-8") as f:
            c = yaml.safe_load(f)
        aoi = pick_files_dialog("Выберите полигон территории исследования",
                                False, [("Векторные", "*.gpkg *.geojson *.json "
                                         "*.shp"), ("Все файлы", "*.*")])
        if aoi:
            c["inputs"]["aoi"] = str(Path(aoi[0]).resolve())
        else:
            print("Полигон не выбран: положите его в inputs/aoi.gpkg проекта.")
        y0 = _ask("Начальный год [2003]: ", "2003")
        y1 = _ask("Конечный год [2024]: ", "2024")
        c["period"]["start_year"], c["period"]["end_year"] = int(y0), int(y1)
        # базовые периоды не должны выходить за пределы периода
        a, b = int(y0), int(y1)
        c["period"]["comparison_baseline"] = [a, min(a + 2, b)]
        c["period"]["anomaly_baseline"] = [a, min(a + 9, b)]
        with open(cfgp, "w", encoding="utf-8") as f:
            yaml.safe_dump(c, f, allow_unicode=True, sort_keys=False)
        print(f"Конфигурация сохранена: {cfgp}")
    try:
        prj = _project(argparse.Namespace(config=cfgp, verbose=False,
                                          offline=False))
        g = prj.grid()
        LOG.info(f"Сетка {g.width}×{g.height} ячеек по {g.res:.0f} м; тайлы "
                 f"MODIS: {', '.join(aoi_sinusoidal_tiles(prj.cfg))}")
    except ConfigError as e:
        print("ОШИБКА КОНФИГУРАЦИИ:", e)
        return 2
    online = prj.cfg["data"]["mode"] == "online"
    menu = {"1": "inventory + plan (поиск гранул и оценка объёма, без загрузки)",
            "2": "download (скачивание, нужен вход NASA Earthdata)",
            "3": "run (все этапы расчёта, с продолжением)",
            "4": "sensitivity (анализ чувствительности)",
            "5": "export (пакет для анализа)",
            "6": "всё подряд: inventory → download → run → sensitivity → "
                 "export",
            "0": "выход"}
    while True:
        print("\nДействия:")
        for k, v in menu.items():
            print(f"  {k} — {v}")
        ch = _ask("Выбор [1]: ", "1")
        if ch == "0":
            break
        is_gee = prj.cfg["data"].get("backend") == "gee"
        if ch in ("1", "6"):
            if is_gee:
                LOG.info("backend=gee: поиск гранул не нужен (данные "
                         "получает Earth Engine на этапе prepare)")
            else:
                run_inventory(prj, online=online)
                run_plan(prj)
        if ch in ("2", "6") and online and not is_gee:
            if not (prj.inv / "granule_catalog.csv").exists():
                run_inventory(prj, online=True)
            run_download(prj)
        if ch in ("3", "6"):
            run_stages(prj, resume=True)
        if ch in ("4", "6"):
            LOG.info(f"Чувствительность: {run_sensitivity(prj)}")
        if ch in ("5", "6"):
            _register_stages()
            LOG.info(f"Экспорт: {stage_export(prj)} → {prj.pkg}")
        if ch == "6":
            break
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="forestmodis",
        description="Динамика и относительная уязвимость лесного покрова по "
                    "продуктам MODIS C6.1")
    ap.add_argument("--version", action="version", version=__version__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("init", help="создать проект с примером конфигурации")
    p.add_argument("--project", required=True)
    p.add_argument("--force", action="store_true")
    for name, hlp in (("validate", "проверить конфигурацию и AOI"),
                      ("inventory", "поиск гранул в NASA CMR без загрузки"),
                      ("plan", "оценка объёмов и ресурсов"),
                      ("download", "скачивание гранул (Earthdata Login)"),
                      ("run", "выполнить этапы анализа"),
                      ("sensitivity", "анализ чувствительности"),
                      ("export", "экспорт таблиц, рисунков и пакета для ИИ"),
                      ("gee-info", "проверить доступ к Google Earth Engine")):
        p = sp.add_parser(name, help=hlp)
        p.add_argument("--config", default=None,
                       help="config.yaml (если не указан — окно выбора)")
        p.add_argument("--verbose", action="store_true")
        p.add_argument("--offline", action="store_true",
                       help="работать только с локальным архивом")
        if name == "run":
            p.add_argument("--from-stage", choices=STAGES)
            p.add_argument("--to-stage", choices=STAGES)
            p.add_argument("--resume", action="store_true", default=True)
            p.add_argument("--no-resume", dest="resume", action="store_false")
            p.add_argument("--with-download", action="store_true",
                           help="перед расчётом выполнить inventory+download")
        if name == "export":
            p.add_argument("--profile", default="ai", choices=["ai", "full"])
    sp.add_parser("interactive", help="пошаговый режим с окнами выбора файлов")
    if argv is None:
        argv = sys.argv[1:]
        if in_notebook():
            # ядро Jupyter передаёт «-f kernel-....json»: это не наши аргументы
            argv = [a for i, a in enumerate(argv)
                    if a != "-f" and not (i and argv[i - 1] == "-f")
                    and not a.endswith(".json")]
    argv = list(argv)
    if not argv:
        # запуск без аргументов (двойной щелчок, Jupyter, IDE) → пошаговый режим
        return cmd_interactive(None)
    args = ap.parse_args(argv)

    if args.cmd == "init":
        return cmd_init(args)
    if args.cmd == "interactive":
        return cmd_interactive(args)
    if args.config is None:
        sel = pick_files_dialog("Выберите config.yaml", False,
                                [("YAML", "*.yaml *.yml")])
        if not sel:
            ap.error("не указан --config")
        args.config = sel[0] if isinstance(sel, (list, tuple)) else sel
    try:
        prj = _project(args)
    except ConfigError as e:
        print("ОШИБКА КОНФИГУРАЦИИ:", e)
        return 2
    online = prj.cfg["data"]["mode"] == "online"
    is_gee = prj.cfg["data"].get("backend") == "gee"
    if args.cmd == "validate":
        g = prj.grid()
        tiles = aoi_sinusoidal_tiles(prj.cfg)
        LOG.info(f"AOI корректна; сетка {g.width}×{g.height} ячеек по "
                 f"{g.res:.0f} м, CRS: {g.crs}; тайлы MODIS: {', '.join(tiles)}")
        return 0
    if args.cmd == "gee-info":
        return cmd_gee_info(prj)
    if is_gee and args.cmd in ("inventory", "plan", "download"):
        LOG.info("backend=gee: гранулы NASA не нужны — этапы inventory/plan/"
                 "download пропущены. Запускайте run (или gee-info для "
                 "проверки доступа).")
        return 0
    if args.cmd == "inventory":
        run_inventory(prj, online=online)
        return 0
    if args.cmd == "plan":
        if not (prj.inv / "granule_catalog.csv").exists():
            run_inventory(prj, online=online)
        run_plan(prj)
        return 0
    if args.cmd == "download":
        if not (prj.inv / "granule_catalog.csv").exists():
            run_inventory(prj, online=True)
        run_download(prj)
        return 0
    if args.cmd == "run":
        if args.with_download and online and not is_gee:
            run_inventory(prj, online=True)
            run_download(prj)
        run_stages(prj, args.from_stage, args.to_stage, args.resume)
        return 0
    if args.cmd == "sensitivity":
        t0 = time.time()
        info = run_sensitivity(prj)
        LOG.info(f"Чувствительность: {info}, {time.time() - t0:.0f} с")
        return 0
    if args.cmd == "export":
        _register_stages()
        t0 = time.time()
        info = stage_export(prj)
        mark_stage(prj, "export", "done", time.time() - t0, info)
        LOG.info(f"Экспорт: {info} → {prj.pkg}")
        return 0


if __name__ == "__main__":
    if in_notebook():
        # в Jupyter не вызываем sys.exit, чтобы не получать SystemExit
        try:
            main()
        except SystemExit:
            pass
    else:
        sys.exit(main() or 0)
