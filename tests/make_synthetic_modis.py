#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор СИНТЕТИЧЕСКИХ гранул MODIS (HDF4) для проверки алгоритмов ForestMODIS
без доступа к Earthdata.

ВНИМАНИЕ: это тестовые данные. Они имеют реальные имена файлов, названия SDS,
типы, коды QA, размеры тайлов и геопривязку синусоидальной сетки MODIS, но
значения смоделированы и НЕ МОГУТ использоваться в исследовательских выводах.
В атрибутах каждого файла записано SYNTHETIC_TEST_DATA=1.

Заложенные «истинные» сигналы (для проверки, что конвейер их находит):
  * сокращение древесного покрытия (−35 п.п.) в блоке «вырубки» с 2006 г.;
  * устойчивый переход меток 1 → 4 в блоке «смены пород» с 2006 г.;
  * случайное мерцание меток MCD12Q1 (≈3 % ячеек в год) — шум;
  * засуха 2005 г.: EVI −0.08, GPP −25 %, ET −30 %, LST +3 K;
  * пожар в июле 2004 г. в отдельном блоке с последующим падением EVI;
  * облачные пропуски ≈10 % в 16-дневных композитах.

Запуск:
    python tests/make_synthetic_modis.py --out <project>/data/raw \
        --tile h20v03 --years 2003 2008
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from pyhdf.SD import SD, SDC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import forestmodis as fm  # noqa: E402

COARSE = 600  # базовое «ландшафтное» разрешение (≈1.85 км для тайла)

TYPES = {"int16": SDC.INT16, "uint8": SDC.UINT8, "uint16": SDC.UINT16,
         "int8": SDC.INT8}


def smooth_field(rng, n=COARSE, scale=12):
    """Пятнистое поле 0..1 (сумма шумов разных масштабов)."""
    out = np.zeros((n, n))
    for s, w in ((scale * 4, 0.5), (scale, 0.35), (scale // 3 or 1, 0.15)):
        m = n // s + 2
        c = rng.random((m, m))
        up = np.kron(c, np.ones((s, s)))[:n, :n]
        out += w * up
    out = (out - out.min()) / (out.max() - out.min())
    return out


def up(a, npix):
    f = npix // a.shape[0]
    return np.kron(a, np.ones((f, f), dtype=a.dtype))


def tile_lonlat_block(tile, lon0, lon1, lat0, lat1, n=COARSE):
    """Маска ячеек coarse-сетки тайла внутри прямоугольника lon/lat."""
    from pyproj import Transformer
    tr = fm.tile_transform(tile, n)
    cols, rows = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
    x = tr.c + cols * tr.a
    y = tr.f + rows * tr.e
    t = Transformer.from_crs(fm.SINU_WKT, "EPSG:4326", always_xy=True)
    lon, lat = t.transform(x, y)
    return (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)


def write_hdf(path: Path, layers: dict, attrs: dict = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    sd = SD(str(path), SDC.WRITE | SDC.CREATE)
    sd.SYNTHETIC_TEST_DATA = 1
    for name, (arr, a_attrs) in layers.items():
        s = sd.create(name, TYPES[str(arr.dtype)], arr.shape)
        s.setcompress(SDC.COMP_DEFLATE, 1)
        s[:] = arr
        for k, v in (a_attrs or {}).items():
            setattr(s, k, v)
        s.endaccess()
    sd.end()


def fname(prod, year, doy, tile):
    return f"{prod}.A{year}{doy:03d}.{tile}.061.2022001000000.hdf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", default="h20v03")
    ap.add_argument("--years", nargs=2, type=int, default=[2003, 2008])
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out)
    tile = a.tile
    rng = np.random.default_rng(a.seed)
    cfg = fm.load_config(None)

    forest = smooth_field(rng)                      # базовая лесистость
    conif = smooth_field(rng, scale=20)             # доля хвойных
    water = smooth_field(rng, scale=30) > 0.93
    cut = tile_lonlat_block(tile, 38.2, 38.8, 55.2, 55.6)
    swap = tile_lonlat_block(tile, 37.3, 37.8, 56.0, 56.3)
    fire = tile_lonlat_block(tile, 39.2, 39.6, 54.8, 55.0)
    # блок «смены пород» делаем сомкнутым лесом, иначе переход 1→4 не наблюдаем
    forest[swap] = np.maximum(forest[swap], 0.95)
    conif[swap] = 0.2
    years = list(range(a.years[0], a.years[1] + 1))
    nfiles = 0
    for y in years:
        yr = np.random.default_rng(a.seed * 1000 + y)
        ptc = 85 * forest ** 0.8 + yr.normal(0, 2, forest.shape)
        if y >= 2006:
            ptc[cut] -= 35
        if y >= 2004:
            ptc[fire] -= 20 * max(0, 1 - (y - 2004) / 6)
        ptc = np.clip(ptc, 0, 100)
        # --- классы IGBP
        lc = np.full(forest.shape, 12, "uint8")
        lc[ptc >= 10] = 9
        lc[ptc >= 30] = 8
        closed = ptc >= 60
        lc[closed & (conif < 0.4)] = 1
        lc[closed & (conif >= 0.4) & (conif < 0.65)] = 5
        lc[closed & (conif >= 0.65)] = 4
        if y >= 2006:
            lc[swap & closed] = 4
        else:
            lc[swap & closed] = 1
        flick = yr.random(forest.shape) < 0.03
        lc[flick & closed] = 5
        lc[water] = 17
        qc = np.where(yr.random(forest.shape) < 0.02, 1, 0).astype("uint8")
        ptc_raw = np.round(ptc).astype("uint8")
        ptc_raw[water] = 200
        write_hdf(out / "MCD12Q1" / fname("MCD12Q1", y, 1, tile),
                  {"LC_Type1": (up(lc, 2400), {"_FillValue": 255}),
                   "QC": (up(qc, 2400), {}),
                   "LW": (up(np.where(water, 1, 2).astype("uint8"), 2400), {})})
        write_hdf(out / "MOD44B" / fname("MOD44B", y, 65, tile),
                  {"Percent_Tree_Cover": (up(ptc_raw, 4800),
                                          {"_FillValue": 253}),
                   "Quality": (up(np.zeros_like(ptc_raw), 4800), {})})
        nfiles += 2
        drought = 1.0 if y == 2005 else 0.0
        # --- MOD13Q1 / MYD13Q1
        for prod in ("MOD13Q1", "MYD13Q1"):
            for st in fm.needed_composites(cfg, prod, y):
                phase = np.sin(np.pi * (st - 100) / 200)
                evi = (0.15 + 0.45 * ptc / 100) * (0.6 + 0.4 * max(phase, 0))
                evi -= 0.08 * drought
                if y >= 2004:
                    evi[fire] -= 0.15 * max(0, 1 - (y - 2004) / 4)
                evi += yr.normal(0, 0.01, evi.shape)
                evi_raw = np.round(np.clip(evi, -0.2, 1) * 1e4).astype("int16")
                ndvi_raw = np.round(np.clip(evi * 1.6, -0.2, 1) * 1e4).astype(
                    "int16")
                rel = np.where(yr.random(evi.shape) < 0.10, 3, 0).astype("int8")
                cdoy = (st + yr.integers(0, 15, evi.shape)).astype("int16")
                write_hdf(out / prod / fname(prod, y, st, tile), {
                    "250m 16 days NDVI": (up(ndvi_raw, 4800),
                                          {"scale_factor": 10000.0}),
                    "250m 16 days EVI": (up(evi_raw, 4800),
                                         {"scale_factor": 10000.0}),
                    "250m 16 days pixel reliability": (up(rel, 4800), {}),
                    "250m 16 days composite day of the year":
                        (up(cdoy, 4800), {})})
                nfiles += 1
        # --- 8-дневные
        for st in fm.needed_composites(cfg, "MCD15A2H", y):
            phase = max(np.sin(np.pi * (st - 100) / 200), 0)
            lai = (0.3 + 5.0 * ptc / 100) * (0.5 + 0.5 * phase) * \
                (1 - 0.15 * drought)
            fpar = np.clip(0.1 + 0.8 * ptc / 100 * (0.6 + 0.4 * phase), 0, 1)
            qcl = np.where(yr.random(lai.shape) < 0.05, 4 << 5, 0).astype(
                "uint8")
            write_hdf(out / "MCD15A2H" / fname("MCD15A2H", y, st, tile), {
                "Lai_500m": (up(np.round(lai * 10).astype("uint8"), 2400),
                             {"scale_factor": 0.1}),
                "Fpar_500m": (up(np.round(fpar * 100).astype("uint8"), 2400),
                              {"scale_factor": 0.01}),
                "FparLai_QC": (up(qcl, 2400), {})})
            gpp = (0.01 + 0.05 * ptc / 100) * (0.5 + 0.5 * phase) * \
                (1 - 0.25 * drought)
            write_hdf(out / "MOD17A2HGF" / fname("MOD17A2HGF", y, st, tile), {
                "Gpp_500m": (up(np.round(gpp * 1e4).astype("int16"), 2400),
                             {"scale_factor": 1e-4}),
                "Psn_QC_500m": (up(np.zeros(gpp.shape, "uint8"), 2400), {})})
            et = (8 + 18 * ptc / 100) * (0.5 + 0.5 * phase) * \
                (1 - 0.30 * drought)
            pet = np.full(et.shape, 32.0) * (0.6 + 0.4 * phase) * \
                (1 + 0.1 * drought)
            write_hdf(out / "MOD16A2GF" / fname("MOD16A2GF", y, st, tile), {
                "ET_500m": (up(np.round(et * 10).astype("int16"), 2400),
                            {"scale_factor": 0.1}),
                "PET_500m": (up(np.round(pet * 10).astype("int16"), 2400),
                             {"scale_factor": 0.1}),
                "ET_QC_500m": (up(np.zeros(et.shape, "uint8"), 2400), {})})
            for prod, dT in (("MOD11A2", 0.0), ("MYD11A2", 1.5)):
                c = forest[::2, ::2]
                p2 = ptc[::2, ::2]
                lstd = 293 + 6 * (1 - p2 / 100) + 3 * drought + dT + \
                    4 * phase + yr.normal(0, 0.3, c.shape)
                lstn = 283 + 2 * (1 - p2 / 100) + 1 * drought + dT / 2
                qcd = np.where(yr.random(c.shape) < 0.08, 2, 0).astype("uint8")
                write_hdf(out / prod / fname(prod, y, st, tile), {
                    "LST_Day_1km": (up(np.round(lstd / 0.02).astype("uint16"),
                                       1200), {"scale_factor": 0.02}),
                    "LST_Night_1km": (up(np.round(lstn / 0.02).astype(
                        "uint16"), 1200), {"scale_factor": 0.02}),
                    "QC_Day": (up(qcd, 1200), {}),
                    "QC_Night": (up(np.zeros_like(qcd), 1200), {})})
                nfiles += 1
            nfiles += 3
        # --- MCD64A1
        for st in fm.composite_starts("MCD64A1", y):
            bd = np.zeros(forest.shape, "int16")
            bd[water] = -2
            if y == 2004 and st == fm.md_to_doy(2004, "07-01"):
                bd[fire] = 200
            qa = np.full(forest.shape, 3, "uint8")
            qa[water] = 2
            write_hdf(out / "MCD64A1" / fname("MCD64A1", y, st, tile), {
                "Burn Date": (up(bd, 2400), {}),
                "QA": (up(qa, 2400), {})})
            nfiles += 1
        print(f"{y}: готово, файлов всего {nfiles}", flush=True)


if __name__ == "__main__":
    main()
