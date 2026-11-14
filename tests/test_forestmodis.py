# -*- coding: utf-8 -*-
"""Юнит-тесты ключевых алгоритмов ForestMODIS (pytest)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy

import forestmodis as fm  # noqa: E402


# ---------------------------------------------------------------- QA
def test_qa_mod13_profiles():
    rel = np.array([0, 1, 2, 3, -1])
    raw = {"250m 16 days pixel reliability": rel}
    assert fm.qa_mod13(raw, "strict").tolist() == [1, 0, 0, 0, 0]
    assert fm.qa_mod13(raw, "standard").tolist() == [1, 1, 0, 0, 0]
    assert fm.qa_mod13(raw, "relaxed").tolist() == [1, 1, 1, 0, 0]


def test_qa_mcd15_scf_bits():
    # SCF_QC в битах 5–7
    qc = np.array([0, 1 << 5, 2 << 5, 4 << 5])
    ok = fm.qa_mcd15({"FparLai_QC": qc}, "standard")
    assert ok.tolist() == [True, True, False, False]


def test_qa_lst_bits():
    qc = np.array([0, 1, 2, 1 | (2 << 6)])
    assert fm._qa_lst(qc, "strict").tolist() == [True, False, False, False]
    assert fm._qa_lst(qc, "standard").tolist() == [True, True, False, False]


def test_qa_mcd64():
    qa = np.array([3, 1, 2, 3 | 4])
    assert fm.qa_mcd64({"QA": qa}, "standard").tolist() == [1, 0, 0, 1]
    assert fm.qa_mcd64({"QA": qa}, "strict").tolist() == [1, 0, 0, 0]


# ---------------------------------------------------------------- календари
def test_calendars():
    assert fm.composite_starts("MOD13Q1", 2004)[:3] == [1, 17, 33]
    assert fm.composite_starts("MYD13Q1", 2004)[:2] == [9, 25]
    assert len(fm.composite_starts("MCD15A2H", 2003)) == 46
    assert fm.composite_interval("MCD15A2H", 2003, 361) == (361, 365)
    assert fm.composite_interval("MCD64A1", 2004, 61) == (61, 91)  # март, високосный
    cfg = fm.load_config(None)
    nc = fm.needed_composites(cfg, "MCD15A2H", 2003)
    assert min(nc) <= fm.md_to_doy(2003, "05-01") and max(nc) <= 273
    # 2003: 1 мая = DOY 121; композит 113–120 сезон не задевает, 121–128 — полностью
    assert fm.season_overlap(2003, 113, 120, ["05-01", "09-30"]) == (0, 8)
    assert fm.season_overlap(2003, 117, 124, ["05-01", "09-30"]) == (4, 8)


def test_granule_name():
    d = fm.parse_granule_name("MOD13Q1.A2010145.h20v03.061.2021161220540.hdf")
    assert d["prod"] == "MOD13Q1" and d["year"] == 2010 and d["doy"] == 145
    assert d["tile"] == "h20v03"


def test_tile_transform_moscow():
    from pyproj import Transformer
    tr = fm.tile_transform("h20v03", 2400)
    t = Transformer.from_crs("EPSG:4326", fm.SINU_WKT, always_xy=True)
    x, y = t.transform(37.62, 55.75)
    col, row = ~tr @ (x, y)
    assert 0 <= col < 2400 and 0 <= row < 2400


# ---------------------------------------------------------------- статистика
def test_mk_sen_known_trend():
    yrs = np.arange(2003, 2025)
    x = 2.0 * (yrs - 2003) + np.random.default_rng(0).normal(0, 0.5, len(yrs))
    r = fm.mk_sen_series(yrs, x)
    assert r["status"] == "ok"
    assert abs(r["slope"] - 2.0) < 0.1
    assert r["p_value"] < 1e-6
    assert r["slope_lo"] <= r["slope"] <= r["slope_hi"]


def test_mk_no_trend_and_short():
    r = fm.mk_sen_series([1, 2, 3], [1, 2, 3])
    assert r["status"] == "insufficient_years"
    yrs = np.arange(20)
    x = np.random.default_rng(1).normal(size=20)
    assert fm.mk_sen_series(yrs, x)["p_value"] > 0.01


def test_pixel_sen_mk_matches_series():
    rng = np.random.default_rng(2)
    yrs = list(range(2003, 2019))
    st = np.vstack([0.5 * np.arange(16) + rng.normal(0, 1, 16),
                    rng.normal(0, 1, 16)]).T
    st[3, 0] = np.nan
    sl, p, n = fm.pixel_sen_mk(st, yrs, 10)
    r0 = fm.mk_sen_series(yrs, st[:, 0], autocorr=False)
    assert np.isclose(sl[0], r0["slope"], atol=1e-6)
    assert np.isclose(p[0], r0["p_value"], atol=1e-6)
    assert n.tolist() == [15, 16]


def test_bh():
    p = np.array([0.01, 0.04, 0.03, np.nan, 0.5])
    q = fm.bh_qvalues(p)
    assert np.isnan(q[3])
    # n=4: 0.01*4/1=0.04; 0.03*4/2=0.06; 0.04*4/3=0.0533 -> монотонизация -> 0.0533
    assert np.allclose(q[[0, 2, 1, 4]], [0.04, 0.16 / 3, 0.16 / 3, 0.5])


# ---------------------------------------------------------------- переходы
def _prj_stub(h, w, cfg=None):
    cfg = cfg or fm.load_config(None)
    grid = SimpleNamespace(aoi_mask=np.ones((h, w), bool),
                           transform=(500, 0, 0, 0, -500, 0), width=w,
                           res=500.0, cell_ha=25.0)
    return SimpleNamespace(cfg=cfg, grid=lambda: grid,
                           work=Path("/nonexistent_forestmodis_work"))


def test_transitions_persistent_vs_flicker():
    yrs = list(range(2003, 2013))
    T = len(yrs)
    S = np.ones((T, 1, 3), "uint8")
    S[5:, 0, 0] = 4                    # устойчивый 1→4 с 2008
    S[5, 0, 1] = 5                     # одиночное мерцание
    S[8:, 0, 2] = 4                    # 1→4 в 2011 (цензура справа)
    ev, mat = fm.detect_transitions(_prj_stub(1, 3), S, yrs, 3, 3,
                                    [1, 3, 4, 5, 8, 9], ptc_stack=None)
    e0 = ev[ev.col == 0]
    assert len(e0) == 1 and e0.status.iloc[0] == "confirmed"
    assert e0.first_new_class_year.iloc[0] == 2008
    assert e0.confirmation_year.iloc[0] == 2010
    assert (ev.col == 1).sum() == 0
    e2 = ev[ev.col == 2]
    assert e2.status.iloc[0] == "unconfirmed_end_of_series"
    assert bool(e2.right_censored.iloc[0])


def test_transitions_nodata_breaks_run():
    yrs = list(range(2003, 2013))
    S = np.ones((10, 1, 1), "uint8")
    S[5:] = 4
    S[7] = fm.NODATA_U8                # пропуск после перехода
    ev, _ = fm.detect_transitions(_prj_stub(1, 1), S, yrs, 3, 3, [1, 4])
    e = ev[ev.first_new_class_year == 2008]
    assert len(e) == 0 or (e.status != "confirmed").all()


def test_ptc_consistency():
    yrs = list(range(2003, 2013))
    S = np.full((10, 1, 2), 1, "uint8")
    S[5:, 0, :] = 9                    # закрытый → разреженный
    P = np.full((10, 1, 2), 70.0)
    P[5:, 0, 0] = 20.0                 # PTC согласованно упал
    P[5:, 0, 1] = 75.0                 # противоречие
    ev, _ = fm.detect_transitions(_prj_stub(1, 2), S, yrs, 3, 3,
                                  [1, 9], ptc_stack=P)
    c = dict(zip(ev.col, ev.ptc_consistency))
    assert c[0] == "consistent" and c[1] == "contradictory"


# ---------------------------------------------------------------- снижение PTC
def test_event_detection():
    yrs = list(range(2003, 2015))
    st = np.full((12, 3), 60.0)
    st[5:, 0] = 30                      # устойчивое снижение без восстановления
    st[5:7, 1] = 30                     # снижение и восстановление
    st[6, 2] = 30                       # одиночный провал
    thr = np.full(3, 10.0)
    tol = np.full(3, 5.0)
    r = fm.event_detection(st, yrs, thr, 3, 2, tol)
    assert r["event_year"][0] == 2008 and r["recovery_status"][0] == 2
    assert r["event_year"][1] == 2008 and r["recovery_status"][1] == 1
    assert r["recovery_years"][1] == 2
    assert np.isnan(r["event_year"][2]) and r["recovery_status"][2] == 0


# ---------------------------------------------------------------- фрагментация
def test_frag_basic():
    m = np.zeros((10, 10), "int8")
    m[1:4, 1:4] = 1                     # 9 ячеек
    m[6:8, 6:8] = 1                     # 4 ячейки
    r = fm.frag_metrics(m, 500.0, edge_depth=1)
    cell = 25.0
    A = 100 * cell
    assert r["NP"] == 2
    assert np.isclose(r["CA"], 13 * cell)
    assert np.isclose(r["PLAND"], 13.0)
    assert np.isclose(r["MESH"], ((9 * cell) ** 2 + (4 * cell) ** 2) / A)
    assert np.isclose(r["LPI"], 9.0)
    # границы: 12 + 8 сторон по 500 м
    assert np.isclose(r["ED"], 20 * 500 / A)
    assert np.isclose(r["TCA"], 1 * cell)   # у 3×3 остаётся центр, у 2×2 — ничего
    assert np.isfinite(r["ENN_MN"])


def test_frag_nodata_not_edge():
    m = np.full((6, 6), -1, "int8")
    m[1:5, 1:5] = 1
    r = fm.frag_metrics(m, 500.0, edge_depth=1)
    assert r["ED"] == 0                  # NoData не фон
    assert np.isclose(r["CPLAND"], 100.0)
    assert np.isnan(r["ENN_MN"])        # одно пятно
    r2 = fm.frag_metrics(m, 500.0, count_outer=True)
    assert r2["ED"] > 0


def test_frag_enn_value():
    m = np.zeros((5, 9), "int8")
    m[2, 1] = 1
    m[2, 5] = 1
    r = fm.frag_metrics(m, 100.0)
    assert np.isclose(r["ENN_MN"], 400.0)


def test_frag_empty():
    m = np.zeros((5, 5), "int8")
    r = fm.frag_metrics(m, 500.0)
    assert r["NP"] == 0 and r["MESH"] == 0 and r["PLAND"] == 0


# ---------------------------------------------------------------- уязвимость
def test_zero_preserving_rank():
    a = np.array([-1, 0, 2, 4, np.nan, 3])
    g = np.array(["x", "x", "x", "x", "x", "y"])
    r = fm.zero_preserving_rank(a, g)
    assert r[0] == 0 and r[1] == 0
    assert np.isclose(r[2], 0.5) and np.isclose(r[3], 1.0)
    assert np.isnan(r[4]) and r[5] == 1.0


def test_geo_mean():
    g = fm.geo_mean([np.array([0.25, 0, 1, np.nan]),
                     np.array([1.0, 0.5, 1, 1])], [1, 1])
    assert np.isclose(g[0], 0.5) and g[1] == 0 and g[2] == 1 and np.isnan(g[3])


def test_config_validation():
    cfg = fm.load_config(None)
    assert isinstance(fm.validate_config(cfg), list)


# ============================ GEE-бэкенд ==================================
class _FakeEE:
    def __init__(self, *_a, **_k):
        pass

    def __getattr__(self, k):
        if k.startswith("__") and k.endswith("__"):
            raise AttributeError(k)
        return _FakeEE()

    def __call__(self, *_a, **_k):
        return _FakeEE()


def _fake_ee():
    import types
    m = types.ModuleType("ee")
    for n in ("Image", "ImageCollection", "Reducer", "Geometry",
              "Projection", "Algorithms"):
        setattr(m, n, _FakeEE())
    return m


def test_gee_blocks_fit_limit_and_cover_grid():
    import forestmodis as f
    g = f.Grid(crs="EPSG:4326", transform=(500, 0, 0, 0, -500, 0),
               width=5327, height=6930, res=500,
               aoi_mask=np.zeros((6930, 5327), bool), zones=None,
               zone_names={})
    for nb in (1, 6, 12):
        blocks = f.gee_blocks(g, nb, max_mb=40)
        covered = sum(r1 - r0 for r0, r1, _, _ in blocks)
        assert covered == g.height
        assert all(c1 - c0 == g.width for _, _, c0, c1 in blocks)
        for r0, r1, c0, c1 in blocks:
            nbytes = (r1 - r0) * (c1 - c0) * nb * 4
            assert nbytes <= 40 * 1024 * 1024, nbytes


def test_gee_assets_cover_all_products():
    import forestmodis as f
    layers = {"MCD12Q1": ["LC_Type1"], "MOD44B": ["Percent_Tree_Cover"],
              "MOD13Q1": ["250m 16 days EVI", "250m 16 days NDVI",
                          "250m 16 days pixel reliability",
                          "250m 16 days composite day of the year"],
              "MYD13Q1": ["250m 16 days EVI"],
              "MCD15A2H": ["Lai_500m", "Fpar_500m", "FparLai_QC"],
              "MOD17A2HGF": ["Gpp_500m", "Psn_QC_500m"],
              "MOD16A2GF": ["ET_500m", "PET_500m", "ET_QC_500m"],
              "MOD11A2": ["LST_Day_1km", "LST_Night_1km", "QC_Day",
                          "QC_Night"],
              "MYD11A2": ["LST_Day_1km", "LST_Night_1km"],
              "MCD64A1": ["Burn Date", "QA"]}
    for prod, ls in layers.items():
        assert prod in f.GEE_ASSETS, prod
        for l in ls:
            assert l in f.GEE_ASSETS[prod]["bands"], (prod, l)


def test_gee_state_stats_band_names():
    import forestmodis as f
    import sys
    sys.modules["ee"] = _fake_ee()
    try:
        cfg = copy.deepcopy(f.DEFAULT_CONFIG)
        ls = [("250m 16 days EVI", "evi_terra"),
              ("250m 16 days NDVI", "ndvi_terra")]
        r = f.gee_state_stats(cfg, "MOD13Q1", 2003, "growing", ls,
                              use_pixel_doy="250m 16 days composite day "
                              "of the year")
    finally:
        del sys.modules["ee"]
    assert r is not None
    _, specs, n_exp = r
    names = [b for b, _, _ in specs]
    for pref in ("evi_terra", "ndvi_terra"):
        for k in ("mean", "median", "max", "p10", "p90", "coverage"):
            assert f"{pref}_growing_{k}" in names, names
    assert n_exp == 11          # композиты MOD13, пересекающие сезон 05-01..09-30


def test_gee_accumulated_band_names():
    import forestmodis as f
    import sys
    sys.modules["ee"] = _fake_ee()
    try:
        cfg = copy.deepcopy(f.DEFAULT_CONFIG)
        r = f.gee_accumulated(cfg, "MOD16A2GF", 2010, "growing",
                              [("ET_500m", "et"), ("PET_500m", "pet")])
    finally:
        del sys.modules["ee"]
    assert r is not None
    _, specs, n_exp = r
    names = [b for b, _, _ in specs]
    assert "et_growing_sum" in names and "pet_growing_sum" in names
    assert "et_growing_coverage" in names and "pet_growing_coverage" in names
    assert "etpet_growing_ratio" in names
    assert n_exp > 0


def test_validate_config_gee():
    import forestmodis as f
    cfg = copy.deepcopy(f.DEFAULT_CONFIG)
    cfg["data"]["backend"] = "gee"
    cfg["data"]["gee_gpp"] = "wrong"
    with pytest.raises(f.ConfigError):
        f.validate_config(cfg)
    cfg["data"]["gee_gpp"] = "annual"
    cfg["data"]["gee_project"] = "test-project"
    w = f.validate_config(cfg)
    assert not any("gee_project" in x for x in w)
    cfg["data"]["products"] = ["MCD12Q1", "BADPROD"]
    with pytest.raises(f.ConfigError):
        f.validate_config(cfg)


def test_gee_download_request_params(tmp_path, monkeypatch):
    """filePerBand=False + правильный порядок crsTransform + region."""
    import forestmodis as f
    import sys, io, types
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    grid = f.Grid(crs="EPSG:32637", transform=(500.0, 0, 700000, 0, -500.0,
                                              6000000),
                  width=6, height=4, res=500.0,
                  aoi_mask=np.ones((4, 6), bool), zones=None, zone_names={})
    prj = SimpleNamespace(cfg=copy.deepcopy(f.DEFAULT_CONFIG),
                          grid=lambda: grid, work=tmp_path)

    class _Img:
        def select(self, *a, **k):
            return self

        def unmask(self, *a, **k):
            return self

        def getDownloadURL(self, params):
            return "http://fake/gee"

    def _fake_get(url, timeout=None):
        arr = np.arange(6 * 4 * 2, dtype="float32").reshape(2, 4, 6)
        buf = io.BytesIO()
        with rasterio.open(buf, "w", driver="GTiff", height=4, width=6,
                           count=2, dtype="float32",
                           transform=from_origin(700000, 6000000, 500,
                                                 500)) as dst:
            dst.write(arr)
        return SimpleNamespace(content=buf.getvalue(),
                               raise_for_status=lambda: None)

    calls = {}

    class _RecImg(_Img):
        def getDownloadURL(self, params):
            calls["params"] = params
            return "http://fake/gee"

    import requests as _rq
    monkeypatch.setattr(_rq, "get", _fake_get)
    fake_ee = types.ModuleType("ee")
    fake_ee.Geometry = SimpleNamespace(Rectangle=lambda *a, **k: k)
    fake_ee.Projection = lambda wkt: wkt
    sys.modules["ee"] = fake_ee
    try:
        st = f.gee_download(prj, "lc", 2003, _RecImg(), [
            ("lc", "float32", np.nan, {}, "primary"),
            ("x", "float32", np.nan, {}, None)])
    finally:
        del sys.modules["ee"]
    p = calls["params"]
    assert p["filePerBand"] is False and p["format"] == "GEO_TIFF"
    assert p["crsTransform"] == [500.0, 0, 0, -500.0, 700000, 6000000.0]
    out = f.annual_path(prj, "lc", 2003)
    assert out.exists()
    with rasterio.open(out) as src:
        v = src.read(1)
    assert v.shape == (4, 6)
    assert st["lc"]["frac_valid"] == 1.0
