"""Forecast backends for the forecast-vetting feature (Component #14).

Several forecasters live behind one interface so the bake-off
(experiments/forecast_bakeoff.py) can compare them on the same footing and we ship whichever
wins as the live model. The point of this module is NOT to predict prices well - on daily stock
DIRECTION every one of these hovers near a coin flip (efficient-market hypothesis). It exists so
the agent has a concrete, price-only forecast to VET against news the model never saw.

Backends:
  naive_forecast   - random walk with drift. The honest baseline/floor. No extra deps.
  stat_forecast    - classical statistical models via statsforecast (AutoARIMA/AutoETS/AutoTheta).
                     Lightweight, no torch. Imported lazily.
  chronos_forecast - Amazon Chronos-Bolt time-series foundation model, zero-shot. Pulls torch;
                     lazy-loaded and gated so everything else works whether or not it's installed.

Every backend returns the SAME dict shape and fits ONLY on the rows handed to it. The optional
`as_of` argument is a tripwire: if given, we assert the last in-sample date is strictly before it,
so a lookahead bug fails loudly instead of silently inflating accuracy. That no-lookahead property
is the whole basis of the vetting claim.
"""
from . import config


def _closes(series):
    """Pull the close column out of a market_data series (US or PH rows both carry 'close')."""
    return [float(r["close"]) for r in series if r.get("close") is not None]


def _direction(last_close, final_point):
    diff = final_point - last_close
    return "up" if diff > 0 else "down" if diff < 0 else "flat"


def _check_no_lookahead(series, as_of):
    """Fail loudly if the caller left in any row dated on/after the forecast's as-of date."""
    if as_of and series and series[-1].get("date") and str(series[-1]["date"]) >= str(as_of):
        raise ValueError(
            f"lookahead: last in-sample date {series[-1]['date']} is not before as_of {as_of}")


def _result(model, series, points, lo=None, hi=None, order=None):
    """Assemble the common return shape from a list of point forecasts (+ optional intervals)."""
    last_close = round(_closes(series)[-1], 2)
    preds = []
    for i, p in enumerate(points):
        preds.append({
            "step": i + 1,
            "point": round(float(p), 2),
            "lo": round(float(lo[i]), 2) if lo is not None else None,
            "hi": round(float(hi[i]), 2) if hi is not None else None,
        })
    out = {
        "ok": True,
        "model": model,
        "horizon": len(points),
        "as_of": series[-1]["date"],   # last close the model was allowed to see
        "last_close": last_close,
        "predictions": preds,
        "direction": _direction(last_close, float(points[-1])),
    }
    if order is not None:
        out["order"] = order
    return out


# --- baseline ---------------------------------------------------------------

def naive_forecast(series, horizon=5, as_of=None) -> dict:
    """Random walk with drift: project the average recent daily change forward. This is the floor
    every other model has to beat, and the honest 'no real signal' comparison."""
    _check_no_lookahead(series, as_of)
    closes = _closes(series)
    if len(closes) < 2:
        return {"ok": False, "model": "naive_drift", "error": "need >= 2 closes"}
    diffs = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    drift = sum(diffs) / len(diffs)
    last = closes[-1]
    points = [last + drift * (k + 1) for k in range(horizon)]
    return _result("naive_drift", series, points)


# --- classical statistical (statsforecast) ----------------------------------

_STAT_MODELS = {"arima": "AutoARIMA", "ets": "AutoETS", "theta": "AutoTheta"}


def stat_forecast(series, horizon=5, model="ets", as_of=None) -> dict:
    """Classical statistical forecast via Nixtla statsforecast. `model` is one of arima/ets/theta.
    season_length=1 (daily closes, no seasonal cycle assumed). No torch."""
    _check_no_lookahead(series, as_of)
    closes = _closes(series)
    if len(closes) < 8:
        return {"ok": False, "model": f"stat_{model}", "error": "need >= 8 closes"}
    cls_name = _STAT_MODELS.get(model)
    if cls_name is None:
        return {"ok": False, "model": f"stat_{model}", "error": f"unknown stat model {model}"}
    try:
        import numpy as np
        from statsforecast import models as sf_models
        m = getattr(sf_models, cls_name)(season_length=1)
        y = np.asarray(closes, dtype="float64")
        res = m.forecast(y=y, h=horizon, level=[95])
        mean = res["mean"]
        lo = res.get("lo-95")
        hi = res.get("hi-95")
        return _result(f"stat_{model}", series, mean, lo=lo, hi=hi)
    except Exception as e:
        return {"ok": False, "model": f"stat_{model}", "error": f"{type(e).__name__}: {e}"}


# --- foundation model (Chronos-Bolt) ----------------------------------------

_chronos = None


def _load_chronos():
    """Lazy module singleton for the Chronos-Bolt pipeline (like rag._kb). Loads torch + the
    pretrained weights on first use only, so importing this module never requires torch."""
    global _chronos
    if _chronos is None:
        import torch
        from chronos import BaseChronosPipeline
        _chronos = BaseChronosPipeline.from_pretrained(
            config.CHRONOS_MODEL, device_map="cpu", torch_dtype=torch.float32)
    return _chronos


def chronos_forecast(series, horizon=5, as_of=None) -> dict:
    """Zero-shot forecast from the Chronos-Bolt foundation model. Returns the same shape as the
    others; gated so a missing torch/chronos install degrades to a clean error instead of raising."""
    _check_no_lookahead(series, as_of)
    closes = _closes(series)
    if len(closes) < 8:
        return {"ok": False, "model": "chronos_bolt", "error": "need >= 8 closes"}
    try:
        import torch
        pipe = _load_chronos()
        context = torch.tensor(closes, dtype=torch.float32)
        # predict_quantiles(inputs, ...) -> (quantiles [batch, horizon, n_levels], mean [batch, horizon])
        quantiles, mean = pipe.predict_quantiles(
            context, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9])
        point = mean[0].tolist()
        lo = quantiles[0, :, 0].tolist()
        hi = quantiles[0, :, 2].tolist()
        return _result("chronos_bolt", series, point, lo=lo, hi=hi)
    except ModuleNotFoundError:
        return {"ok": False, "model": "chronos_bolt", "error": "chronos unavailable (torch/chronos not installed)"}
    except Exception as e:
        return {"ok": False, "model": "chronos_bolt", "error": f"{type(e).__name__}: {e}"}


# --- dispatcher --------------------------------------------------------------

def forecast(series, horizon=5, backend="auto", as_of=None) -> dict:
    """Run one backend by name. backend='auto' uses config.FORECAST_BACKEND (the bake-off winner).
    Valid backends: naive, ets, arima, theta, chronos."""
    b = config.FORECAST_BACKEND if backend == "auto" else backend
    if b == "naive":
        return naive_forecast(series, horizon, as_of=as_of)
    if b in _STAT_MODELS:
        return stat_forecast(series, horizon, model=b, as_of=as_of)
    if b == "chronos":
        return chronos_forecast(series, horizon, as_of=as_of)
    return {"ok": False, "model": b, "error": f"unknown backend {b}"}
