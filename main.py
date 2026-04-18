#!/usr/bin/env python3
"""
Research/backtesting script for BTC 5-minute Up/Down prediction.

Note:
- This is a research-first script (not live execution).
- 5-minute candle boundaries are aligned exactly to exchange timestamps.
- Polymarket settlement is based on Chainlink BTC/USD, which can differ from exchange or chart-provider prices.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


def _safe_import_ccxt():
    try:
        import ccxt  # type: ignore

        return ccxt
    except Exception:
        return None


def _safe_import_ta():
    try:
        import ta  # type: ignore

        return ta
    except Exception:
        return None


def _safe_import_joblib():
    try:
        import joblib  # type: ignore

        return joblib
    except Exception:
        return None


def _safe_import_xgboost():
    try:
        from xgboost import XGBClassifier  # type: ignore

        return XGBClassifier
    except Exception:
        return None


def _safe_import_lightgbm():
    try:
        from lightgbm import LGBMClassifier  # type: ignore

        return LGBMClassifier
    except Exception:
        return None


from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelArtifacts:
    model_name: str
    model: Any
    feature_cols: List[str]
    threshold_up: float
    threshold_down: float


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    summary: Dict[str, Any]


def fetch_ohlcv(
    exchange_id: str = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    since: Optional[int] = None,
    limit: int = 2000,
    max_batches: int = 10,
) -> pd.DataFrame:
    ccxt = _safe_import_ccxt()
    if ccxt is None:
        raise ImportError("ccxt is required. Install with: pip install ccxt")

    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt")

    ex_class = getattr(ccxt, exchange_id)
    exchange = ex_class({"enableRateLimit": True})

    if timeframe not in exchange.timeframes:
        raise ValueError(f"Timeframe '{timeframe}' unsupported on {exchange_id}")

    if since is None:
        now_ms = exchange.milliseconds()
        since = now_ms - limit * max_batches * 60_000

    all_rows: List[List[float]] = []
    cursor = since

    try:
        for _ in range(max_batches):
            rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
            if not rows:
                break
            all_rows.extend(rows)
            last_ts = rows[-1][0]
            next_cursor = last_ts + 60_000
            if next_cursor <= cursor:
                break
            cursor = next_cursor
    except Exception as e:
        raise RuntimeError(f"Failed fetching OHLCV from {exchange_id} {symbol}: {e}")

    if not all_rows:
        raise RuntimeError("No OHLCV data returned")

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime")

    # Keep fully closed candles only
    now_utc = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow()
    df = df[df.index <= now_utc.floor("1min")]
    return df


def _add_indicators_with_ta(df: pd.DataFrame) -> pd.DataFrame:
    ta = _safe_import_ta()
    out = df.copy()
    if ta is None:
        return out

    out["ema_9"] = ta.trend.ema_indicator(out["close"], window=9)
    out["ema_21"] = ta.trend.ema_indicator(out["close"], window=21)
    out["ema_50"] = ta.trend.ema_indicator(out["close"], window=50)
    out["rsi_14"] = ta.momentum.rsi(out["close"], window=14)

    macd = ta.trend.MACD(out["close"], window_fast=12, window_slow=26, window_sign=9)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(out["close"], window=20, window_dev=2)
    out["bb_mavg"] = bb.bollinger_mavg()
    out["bb_hband"] = bb.bollinger_hband()
    out["bb_lband"] = bb.bollinger_lband()
    out["bb_width"] = (out["bb_hband"] - out["bb_lband"]) / out["bb_mavg"].replace(0, np.nan)

    out["atr_14"] = ta.volatility.average_true_range(out["high"], out["low"], out["close"], window=14)
    return out


def _fallback_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_9"] = out["close"].ewm(span=9, adjust=False).mean()
    out["ema_21"] = out["close"].ewm(span=21, adjust=False).mean()
    out["ema_50"] = out["close"].ewm(span=50, adjust=False).mean()

    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_diff"] = out["macd"] - out["macd_signal"]

    mavg = out["close"].rolling(20).mean()
    mstd = out["close"].rolling(20).std()
    out["bb_mavg"] = mavg
    out["bb_hband"] = mavg + 2 * mstd
    out["bb_lband"] = mavg - 2 * mstd
    out["bb_width"] = (out["bb_hband"] - out["bb_lband"]) / out["bb_mavg"].replace(0, np.nan)

    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            (out["high"] - out["low"]).abs(),
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()
    return out


def make_1m_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    if df_1m.empty:
        raise ValueError("Input OHLCV dataframe is empty")

    df = df_1m.copy()

    with_ta = _add_indicators_with_ta(df)
    if {"ema_9", "rsi_14", "atr_14"}.issubset(with_ta.columns):
        df = with_ta
    else:
        df = _fallback_indicators(df)

    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    df["vwap"] = (tp * df["volume"]).cumsum() / cum_vol

    df["body"] = df["close"] - df["open"]
    df["range"] = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_abs"] = df["body"].abs()
    df["upper_wick"] = (df[["open", "close"]].max(axis=1) - df["high"]).abs()
    df["lower_wick"] = (df["low"] - df[["open", "close"]].min(axis=1)).abs()
    df["upper_wick_ratio"] = df["upper_wick"] / df["range"]
    df["lower_wick_ratio"] = df["lower_wick"] / df["range"]
    df["body_to_range"] = df["body_abs"] / df["range"]

    for n in [1, 3, 5, 15]:
        df[f"ret_{n}m"] = df["close"].pct_change(n)
        df[f"mom_{n}m"] = df["close"] - df["close"].shift(n)

    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["realized_vol_5m"] = log_ret.rolling(5).std() * np.sqrt(5)
    df["realized_vol_15m"] = log_ret.rolling(15).std() * np.sqrt(15)

    for col in ["ema_9", "ema_21", "ema_50", "vwap", "bb_mavg", "bb_hband", "bb_lband"]:
        if col in df.columns:
            df[f"dist_{col}"] = (df["close"] - df[col]) / df["close"].replace(0, np.nan)

    ema_spread = (df["ema_9"] - df["ema_21"]).abs() / df["close"].replace(0, np.nan)
    df["trend_strength"] = ema_spread
    df["chop_score"] = (df["atr_14"] / df["close"].replace(0, np.nan)) / (ema_spread.replace(0, np.nan))
    df["trend_regime"] = (df["ema_9"] > df["ema_21"]).astype(float)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def resample_to_5m(df_1m_features: pd.DataFrame) -> pd.DataFrame:
    if df_1m_features.empty:
        raise ValueError("1m feature dataframe is empty")

    ohlc_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df_1m_features.columns]
    if missing:
        raise ValueError(f"Missing required columns for resample: {missing}")

    bars_5m = df_1m_features[required_cols].resample("5min", label="left", closed="left").agg(ohlc_dict)

    feat_cols = [c for c in df_1m_features.columns if c not in required_cols + ["timestamp"]]
    feat_agg = df_1m_features[feat_cols].resample("5min", label="left", closed="left").last()

    out = bars_5m.join(feat_agg, how="left")

    # Keep only completed 5m bars aligned to exchange timestamp boundaries.
    now = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow()
    last_closed_bar_start = now.floor("5min") - pd.Timedelta(minutes=5)
    out = out[out.index <= last_closed_bar_start]

    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def make_labels(df_5m: pd.DataFrame) -> pd.DataFrame:
    if df_5m.empty:
        raise ValueError("5m dataframe is empty")

    df = df_5m.copy()
    df["next_open"] = df["open"].shift(-1)
    df["next_close"] = df["close"].shift(-1)
    df["target"] = (df["next_close"] >= df["next_open"]).astype(float)

    df = df.iloc[:-1].copy()
    return df


def _time_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    train = df.iloc[:i_train].copy()
    val = df.iloc[i_train:i_val].copy()
    test = df.iloc[i_val:].copy()
    return train, val, test


def _build_logistic_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, slice(0, None))])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
    return pipe


def _build_boosting_model(random_state: int = 42):
    XGBClassifier = _safe_import_xgboost()
    if XGBClassifier is not None:
        return (
            "xgboost",
            XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=4,
            ),
        )

    LGBMClassifier = _safe_import_lightgbm()
    if LGBMClassifier is not None:
        return (
            "lightgbm",
            LGBMClassifier(
                n_estimators=400,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
            ),
        )

    return (
        "hist_gradient_boosting",
        HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.03,
            max_depth=6,
            random_state=random_state,
        ),
    )


def _predict_proba_safe(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        return p.ravel()
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1 / (1 + np.exp(-z))
    preds = model.predict(X)
    return np.asarray(preds).astype(float)


def _evaluate(y_true: np.ndarray, p_up: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    pred = (p_up >= threshold).astype(int)
    acc = accuracy_score(y_true, pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, pred)
    out: Dict[str, Any] = {
        "accuracy": float(acc),
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "class_balance": float(np.mean(y_true)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, p_up))
    except Exception:
        out["roc_auc"] = np.nan
    return out


def _optimize_thresholds(
    y_val: np.ndarray,
    p_val: np.ndarray,
    fee_buffer: float = 0.0,
    min_coverage: float = 0.05,
) -> Tuple[float, float, Dict[str, Any]]:
    best = {
        "score": -np.inf,
        "threshold_up": 0.6,
        "threshold_down": 0.4,
        "coverage": 0.0,
        "directional_acc": 0.0,
    }

    grid = np.arange(0.50, 0.86, 0.02)
    for up in grid:
        for down in (1 - grid):
            if down >= up:
                continue
            long_mask = p_val >= up
            short_mask = p_val <= down
            trade_mask = long_mask | short_mask
            coverage = float(trade_mask.mean())
            if coverage < min_coverage:
                continue

            pred = np.full_like(y_val, fill_value=-1, dtype=int)
            pred[long_mask] = 1
            pred[short_mask] = 0

            traded = pred != -1
            if traded.sum() == 0:
                continue
            directional_acc = (pred[traded] == y_val[traded]).mean()
            score = directional_acc - fee_buffer

            if score > best["score"]:
                best.update(
                    {
                        "score": float(score),
                        "threshold_up": float(up),
                        "threshold_down": float(down),
                        "coverage": coverage,
                        "directional_acc": float(directional_acc),
                    }
                )

    return best["threshold_up"], best["threshold_down"], best


def train_models(
    df_labeled: pd.DataFrame,
    target_col: str = "target",
    random_state: int = 42,
) -> Dict[str, Any]:
    drop_cols = {
        target_col,
        "next_open",
        "next_close",
    }

    feature_cols = [c for c in df_labeled.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df_labeled[c])]
    work = df_labeled[feature_cols + [target_col]].dropna().copy()

    if len(work) < 500:
        raise ValueError("Not enough cleaned samples after feature engineering; need at least 500 rows")

    train_df, val_df, test_df = _time_split(work)

    X_train, y_train = train_df[feature_cols], train_df[target_col].astype(int).values
    X_val, y_val = val_df[feature_cols], val_df[target_col].astype(int).values
    X_test, y_test = test_df[feature_cols], test_df[target_col].astype(int).values

    logit = _build_logistic_pipeline()
    boost_name, boost = _build_boosting_model(random_state=random_state)

    logit.fit(X_train, y_train)
    boost.fit(X_train, y_train)

    p_val_logit = _predict_proba_safe(logit, X_val)
    p_val_boost = _predict_proba_safe(boost, X_val)

    val_logit = _evaluate(y_val, p_val_logit)
    val_boost = _evaluate(y_val, p_val_boost)

    best_name = "logistic" if val_logit.get("roc_auc", -np.inf) >= val_boost.get("roc_auc", -np.inf) else boost_name
    best_model = logit if best_name == "logistic" else boost
    p_val_best = p_val_logit if best_name == "logistic" else p_val_boost

    threshold_up, threshold_down, threshold_meta = _optimize_thresholds(y_val, p_val_best, fee_buffer=0.0)

    p_test_best = _predict_proba_safe(best_model, X_test)
    test_metrics = _evaluate(y_test, p_test_best)

    artifacts = ModelArtifacts(
        model_name=best_name,
        model=best_model,
        feature_cols=feature_cols,
        threshold_up=threshold_up,
        threshold_down=threshold_down,
    )

    return {
        "artifacts": artifacts,
        "splits": {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        },
        "metrics": {
            "val_logistic": val_logit,
            f"val_{boost_name}": val_boost,
            "test_best": test_metrics,
            "threshold_optimization": threshold_meta,
        },
    }


def walk_forward_backtest(
    df_labeled: pd.DataFrame,
    feature_cols: List[str],
    threshold_up: float,
    threshold_down: float,
    min_train_size: int = 1500,
    step_size: int = 50,
) -> BacktestResult:
    data = df_labeled[feature_cols + ["target"]].dropna().copy()
    if len(data) <= min_train_size + step_size:
        raise ValueError("Not enough data for walk-forward backtest")

    rows = []
    model_template = _build_logistic_pipeline()

    for i in range(min_train_size, len(data) - 1, step_size):
        train = data.iloc[:i]
        test = data.iloc[i : i + step_size]
        if test.empty:
            continue

        X_tr, y_tr = train[feature_cols], train["target"].astype(int)
        X_te, y_te = test[feature_cols], test["target"].astype(int)

        model = clone(model_template)
        model.fit(X_tr, y_tr)
        p = _predict_proba_safe(model, X_te)

        action = np.where(p >= threshold_up, 1, np.where(p <= threshold_down, 0, -1))
        traded = action != -1

        result = pd.DataFrame(
            {
                "timestamp": test.index,
                "p_up": p,
                "y_true": y_te.values,
                "action": action,
                "traded": traded,
            }
        )

        result["correct"] = np.where(result["traded"], result["action"] == result["y_true"], np.nan)
        rows.append(result)

    trades = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if trades.empty:
        summary = {
            "n_samples": 0,
            "n_trades": 0,
            "coverage": 0.0,
            "directional_accuracy": np.nan,
        }
        return BacktestResult(trades=trades, summary=summary)

    n_samples = len(trades)
    n_trades = int(trades["traded"].sum())
    coverage = n_trades / n_samples if n_samples else 0.0
    directional_accuracy = float(trades.loc[trades["traded"], "correct"].mean()) if n_trades > 0 else np.nan

    summary = {
        "n_samples": int(n_samples),
        "n_trades": int(n_trades),
        "coverage": float(coverage),
        "directional_accuracy": directional_accuracy,
    }
    return BacktestResult(trades=trades, summary=summary)


def compute_ev(
    p_up: float,
    polymarket_up_price: float,
    polymarket_down_price: float,
    estimated_fee_rate: float = 0.02,
) -> Dict[str, float]:
    p_up = float(np.clip(p_up, 0.0, 1.0))
    p_down = 1.0 - p_up

    up_price = float(np.clip(polymarket_up_price, 1e-6, 0.999999))
    down_price = float(np.clip(polymarket_down_price, 1e-6, 0.999999))

    payout_up_win = (1.0 - up_price) * (1.0 - estimated_fee_rate)
    loss_up = up_price * (1.0 + estimated_fee_rate)

    payout_down_win = (1.0 - down_price) * (1.0 - estimated_fee_rate)
    loss_down = down_price * (1.0 + estimated_fee_rate)

    ev_up = p_up * payout_up_win - (1.0 - p_up) * loss_up
    ev_down = p_down * payout_down_win - (1.0 - p_down) * loss_down

    return {
        "p_up": p_up,
        "p_down": p_down,
        "ev_up": float(ev_up),
        "ev_down": float(ev_down),
        "payout_up_win": float(payout_up_win),
        "payout_down_win": float(payout_down_win),
        "loss_up": float(loss_up),
        "loss_down": float(loss_down),
    }


def latest_signal(
    artifacts: ModelArtifacts,
    df_labeled: pd.DataFrame,
    polymarket_up_price: float,
    polymarket_down_price: float,
    estimated_fee_rate: float,
    min_ev_edge: float = 0.0,
) -> Dict[str, Any]:
    feat_cols = artifacts.feature_cols
    latest_row = df_labeled[feat_cols].dropna().iloc[-1:]

    p_up = float(_predict_proba_safe(artifacts.model, latest_row)[0])

    action_model = "NO TRADE"
    if p_up >= artifacts.threshold_up:
        action_model = "UP"
    elif p_up <= artifacts.threshold_down:
        action_model = "DOWN"

    ev = compute_ev(
        p_up=p_up,
        polymarket_up_price=polymarket_up_price,
        polymarket_down_price=polymarket_down_price,
        estimated_fee_rate=estimated_fee_rate,
    )

    action_ev = "NO TRADE"
    if ev["ev_up"] > ev["ev_down"] and ev["ev_up"] > min_ev_edge:
        action_ev = "UP"
    elif ev["ev_down"] > ev["ev_up"] and ev["ev_down"] > min_ev_edge:
        action_ev = "DOWN"

    return {
        "latest_features": latest_row.iloc[0].to_dict(),
        "p_up": p_up,
        "model_action": action_model,
        "ev": ev,
        "ev_action": action_ev,
    }


def _print_feature_importance(model: Any, feature_cols: List[str], top_n: int = 15) -> None:
    print("\nTop feature importances (if available):")
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_)
        idx = np.argsort(vals)[::-1][:top_n]
        for i in idx:
            print(f"  {feature_cols[i]}: {vals[i]:.6f}")
        return

    if hasattr(model, "named_steps") and "model" in model.named_steps:
        inner = model.named_steps["model"]
        if hasattr(inner, "coef_"):
            coef = np.ravel(inner.coef_)
            idx = np.argsort(np.abs(coef))[::-1][:top_n]
            for i in idx:
                print(f"  {feature_cols[i]}: {coef[i]:+.6f}")
            return

    print("  Not available for this model type.")


def main() -> int:
    parser = argparse.ArgumentParser(description="BTC 5m Up/Down research/backtest script")
    parser.add_argument("--exchange", default="binance", help="ccxt exchange id")
    parser.add_argument("--symbol", default="BTC/USDT", help="Market symbol")
    parser.add_argument("--limit", type=int, default=1500, help="Per-fetch limit")
    parser.add_argument("--max-batches", type=int, default=8, help="Number of paginated fetches")
    parser.add_argument("--save-model", action="store_true", help="Save model artifacts via joblib")
    parser.add_argument("--model-path", default="model_artifacts.joblib", help="Path to save model artifacts")
    parser.add_argument("--save-backtest", action="store_true", help="Save backtest trades to CSV")
    parser.add_argument("--backtest-path", default="backtest_trades.csv", help="CSV path for backtest trades")

    # Polymarket-specific placeholders (replace with live values when integrated)
    parser.add_argument("--polymarket-up-price", type=float, default=0.52)
    parser.add_argument("--polymarket-down-price", type=float, default=0.48)
    parser.add_argument("--estimated-fee-rate", type=float, default=0.02)
    parser.add_argument("--min-ev-edge", type=float, default=0.0)

    args = parser.parse_args()

    try:
        df_1m = fetch_ohlcv(
            exchange_id=args.exchange,
            symbol=args.symbol,
            timeframe="1m",
            limit=args.limit,
            max_batches=args.max_batches,
        )

        feats_1m = make_1m_features(df_1m)
        bars_5m = resample_to_5m(feats_1m)
        labeled = make_labels(bars_5m)

        train_out = train_models(labeled)
        artifacts: ModelArtifacts = train_out["artifacts"]

        bt = walk_forward_backtest(
            labeled,
            artifacts.feature_cols,
            threshold_up=artifacts.threshold_up,
            threshold_down=artifacts.threshold_down,
        )

        sig = latest_signal(
            artifacts=artifacts,
            df_labeled=labeled,
            polymarket_up_price=args.polymarket_up_price,
            polymarket_down_price=args.polymarket_down_price,
            estimated_fee_rate=args.estimated_fee_rate,
            min_ev_edge=args.min_ev_edge,
        )

        print("=" * 80)
        print("MODEL SELECTION / METRICS")
        print("=" * 80)
        print(f"Best model: {artifacts.model_name}")
        print(json.dumps(train_out["metrics"], indent=2, default=str))

        print("\n" + "=" * 80)
        print("LATEST FEATURE SNAPSHOT")
        print("=" * 80)
        latest_series = pd.Series(sig["latest_features"]).sort_index()
        print(latest_series.tail(30).to_string())

        print("\n" + "=" * 80)
        print("LATEST PREDICTION")
        print("=" * 80)
        print(f"p_up(next 5m): {sig['p_up']:.4f}")
        print(
            f"Decision thresholds: up >= {artifacts.threshold_up:.2f}, down <= {artifacts.threshold_down:.2f}"
        )
        print(f"Model action: {sig['model_action']}")

        print("\n" + "=" * 80)
        print("POLYMARKET EV LAYER (PLACEHOLDER PRICES)")
        print("=" * 80)
        ev = sig["ev"]
        print(f"polymarket_up_price: {args.polymarket_up_price:.4f}")
        print(f"polymarket_down_price: {args.polymarket_down_price:.4f}")
        print(f"estimated_fee_rate: {args.estimated_fee_rate:.4f}")
        print(f"ev_up: {ev['ev_up']:.6f}")
        print(f"ev_down: {ev['ev_down']:.6f}")
        print(f"EV action: {sig['ev_action']}")

        print("\n" + "=" * 80)
        print("WALK-FORWARD BACKTEST SUMMARY")
        print("=" * 80)
        print(json.dumps(bt.summary, indent=2, default=str))

        # Transaction-cost / edge filter placeholders (research hook)
        print("\nEdge filter placeholder: require expected edge > fees/slippage before live deployment.")

        _print_feature_importance(artifacts.model, artifacts.feature_cols)

        if args.save_model:
            joblib = _safe_import_joblib()
            if joblib is None:
                print("\n[WARN] joblib not installed; skipping model save.")
            else:
                payload = {
                    "model_name": artifacts.model_name,
                    "model": artifacts.model,
                    "feature_cols": artifacts.feature_cols,
                    "threshold_up": artifacts.threshold_up,
                    "threshold_down": artifacts.threshold_down,
                }
                joblib.dump(payload, args.model_path)
                print(f"\nSaved model artifacts -> {args.model_path}")

        if args.save_backtest:
            bt.trades.to_csv(args.backtest_path, index=False)
            print(f"Saved backtest trades -> {args.backtest_path}")

        # Final recommended action combining model confidence and EV layer
        final_action = sig["ev_action"] if sig["model_action"] != "NO TRADE" else "NO TRADE"
        print("\n" + "=" * 80)
        print(f"RECOMMENDED ACTION: {final_action}")
        print("=" * 80)

        return 0

    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
