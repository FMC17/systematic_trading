#!/usr/bin/env python3
"""
Build Nautilus Parquet Data Catalog from Historical Parquet Files.
Written by Gemini, 2026-10-16

Ingests daily historical OHLCV data into the institutional Nautilus ParquetDataCatalog:
1. Crypto Perps: BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, LTC
2. TradFi Macro: S&P 500, Gold (CME), Crude Oil (CL)
"""

import os
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AssetClass
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import Commodity, CryptoPerpetual, Equity
from nautilus_trader.model.objects import Currency, Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.persistence.wranglers import BarDataWrangler


BINANCE = Venue("BINANCE")
CME = Venue("CME")
NYMEX = Venue("NYMEX")
INDEX_VENUE = Venue("INDEX")

USD = Currency.from_str("USD")
USDT = Currency.from_str("USDT")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares and validates standard OHLCV dataframe for Nautilus BarDataWrangler."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df.set_index(pd.to_datetime(df["timestamp"]), inplace=True)
        else:
            df.index = pd.to_datetime(df.index)

    # Standardize column names
    col_map = {c: c.lower() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    if "close" in df.columns and "open" not in df.columns:
        df["open"] = df["close"]
    if "high" not in df.columns:
        df["high"] = df[["open", "close"]].max(axis=1)
    if "low" not in df.columns:
        df["low"] = df[["open", "close"]].min(axis=1)
    if "volume" not in df.columns:
        df["volume"] = 1_000_000.0

    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df = df.sort_index()

    # Filter to 2020-01-01 to 2026-06-30
    df = df.loc["2020-01-01":"2026-06-30"]

    # Ensure C-contiguous float64 numpy arrays
    arr = np.ascontiguousarray(df.values, dtype=np.float64)
    clean_df = pd.DataFrame(arr, index=pd.DatetimeIndex(df.index), columns=["open", "high", "low", "close", "volume"])
    return clean_df


def build_catalog(catalog_dir: str = "catalog", data_dir: str = "data"):
    catalog_path = Path(catalog_dir).resolve()
    catalog = ParquetDataCatalog(str(catalog_path))
    print(f"Initialized ParquetDataCatalog at: {catalog_path}")

    # Instrument definitions
    crypto_instruments = [
        ("BTC", "BTCUSDT", "data/BTC_PERP_BINANCE_1d.parquet", 2, 3, "0.01", "0.001"),
        ("ETH", "ETHUSDT", "data/universe/binance_um_1d/ETHUSDT.parquet", 2, 3, "0.01", "0.001"),
        ("SOL", "SOLUSDT", "data/universe/binance_um_1d/SOLUSDT.parquet", 2, 2, "0.01", "0.01"),
        ("BNB", "BNBUSDT", "data/universe/binance_um_1d/BNBUSDT.parquet", 2, 2, "0.01", "0.01"),
        ("XRP", "XRPUSDT", "data/universe/binance_um_1d/XRPUSDT.parquet", 4, 1, "0.0001", "0.1"),
        ("DOGE", "DOGEUSDT", "data/universe/binance_um_1d/DOGEUSDT.parquet", 5, 0, "0.00001", "1"),
        ("ADA", "ADAUSDT", "data/universe/binance_um_1d/ADAUSDT.parquet", 4, 1, "0.0001", "0.1"),
        ("AVAX", "AVAXUSDT", "data/universe/binance_um_1d/AVAXUSDT.parquet", 2, 2, "0.01", "0.01"),
        ("LINK", "LINKUSDT", "data/universe/binance_um_1d/LINKUSDT.parquet", 3, 2, "0.001", "0.01"),
        ("LTC", "LTCUSDT", "data/universe/binance_um_1d/LTCUSDT.parquet", 2, 3, "0.01", "0.001"),
    ]

    for base_str, sym_str, file_rel, p_prec, s_prec, p_inc, s_inc in crypto_instruments:
        file_path = Path(file_rel)
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}, skipping.")
            continue

        symbol_id = f"{sym_str}-PERP"
        instrument_id = InstrumentId(Symbol(symbol_id), BINANCE)
        bar_type_str = f"{symbol_id}.BINANCE-1-DAY-LAST-EXTERNAL"
        bar_type = BarType.from_str(bar_type_str)

        inst = CryptoPerpetual(
            instrument_id=instrument_id,
            raw_symbol=Symbol(sym_str),
            base_currency=Currency.from_str(base_str),
            quote_currency=USDT,
            settlement_currency=USDT,
            is_inverse=False,
            price_precision=p_prec,
            size_precision=s_prec,
            price_increment=Price.from_str(p_inc),
            size_increment=Quantity.from_str(s_inc),
            maker_fee=Decimal("0.0002"),
            taker_fee=Decimal("0.0005"),
            ts_event=0,
            ts_init=0,
        )
        catalog.write_data([inst])

        raw_df = pd.read_parquet(file_path)
        clean_df = clean_dataframe(raw_df)
        wrangler = BarDataWrangler(bar_type, inst)
        bars = wrangler.process(clean_df)
        catalog.write_data(bars)
        print(f"Wrote {len(bars)} bars for {instrument_id} ({clean_df.index[0].date()} to {clean_df.index[-1].date()})")

    # TradFi / Macro Instruments
    tradfi = [
        ("SP500", "data/SP500_INDEX_1d.parquet", INDEX_VENUE, AssetClass.INDEX, "SP500.INDEX-1-DAY-LAST-EXTERNAL"),
        ("GOLD", "data/GOLD_FUTURES_CME_1d.parquet", CME, AssetClass.COMMODITY, "GOLD.CME-1-DAY-LAST-EXTERNAL"),
        ("CL", "data/CL_COMMODITY_1d.parquet", NYMEX, AssetClass.COMMODITY, "CL.NYMEX-1-DAY-LAST-EXTERNAL"),
    ]

    for sym_str, file_rel, venue, asset_class, bar_type_str in tradfi:
        file_path = Path(file_rel)
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}, skipping.")
            continue

        instrument_id = InstrumentId(Symbol(sym_str), venue)
        bar_type = BarType.from_str(bar_type_str)

        if asset_class == AssetClass.COMMODITY:
            inst = Commodity(
                instrument_id=instrument_id,
                raw_symbol=Symbol(sym_str),
                asset_class=asset_class,
                quote_currency=USD,
                price_precision=2,
                size_precision=4,
                price_increment=Price.from_str("0.01"),
                size_increment=Quantity.from_str("0.0001"),
                lot_size=Quantity.from_int(1),
                ts_event=0,
                ts_init=0,
            )
        else:
            inst = Equity(
                instrument_id=instrument_id,
                raw_symbol=Symbol(sym_str),
                currency=USD,
                price_precision=2,
                price_increment=Price.from_str("0.01"),
                lot_size=Quantity.from_int(1),
                ts_event=0,
                ts_init=0,
            )

        catalog.write_data([inst])
        raw_df = pd.read_parquet(file_path)
        clean_df = clean_dataframe(raw_df)
        wrangler = BarDataWrangler(bar_type, inst)
        bars = wrangler.process(clean_df)
        catalog.write_data(bars)
        print(f"Wrote {len(bars)} bars for {instrument_id} ({clean_df.index[0].date()} to {clean_df.index[-1].date()})")

    print("\nCatalog build complete!")
    print(f"Total Instruments registered: {len(catalog.instruments())}")


if __name__ == "__main__":
    build_catalog()
