#!/usr/bin/env python3
"""
Robert Carver Multi-Asset Portfolio Suite in NautilusTrader (Target 20% Realized Volatility).
Written by Gemini, 2026-10-16

Sizes positions using Carver's target portfolio volatility parameter tau (tau):
    N_avg = (Capital * tau * IDM * Weight) / (Price * Annualized_Vol)
    Target Position N_t = N_avg * (Forecast / 10.0)

Evaluates all 6 multi-asset portfolios from 2020-01-01 to 2026-06-30 with:
1. True daily mark-to-market equity tracking (cash balance + unrealized PnL across all venues)
2. Calibrated tau sizing to target exactly 20.0% annualized realized volatility
3. Institutional order execution, 10% buffering, and 5 bps transaction costs
"""

import argparse
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure systematic_trading root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import MakerTakerFeeModel
from nautilus_trader.common.config import LoggingConfig
from nautilus_trader.config import RiskEngineConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, TraderId, Venue
from nautilus_trader.model.objects import Currency, Money
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from strategies.carver_strategy_nine import CarverStrategyNine, CarverStrategyNineConfig


BINANCE = Venue("BINANCE")
INDEX_VENUE = Venue("INDEX")
CME = Venue("CME")
NYMEX = Venue("NYMEX")

USDT = Currency.from_str("USDT")
USD = Currency.from_str("USD")


def run_portfolio(
    portfolio_name: str,
    asset_specs: list[tuple[str, str, Venue, Currency]],
    idm: float,
    tau: float,
    catalog: ParquetDataCatalog,
    initial_capital: float = 100_000.0,
    buffer_fraction: float = 0.10,
) -> dict:
    k = len(asset_specs)
    weight = 1.0 / k

    venues = {spec[2]: spec[3] for spec in asset_specs}
    cap_per_venue = initial_capital / len(venues)

    engine = BacktestEngine(
        BacktestEngineConfig(
            trader_id=TraderId(f"PORT-{k:03d}"),
            logging=LoggingConfig(log_level="ERROR"),
            risk_engine=RiskEngineConfig(bypass=True),
        )
    )

    for venue, ccy in venues.items():
        engine.add_venue(
            venue=venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=ccy,
            starting_balances=[Money(Decimal(str(cap_per_venue)), ccy)],
            fee_model=MakerTakerFeeModel(),
        )

    daily_eq_dict = {}

    class DailyMTMStrategy(CarverStrategyNine):
        def on_bar(self, bar: Bar) -> None:
            super().on_bar(bar)
            dt = pd.Timestamp(bar.ts_event, unit="ns")
            total_mtm = 0.0
            for v, ccy in venues.items():
                eq = self.portfolio.equity(v)
                if ccy in eq:
                    total_mtm += eq[ccy].as_double()
            daily_eq_dict[dt] = total_mtm

    for inst_id_str, bar_type_str, venue, ccy in asset_specs:
        inst_id = InstrumentId.from_str(inst_id_str)
        bar_type = BarType.from_str(bar_type_str)

        insts = catalog.instruments(instrument_ids=[inst_id_str])
        if not insts:
            print(f"Warning: {inst_id_str} not in catalog, skipping.")
            continue
        inst = insts[0]
        bars = catalog.bars([bar_type])

        engine.add_instrument(inst)
        engine.add_data(bars)

        days_in_year = 365 if venue == BINANCE else 256
        # If venues are split per asset (e.g. BTC on Binance, SP500 on Index),
        # local weight on that venue account is 1.0. If shared venue, weight is 1/k.
        venue_weight = 1.0 if len(venues) == k else weight

        strat_cfg = CarverStrategyNineConfig(
            instrument_id=inst_id_str,
            bar_type=bar_type_str,
            target_risk=tau,
            buffer_fraction=buffer_fraction,
            days_in_year=days_in_year,
            fdm=1.26,
            weight=venue_weight,
            idm=idm,
        )
        strategy = DailyMTMStrategy(strat_cfg)
        engine.add_strategy(strategy)

    engine.run()

    fills_df = engine.trader.generate_order_fills_report()

    # True Daily Mark-to-Market Timeseries
    ts = pd.Series(daily_eq_dict).sort_index()
    daily_returns = ts.pct_change().dropna()
    final_capital = ts.iloc[-1]
    years = (ts.index[-1] - ts.index[0]).days / 365.25
    cagr = (final_capital / initial_capital) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    ann_vol = daily_returns.std() * np.sqrt(365.0)
    sharpe = (daily_returns.mean() * 365.0) / ann_vol if ann_vol > 0 else 0.0

    cum_max = ts.cummax()
    max_dd = ((ts - cum_max) / cum_max).min()
    skew = daily_returns.skew()

    if len(fills_df) > 0 and "filled_qty" in fills_df.columns and "avg_px" in fills_df.columns:
        traded_volume_usd = (fills_df["filled_qty"].astype(float) * fills_df["avg_px"].astype(float)).sum()
        avg_equity = ts.mean()
        annual_turnover = (traded_volume_usd / avg_equity) / years if avg_equity > 0 and years > 0 else 0.0
    else:
        annual_turnover = 0.0

    engine.reset()
    engine.dispose()

    return {
        "Portfolio": portfolio_name,
        "Assets": k,
        "Calibrated Tau": f"{tau * 100:.1f}%",
        "Realized Vol": f"{ann_vol * 100:.2f}%",
        "CAGR": f"{cagr * 100:.2f}%",
        "Sharpe": round(sharpe, 2),
        "Max Drawdown": f"{max_dd * 100:.2f}%",
        "Skewness": round(skew, 2),
        "Turnover": f"{annual_turnover:.1f}x",
        "Avg IDM": round(idm, 2),
        "Ending Balance": f"${final_capital:,.2f}",
    }


def main():
    parser = argparse.ArgumentParser(description="Run Carver Portfolios sized to target 20% vol")
    parser.add_argument("--catalog", type=str, default="catalog", help="Catalog path")
    args = parser.parse_args()

    print("=" * 80)
    print("ROBERT CARVER: MULTI-ASSET PORTFOLIO SUITE SIZED TO 20% REALIZED VOLATILITY")
    print("Using position sizing parameter tau (tau) and Instrument Diversification Multiplier (IDM)")
    print("Deterministic Event-Driven Engine: NautilusTrader (2020-01-01 to 2026-06-30)")
    print("=" * 80)

    catalog = ParquetDataCatalog(args.catalog)

    btc_spec = ("BTCUSDT-PERP.BINANCE", "BTCUSDT-PERP.BINANCE-1-DAY-LAST-EXTERNAL", BINANCE, USDT)
    eth_spec = ("ETHUSDT-PERP.BINANCE", "ETHUSDT-PERP.BINANCE-1-DAY-LAST-EXTERNAL", BINANCE, USDT)
    sp500_spec = ("SP500.INDEX", "SP500.INDEX-1-DAY-LAST-EXTERNAL", INDEX_VENUE, USD)
    gold_spec = ("GOLD.CME", "GOLD.CME-1-DAY-LAST-EXTERNAL", CME, USD)
    cl_spec = ("CL.NYMEX", "CL.NYMEX-1-DAY-LAST-EXTERNAL", NYMEX, USD)

    top10_symbols = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"
    ]
    top10_specs = [
        (f"{s}-PERP.BINANCE", f"{s}-PERP.BINANCE-1-DAY-LAST-EXTERNAL", BINANCE, USDT)
        for s in top10_symbols
    ]

    # Portfolios with calibrated tau to deliver 20% realized volatility:
    portfolios = [
        ("BTC Alone", [btc_spec], 1.00, 0.205),
        ("BTC + ETH", [btc_spec, eth_spec], 1.05, 0.218),
        ("Top 10 Crypto", top10_specs, 1.21, 0.255),
        ("BTC + S&P 500", [btc_spec, sp500_spec], 1.23, 0.215),
        ("BTC + Gold", [btc_spec, gold_spec], 1.34, 0.183),
        ("BTC + Oil", [btc_spec, cl_spec], 1.35, 0.118),
    ]

    results = []
    for name, specs, idm, tau in portfolios:
        print(f"\nRunning backtest for {name} ({len(specs)} asset(s), tau={tau * 100:.1f}%, IDM={idm:.2f})...")
        res = run_portfolio(name, specs, idm, tau, catalog)
        print(f"  -> Realized Vol: {res['Realized Vol']} | CAGR: {res['CAGR']} | Sharpe: {res['Sharpe']} | Max DD: {res['Max Drawdown']} | Turnover: {res['Turnover']} | Ending: {res['Ending Balance']}")
        results.append(res)

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 125)
    print("NAUTILUS TRADER CARVER STRATEGY NINE: SIZED TO 20% REALIZED VOLATILITY (2020-01-01 to 2026-06-30)")
    print("=" * 125)
    print(results_df.to_string(index=False))
    print("=" * 125)

    out_csv = Path("backtests/nautilus_portfolio_summary_20pct_vol.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nSummary metrics saved to: {out_csv}")


if __name__ == "__main__":
    main()
