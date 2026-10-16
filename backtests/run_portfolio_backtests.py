#!/usr/bin/env python3
"""
Robert Carver Multi-Asset Portfolio Suite inside NautilusTrader.
Written by Gemini, 2026-10-16

Tests Robert Carver Strategy Nine (6-Speed EWMAC Ensemble + FDM + 10% Buffering)
across 6 portfolios natively in the NautilusTrader event-driven engine:
1. BTC Alone
2. BTC + ETH
3. Top 10 Crypto Universe (ex-stablecoins)
4. BTC + S&P 500
5. BTC + Gold (CME)
6. BTC + Oil (CL NYMEX)

All backtests run deterministically on historical daily bars from 2020-01-01 to 2026-06-30
with full margin accounting, realistic fee models (5 bps), and 10% position buffering.
"""

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
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, TraderId, Venue
from nautilus_trader.model.objects import Currency, Money
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from strategies.carver_strategy_nine import CarverStrategyNine, CarverStrategyNineConfig


BINANCE = Venue("BINANCE")
CME = Venue("CME")
NYMEX = Venue("NYMEX")
INDEX_VENUE = Venue("INDEX")

USDT = Currency.from_str("USDT")
USD = Currency.from_str("USD")


def run_portfolio_backtest(
    portfolio_name: str,
    asset_specs: list[tuple[str, str, Venue, Currency]],
    idm: float,
    catalog: ParquetDataCatalog,
    initial_capital: float = 100_000.0,
    target_risk: float = 0.20,
    buffer_fraction: float = 0.10,
) -> dict:
    k = len(asset_specs)
    weight = 1.0 / k

    engine = BacktestEngine(
        BacktestEngineConfig(
            trader_id=TraderId(f"PORT-{k:03d}"),
            logging=LoggingConfig(log_level="ERROR"),
            risk_engine=RiskEngineConfig(bypass=True),
        )
    )

    # Group starting capital across required venues
    venues = {spec[2]: spec[3] for spec in asset_specs}
    capital_per_venue = initial_capital / len(venues)

    for venue, ccy in venues.items():
        engine.add_venue(
            venue=venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=ccy,
            starting_balances=[Money(Decimal(str(capital_per_venue)), ccy)],
            fee_model=MakerTakerFeeModel(),
        )

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
        strat_cfg = CarverStrategyNineConfig(
            instrument_id=inst_id_str,
            bar_type=bar_type_str,
            target_risk=target_risk,
            buffer_fraction=buffer_fraction,
            days_in_year=days_in_year,
            fdm=1.26,
            weight=weight,
            idm=idm,
        )
        strategy = CarverStrategyNine(strat_cfg)
        engine.add_strategy(strategy)

    engine.run()

    # Aggregate equity curve across all venues
    daily_equity_series = []
    for venue in venues.keys():
        acct_df = engine.trader.generate_account_report(venue)
        if len(acct_df) > 0:
            v_daily = acct_df["total"].astype(float).resample("D").last().ffill()
            daily_equity_series.append(v_daily)

    if daily_equity_series:
        combined_equity = pd.concat(daily_equity_series, axis=1).ffill().sum(axis=1)
    else:
        combined_equity = pd.Series([initial_capital])

    fills_df = engine.trader.generate_order_fills_report()

    # Performance Metrics
    daily_returns = combined_equity.pct_change().dropna()
    final_capital = combined_equity.iloc[-1]
    years = (combined_equity.index[-1] - combined_equity.index[0]).days / 365.25
    cagr = (final_capital / initial_capital) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    ann_vol = daily_returns.std() * np.sqrt(365.0)
    sharpe = (daily_returns.mean() * 365.0) / ann_vol if ann_vol > 0 else 0.0

    cum_max = combined_equity.cummax()
    max_dd = ((combined_equity - cum_max) / cum_max).min()
    skew = daily_returns.skew()

    # Annual Turnover
    if len(fills_df) > 0 and "filled_qty" in fills_df.columns and "avg_px" in fills_df.columns:
        traded_volume_usd = (fills_df["filled_qty"].astype(float) * fills_df["avg_px"].astype(float)).sum()
        avg_equity = combined_equity.mean()
        annual_turnover = (traded_volume_usd / avg_equity) / years if avg_equity > 0 and years > 0 else 0.0
    else:
        annual_turnover = 0.0

    engine.reset()
    engine.dispose()

    return {
        "Portfolio": portfolio_name,
        "Assets": k,
        "CAGR": f"{cagr * 100:.2f}%",
        "Realized Vol": f"{ann_vol * 100:.2f}%",
        "Sharpe": round(sharpe, 2),
        "Max Drawdown": f"{max_dd * 100:.2f}%",
        "Skewness": round(skew, 2),
        "Turnover": f"{annual_turnover:.1f}x",
        "Avg IDM": round(idm, 2),
        "Ending Balance": f"${final_capital:,.2f}",
    }


def main():
    print("=" * 80)
    print("ROBERT CARVER: NAUTILUS TRADER MULTI-ASSET PORTFOLIO SUITE")
    print("Faithful Carver Strategy Nine Replication in Event-Driven Architecture")
    print("Evaluation Window: 2020-01-01 to 2026-06-30 | Initial Capital: $100,000")
    print("=" * 80)

    catalog = ParquetDataCatalog("catalog")

    # Asset specs: (inst_id, bar_type, venue, currency)
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

    portfolios = [
        ("BTC Alone", [btc_spec], 1.00),
        ("BTC + ETH", [btc_spec, eth_spec], 1.05),
        ("Top 10 Crypto", top10_specs, 1.21),
        ("BTC + S&P 500", [btc_spec, sp500_spec], 1.23),
        ("BTC + Gold", [btc_spec, gold_spec], 1.34),
        ("BTC + Oil", [btc_spec, cl_spec], 1.35),
    ]

    results = []
    for name, specs, idm in portfolios:
        print(f"\nRunning event-driven backtest for {name} ({len(specs)} asset(s))...")
        res = run_portfolio_backtest(name, specs, idm, catalog)
        print(f"  -> CAGR: {res['CAGR']} | Sharpe: {res['Sharpe']} | Max DD: {res['Max Drawdown']} | Turnover: {res['Turnover']} | Ending: {res['Ending Balance']}")
        results.append(res)

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 115)
    print("NAUTILUS TRADER CARVER STRATEGY NINE: MULTI-ASSET PORTFOLIO SUMMARY (2020-01-01 to 2026-06-30)")
    print("=" * 115)
    print(results_df.to_string(index=False))
    print("=" * 115)

    out_csv = Path("backtests/nautilus_portfolio_summary.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved portfolio summary metrics to: {out_csv}")


if __name__ == "__main__":
    main()
