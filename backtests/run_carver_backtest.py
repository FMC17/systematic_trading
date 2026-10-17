#!/usr/bin/env python3
"""
Robert Carver Strategy Nine Event-Driven Backtest Runner.
Written by Gemini, 2026-10-16

Executes Robert Carver Strategy Nine inside the NautilusTrader BacktestEngine,
sourcing historical bars directly from ParquetDataCatalog.

Compares event-driven execution with our pure Python benchmark in CARVER_REPLICATION/.
"""

import argparse
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

# Ensure systematic_trading root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import MakerTakerFeeModel
from nautilus_trader.config import RiskEngineConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, TraderId, Venue
from nautilus_trader.model.objects import Currency, Money
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from strategies.carver_strategy_nine import CarverStrategyNine, CarverStrategyNineConfig


BINANCE = Venue("BINANCE")
USDT = Currency.from_str("USDT")
USD = Currency.from_str("USD")


def run_carver_backtest(
    instrument_id_str: str = "BTCUSDT-PERP.BINANCE",
    bar_type_str: str = "BTCUSDT-PERP.BINANCE-1-DAY-LAST-EXTERNAL",
    initial_capital: float = 100_000.0,
    target_risk: float = 0.20,
    buffer_fraction: float = 0.10,
    catalog_path: str = "catalog",
) -> None:
    print("=" * 80)
    print("NAUTILUS TRADER: ROBERT CARVER STRATEGY NINE EVENT-DRIVEN BACKTEST")
    print(f"Instrument: {instrument_id_str} | Risk Target: {target_risk * 100:.1f}% | Buffer: {buffer_fraction * 100:.1f}%")
    print(f"Initial Capital: ${initial_capital:,.2f} | Catalog: {catalog_path}")
    print("=" * 80)

    # 1. Initialize ParquetDataCatalog
    catalog = ParquetDataCatalog(catalog_path)
    instrument_id = InstrumentId.from_str(instrument_id_str)
    bar_type = BarType.from_str(bar_type_str)

    instruments = catalog.instruments(instrument_ids=[str(instrument_id)])
    if not instruments:
        raise ValueError(f"Instrument {instrument_id} not found in catalog {catalog_path}")
    instrument = instruments[0]

    bars = catalog.bars([bar_type])
    if not bars:
        raise ValueError(f"No bars found for {bar_type} in catalog {catalog_path}")
    print(f"Loaded {len(bars)} historical daily bars from catalog.")

    # 2. Configure Backtest Engine
    engine = BacktestEngine(
        BacktestEngineConfig(
            trader_id=TraderId("CARVER-001"),
            risk_engine=RiskEngineConfig(bypass=True),
        )
    )

    # 3. Add Venue & Margin Account
    settlement_currency = instrument.quote_currency if hasattr(instrument, "quote_currency") else USD
    starting_balance = Money(Decimal(str(initial_capital)), settlement_currency)

    engine.add_venue(
        venue=instrument_id.venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=settlement_currency,
        starting_balances=[starting_balance],
        fee_model=MakerTakerFeeModel(),
    )

    # 4. Add Instrument & Market Data
    engine.add_instrument(instrument)
    engine.add_data(bars)

    # 5. Instantiate Strategy
    config = CarverStrategyNineConfig(
        instrument_id=instrument_id_str,
        bar_type=bar_type_str,
        target_risk=target_risk,
        buffer_fraction=buffer_fraction,
        days_in_year=365 if instrument_id.venue == BINANCE else 256,
        fdm=1.26,
        weight=1.0,
        idm=1.0,
    )
    strategy = CarverStrategyNine(config)
    engine.add_strategy(strategy)

    # 6. Execute Deterministic Event-Driven Simulation
    print("\nExecuting backtest simulation...")
    engine.run()

    # 7. Print Performance Reports
    with pd.option_context("display.max_rows", 100, "display.max_columns", None, "display.width", 300):
        print("\n" + "=" * 80)
        print("ACCOUNT PERFORMANCE REPORT")
        print("=" * 80)
        print(engine.trader.generate_account_report(instrument_id.venue))

        print("\n" + "=" * 80)
        print("POSITIONS REPORT")
        print("=" * 80)
        print(engine.trader.generate_positions_report())

        print("\n" + "=" * 80)
        print("ORDER FILLS REPORT (Sample)")
        print("=" * 80)
        fills = engine.trader.generate_order_fills_report()
        print(fills.head(15))
        if len(fills) > 15:
            print(f"... ({len(fills) - 15} more fills)")

    # 8. Clean up
    engine.reset()
    engine.dispose()
    print("\nBacktest complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Carver Strategy Nine in NautilusTrader")
    parser.add_argument("--instrument", type=str, default="BTCUSDT-PERP.BINANCE", help="Instrument ID")
    parser.add_argument("--bar-type", type=str, default="BTCUSDT-PERP.BINANCE-1-DAY-LAST-EXTERNAL", help="Bar type string")
    parser.add_argument("--capital", type=float, default=100_000.0, help="Starting capital")
    parser.add_argument("--tau", type=float, default=0.20, help="Target risk (default 0.20)")
    parser.add_argument("--buffer", type=float, default=0.10, help="Buffer fraction (default 0.10)")
    parser.add_argument("--catalog", type=str, default="catalog", help="Catalog path")
    args = parser.parse_args()

    run_carver_backtest(
        instrument_id_str=args.instrument,
        bar_type_str=args.bar_type,
        initial_capital=args.capital,
        target_risk=args.tau,
        buffer_fraction=args.buffer,
        catalog_path=args.catalog,
    )
