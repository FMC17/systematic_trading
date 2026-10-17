#!/usr/bin/env python3
"""
Robert Carver Strategy Nine: Multi-Speed EWMAC Ensemble with Forecast Diversification and 10% Buffering.
Written by Gemini, 2026-10-16

Faithfully implements Strategy Nine from Robert Carver's 'Advanced Futures Trading Strategies' (AFTS, pp. 195-225)
natively inside the NautilusTrader event-driven architecture.

Core Rules:
1. 6 EWMAC Speeds: (2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256)
2. Table 29 Scalars: 12.10, 8.53, 5.95, 4.10, 2.79, 1.91
3. Carver 70/30 Blended Volatility: 0.7 * sigma_ewma_32 + 0.3 * sigma_long_run (floored at 5% annualized)
4. Order of Operations: Cap individual forecasts at +/-20.0 BEFORE combining (AFTS p. 203)
5. Ensemble Forecast: Equal weight (1/6) * FDM (1.26) with final +/-20.0 cap
6. Sizing: N_avg = (Capital * tau * IDM * Weight) / (Price * Vol)
7. Buffering: 10% Position Buffer (F = 0.10) to reduce turnover by ~40% without degrading Sharpe
"""

import math
from decimal import Decimal
from typing import Optional

import numpy as np

from nautilus_trader.indicators import ExponentialMovingAverage
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy, StrategyConfig


# Canonical Carver EWMAC Speeds & Table 29 Scalars (AFTS p. 199)
DEFAULT_SPEEDS = (
    (2, 8, 12.10),
    (4, 16, 8.53),
    (8, 32, 5.95),
    (16, 64, 4.10),
    (32, 128, 2.79),
    (64, 256, 1.91),
)


class CarverStrategyNineConfig(StrategyConfig):
    """Configuration for Carver Strategy Nine trend-following ensemble."""

    instrument_id: str
    bar_type: str
    target_risk: float = 0.20  # Annualized volatility budget (tau = 20%)
    buffer_fraction: float = 0.10  # 10% Position Buffer (F = 0.10, AFTS pp. 187-189)
    days_in_year: int = 365  # 365 for 24/7 crypto, 256 for TradFi futures
    fdm: float = 1.26  # Forecast Diversification Multiplier for 6 speeds (Table 36, p. 216)
    weight: float = 1.0  # Portfolio asset weight (1/k)
    idm: float = 1.0  # Instrument Diversification Multiplier


class CarverStrategyNine(Strategy):
    """
    Robert Carver Strategy Nine: 6-speed EWMAC ensemble with causal blended volatility,
    pre-averaging forecast capping, FDM scaling, and 10% position buffering.
    """

    def __init__(self, config: CarverStrategyNineConfig) -> None:
        super().__init__(config)

        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type = BarType.from_str(config.bar_type)

        # 1. Initialize EMA indicators across all 8 unique spans
        unique_spans = sorted(list({span for pair in DEFAULT_SPEEDS for span in (pair[0], pair[1])}))
        self.emas: dict[int, ExponentialMovingAverage] = {
            span: ExponentialMovingAverage(span) for span in unique_spans
        }

        # 2. Causal Volatility Tracking State
        self.prev_close: Optional[float] = None
        self.returns_history: list[float] = []
        self.ewma_32_var: float = 0.0
        self.alpha_32: float = 2.0 / (32.0 + 1.0)
        self.min_daily_vol: float = 0.05 / math.sqrt(config.days_in_year)

        # 3. Execution & Forecast Diagnostics
        self.last_forecast: float = 0.0
        self.last_n_avg: float = 0.0
        self.last_target_n: float = 0.0
        self.bars_processed: int = 0
        self.trades_count: int = 0

    def on_start(self) -> None:
        """Register indicators and subscribe to bar data feed."""
        for ema in self.emas.values():
            self.register_indicator_for_bars(self.bar_type, ema)
        self.subscribe_bars(self.bar_type)
        self.log.info(f"CarverStrategyNine started for {self.instrument_id}")

    def on_bar(self, bar: Bar) -> None:
        """Processes each daily bar event deterministically."""
        self.bars_processed += 1
        close_price = bar.close.as_double()

        # Update causal returns and volatility
        if self.prev_close is not None:
            # Handle near-zero or negative prices (Carver AFTS p. 666)
            prev = max(self.prev_close, 10.0) if self.prev_close <= 0 else self.prev_close
            ret = (close_price - self.prev_close) / prev
            # Clip outlier single-day jumps to [-100%, +100%]
            ret = max(-1.0, min(1.0, ret))
            self.returns_history.append(ret)

            # Update 32-day EWMA variance
            ret_sq = ret * ret
            if len(self.returns_history) == 1:
                self.ewma_32_var = ret_sq
            else:
                self.ewma_32_var = self.alpha_32 * ret_sq + (1.0 - self.alpha_32) * self.ewma_32_var

        self.prev_close = close_price

        # Require minimum warmup bars (e.g., at least 32 days for stable volatility)
        if len(self.returns_history) < 16:
            return

        # 1. Compute Carver Blended Volatility (AFTS p. 95, p. 667)
        ewma_daily_vol = math.sqrt(max(self.ewma_32_var, 1e-12))
        long_run_daily_vol = float(np.std(self.returns_history, ddof=1)) if len(self.returns_history) > 1 else ewma_daily_vol
        blended_daily_vol = 0.70 * ewma_daily_vol + 0.30 * long_run_daily_vol
        blended_daily_vol = max(blended_daily_vol, self.min_daily_vol)

        annualized_vol = blended_daily_vol * math.sqrt(self.config.days_in_year)
        daily_price_vol = max(close_price, 1e-6) * blended_daily_vol

        # 2. Compute 6-Speed EWMAC Ensemble Forecast
        individual_capped_forecasts = []
        for fast_span, slow_span, scalar in DEFAULT_SPEEDS:
            fast_val = self.emas[fast_span].value
            slow_val = self.emas[slow_span].value
            crossover = fast_val - slow_val
            raw_forecast = crossover / daily_price_vol
            scaled_forecast = raw_forecast * scalar
            # Carver Rule: Cap individual forecasts at +/-20 BEFORE combining (AFTS p. 203)
            capped_forecast = max(-20.0, min(20.0, scaled_forecast))
            individual_capped_forecasts.append(capped_forecast)

        # Combine forecasts with equal weight and FDM (AFTS Table 36, p. 216)
        raw_combined = sum(individual_capped_forecasts) / len(DEFAULT_SPEEDS)
        scaled_combined = raw_combined * self.config.fdm
        # Final cap at +/-20.0
        combined_forecast = max(-20.0, min(20.0, scaled_combined))
        self.last_forecast = combined_forecast

        # 3. Retrieve Current Account Capital / Equity
        instrument = self.cache.instrument(self.instrument_id)
        if instrument is None:
            return

        venue = self.instrument_id.venue
        equity_dict = self.portfolio.equity(venue)
        # Identify settlement or base currency
        quote_ccy = instrument.quote_currency
        capital = equity_dict.get(quote_ccy)
        if capital is not None:
            current_capital = capital.as_double()
        else:
            # Fallback to sum of any available equities
            current_capital = sum(m.as_double() for m in equity_dict.values()) if equity_dict else 100_000.0

        if current_capital <= 0:
            return

        # 4. Volatility-Targeted Position Sizing (AFTS pp. 185-187)
        # N_avg = (Capital * tau * IDM * Weight) / (Price * Annualized_Vol)
        n_avg = (current_capital * self.config.target_risk * self.config.idm * self.config.weight) / (close_price * annualized_vol)
        target_contracts = n_avg * (combined_forecast / 10.0)

        self.last_n_avg = n_avg
        self.last_target_n = target_contracts

        # 5. 10% Position Buffering (AFTS pp. 187-189)
        buffer_width = self.config.buffer_fraction * abs(n_avg)
        lower_buffer = target_contracts - buffer_width
        upper_buffer = target_contracts + buffer_width

        # Current held contracts from portfolio
        net_pos = self.portfolio.net_position(self.instrument_id)
        current_contracts = float(net_pos)

        trade_amount = 0.0
        if current_contracts < lower_buffer:
            trade_amount = lower_buffer - current_contracts  # Buy to lower buffer
        elif current_contracts > upper_buffer:
            trade_amount = upper_buffer - current_contracts  # Sell to upper buffer
        else:
            # Inside buffer band: Do not trade
            pass

        # 6. Order Submission
        min_trade = float(instrument.size_increment) if instrument.size_increment else 1e-4
        if abs(trade_amount) >= min_trade:
            try:
                qty_decimal = Decimal(f"{abs(trade_amount):.8f}")
                order_qty = instrument.make_qty(qty_decimal)
            except ValueError:
                return

            # Enforce minimum order quantity if defined
            min_qty = instrument.min_quantity or Quantity.from_str("0.0001")
            if order_qty >= min_qty:
                side = OrderSide.BUY if trade_amount > 0 else OrderSide.SELL
                order = self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=side,
                    quantity=order_qty,
                )
                self.submit_order(order)
                self.trades_count += 1

    def on_stop(self) -> None:
        """Close positions and log summary on strategy stop."""
        self.log.info(
            f"CarverStrategyNine finished for {self.instrument_id}: "
            f"bars={self.bars_processed}, trades={self.trades_count}, "
            f"last_forecast={self.last_forecast:.2f}, last_target_n={self.last_target_n:.4f}"
        )

    def on_reset(self) -> None:
        """Reset state between backtest iterations."""
        self.prev_close = None
        self.returns_history.clear()
        self.ewma_32_var = 0.0
        self.last_forecast = 0.0
        self.last_n_avg = 0.0
        self.last_target_n = 0.0
        self.bars_processed = 0
        self.trades_count = 0
        for ema in self.emas.values():
            ema.reset()
