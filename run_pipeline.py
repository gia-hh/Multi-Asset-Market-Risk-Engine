from __future__ import annotations

import argparse
import pandas as pd

from src.config import (PORTFOLIO, PORTFOLIO_VALUE, START_DATE, END_DATE, CRISIS_PERIODS,
                        BASE_LIQUIDITY_ASSUMPTIONS, LIQUIDITY_SCENARIOS, TABLE_DIR, FIGURE_DIR, RunConfig)
from src.data import fetch_live_prices, generate_synthetic_prices, compute_log_returns, data_quality_summary
from src.portfolio import portfolio_series
from src.historical_var import rolling_hs_forecasts
from src.garch_fhs import fit_garch11_initial, rolling_garch_fhs
from src.monte_carlo import monte_carlo_snapshot
from src.backtesting import evaluate_forecasts
from src.liquidity import liquidity_adjusted_table
from src.stress import worst_volatility_window, crisis_comparison
from src.reporting import ensure_dirs, backtest_summary_row, save_core_tables, make_figures


def run(data_mode: str = "synthetic"):
    cfg = RunConfig()
    ensure_dirs(TABLE_DIR, FIGURE_DIR)
    tickers = list(PORTFOLIO)

    if data_mode == "live":
        prices = fetch_live_prices(
            tickers,
            START_DATE,
            END_DATE,
        )
        data_source = "Yahoo Finance via yfinance"
        requested_end = (
            END_DATE
            if END_DATE is not None
            else "latest available at runtime"
        )

    elif data_mode == "synthetic":
        prices = generate_synthetic_prices(tickers)
        data_source = "deterministic synthetic fixture"
        requested_end = "synthetic fixture end"

    else:
        raise ValueError(
            "data_mode must be 'live' or 'synthetic'"
        )

    returns = compute_log_returns(prices)
    dq = data_quality_summary(prices, returns)

    if returns.empty:
        raise RuntimeError(
            "No aligned portfolio return observations were produced."
        )

    actual_price_start = prices.dropna(how="any").index.min()
    actual_price_end = prices.dropna(how="any").index.max()

    aligned_start = returns.index.min()
    aligned_end = returns.index.max()

    run_metadata = pd.DataFrame([{
        "run_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "data_mode": data_mode,
        "data_source": data_source,
        "requested_start": START_DATE,
        "requested_end": requested_end,
        "actual_common_price_start": actual_price_start.date().isoformat(),
        "actual_common_price_end": actual_price_end.date().isoformat(),
        "aligned_return_start": aligned_start.date().isoformat(),
        "aligned_return_end": aligned_end.date().isoformat(),
        "aligned_return_observations": len(returns),
    }])

    run_metadata.to_csv(
        TABLE_DIR / "run_metadata.csv",
        index=False,
    )
    port_ret, pnl, loss, weights = portfolio_series(returns, PORTFOLIO, PORTFOLIO_VALUE)

    hs = rolling_hs_forecasts(loss, cfg.hs_lookback, cfg.confidence)
    hs_eval, hs_stats = evaluate_forecasts(hs, cfg.confidence, cfg.traffic_window)

    gfit = fit_garch11_initial(
    port_ret,
    cfg.garch_train_window,
)

    if not gfit.success:
        raise RuntimeError(
            "GARCH(1,1) optimization did not converge. "
            "The pipeline will not use fallback parameters for "
            "recruiter-facing market results. "
            f"Optimizer message: {gfit.message}"
        )

    garch = rolling_garch_fhs(
        port_ret,
        PORTFOLIO_VALUE,
        gfit,
        cfg.garch_resid_window,
        cfg.confidence,
    )
    garch_eval, garch_stats = evaluate_forecasts(garch, cfg.confidence, cfg.traffic_window)

    mc = monte_carlo_snapshot(returns, weights, PORTFOLIO_VALUE, cfg.confidence, cfg.mc_lookback,
                              cfg.mc_simulations, cfg.student_t_df, cfg.mc_seed)
    mc_rows = [
        {"model": "Gaussian Monte Carlo", "VaR_99": mc["Gaussian"]["VaR"], "ES_99": mc["Gaussian"]["ES"],
         "lookback_observations": mc["lookback_observations"], "student_t_df": None},
        {"model": "Student-t Monte Carlo", "VaR_99": mc["Student-t"]["VaR"], "ES_99": mc["Student-t"]["ES"],
         "lookback_observations": mc["lookback_observations"], "student_t_df": mc["student_t_df"]},
    ]

    stress = worst_volatility_window(port_ret, PORTFOLIO_VALUE, 250, cfg.confidence)
    crisis_df = crisis_comparison(port_ret, hs_eval, garch_eval, CRISIS_PERIODS, PORTFOLIO_VALUE)

    base_var = float(hs_eval["var_forecast"].iloc[-1])
    liq_table, liq_breakdowns = liquidity_adjusted_table(base_var, PORTFOLIO, PORTFOLIO_VALUE,
                                                          BASE_LIQUIDITY_ASSUMPTIONS, LIQUIDITY_SCENARIOS)
    backtest_summary = [backtest_summary_row("Historical Simulation", hs_stats),
                        backtest_summary_row("GARCH-FHS", garch_stats)]

    save_core_tables(TABLE_DIR, prices, dq, hs_eval, garch_eval, backtest_summary, mc_rows,
                     stress, crisis_df, liq_table, liq_breakdowns, gfit)
    make_figures(FIGURE_DIR, pnl, hs_eval, garch_eval, mc, liq_table)

    print("\nMulti-Asset Market Risk Engine")
    print("=" * 58)

    print(f"Data mode: {data_mode}")
    print(f"Data source: {data_source}")
    print(f"Requested start: {START_DATE}")
    print(f"Requested end: {requested_end}")

    print(
        "Actual common price period: "
        f"{actual_price_start.date()} to "
        f"{actual_price_end.date()}"
    )

    print(
        "Aligned return period: "
        f"{aligned_start.date()} to "
        f"{aligned_end.date()}"
    )

    print(
        f"Aligned return observations: "
        f"{len(returns):,}"
    )

    print(
        f"HS backtest observations: "
        f"{len(hs_eval):,}"
    )

    print(
        f"GARCH-FHS backtest observations: "
        f"{len(garch_eval):,}"
    )

    print(
        f"Latest HS VaR 99%: "
        f"${base_var:,.0f}"
    )

    print(
        f"Latest HS ES 99%:  "
        f"${hs_eval['expected_shortfall'].iloc[-1]:,.0f}"
    )

    print(
        f"GARCH optimizer success: "
        f"{gfit.success}"
    )

    print(
        f"GARCH optimizer message: "
        f"{gfit.message}"
    )

    print(
        f"GARCH persistence: "
        f"{gfit.persistence:.4f}"
    )

    print(
        "Stress window: "
        f"{pd.Timestamp(stress['start']).date()} "
        "to "
        f"{pd.Timestamp(stress['end']).date()}"
    )

    print(
        f"Stress VaR 99%: "
        f"${stress['stress_var']:,.0f}"
    )

    print(f"Outputs: {TABLE_DIR.parent}")



    if data_mode == "live":
        print(
            "NOTE: live mode downloads real historical daily "
            "market data at runtime; it is not an intraday "
            "real-time market-data feed."
        )
    if data_mode == "synthetic":
        print("NOTE: synthetic mode validates the software only; do not use its metrics as market evidence.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Asset Market Risk Engine")
    parser.add_argument("--data-mode", choices=["live", "synthetic"], default="synthetic",
                        help="live downloads public ETF data; synthetic runs an offline deterministic fixture")
    args = parser.parse_args()
    run(args.data_mode)
