from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dirs(table_dir: Path, figure_dir: Path):
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)


def backtest_summary_row(model: str, stats: dict):
    k, c, cc, tl = stats["kupiec"], stats["christoffersen"], stats["conditional_coverage"], stats["traffic_light"]
    row = {
        "model": model,
        "observations": k["N"],
        "exceptions": k["exceptions"],
        "exception_rate": k["observed_rate"],
        "expected_exception_rate": k["expected_rate"],
        "kupiec_p": k["p_value"],
        "christoffersen_p": c["p_value"],
        "conditional_coverage_p": cc["p_value"],
    }
    if tl:
        row.update({"traffic_observations": tl["observations"], "traffic_exceptions": tl["exceptions"], "traffic_zone": tl["zone"]})
    return row


def save_core_tables(table_dir: Path, prices, dq, hs, garch, backtest_summary, mc_rows,
                     stress_row, crisis_df, liquidity_table, liquidity_breakdowns, garch_fit):
    prices.to_csv(table_dir / "prices_used.csv")
    dq.to_csv(table_dir / "data_quality.csv", index=False)
    hs.to_csv(table_dir / "hs_backtest.csv")
    garch.to_csv(table_dir / "garch_fhs_backtest.csv")
    pd.DataFrame(backtest_summary).to_csv(table_dir / "backtest_summary.csv", index=False)
    pd.DataFrame(mc_rows).to_csv(table_dir / "monte_carlo_snapshot.csv", index=False)
    pd.DataFrame([stress_row]).to_csv(table_dir / "stress_window.csv", index=False)
    crisis_df.to_csv(table_dir / "crisis_comparison.csv", index=False)
    liquidity_table.to_csv(table_dir / "liquidity_scenarios.csv", index=False)
    for name, df in liquidity_breakdowns.items():
        safe = name.lower().replace(" ", "_")
        df.to_csv(table_dir / f"liquidity_breakdown_{safe}.csv", index=False)
    pd.DataFrame([{
        "omega": garch_fit.omega, "alpha": garch_fit.alpha, "beta": garch_fit.beta,
        "persistence": garch_fit.persistence, "train_n": garch_fit.train_n,
        "optimizer_success": garch_fit.success, "message": garch_fit.message,
    }]).to_csv(table_dir / "garch_parameters.csv", index=False)


def make_figures(figure_dir: Path, pnl: pd.Series, hs: pd.DataFrame, garch: pd.DataFrame,
                 mc: dict, liquidity_table: pd.DataFrame):
    # 1. P&L and VaR
    fig, ax = plt.subplots(figsize=(12, 5))
    aligned = pd.DataFrame({"PnL": pnl}).join(hs[["var_forecast"]].rename(columns={"var_forecast":"HS_VaR"}), how="left")
    ax.plot(aligned.index, aligned["PnL"] / 1e6, linewidth=0.8, label="Daily P&L ($M)")
    ax.plot(aligned.index, -aligned["HS_VaR"] / 1e6, linewidth=1.0, label="-HS VaR ($M)")
    ax.set_title("Portfolio P&L vs 99% Historical Simulation VaR")
    ax.set_ylabel("$ millions")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(figure_dir / "hs_var_backtest.png", dpi=160); plt.close(fig)

    # 2. Model VaR comparison over overlapping history
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hs.index, hs["var_forecast"] / 1e6, linewidth=1.0, label="HS VaR")
    if not garch.empty:
        ax.plot(garch.index, garch["var_forecast"] / 1e6, linewidth=1.0, label="GARCH-FHS VaR")
    ax.set_title("99% VaR Forecasts")
    ax.set_ylabel("$ millions")
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(figure_dir / "var_model_comparison.png", dpi=160); plt.close(fig)

    # 3. Monte Carlo loss tails
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(mc["Gaussian"]["losses"] / 1e6, bins=120, density=True, alpha=0.45, label="Gaussian")
    ax.hist(mc["Student-t"]["losses"] / 1e6, bins=120, density=True, alpha=0.45, label="Student-t")
    ax.set_xlim(np.quantile(np.r_[mc["Gaussian"]["losses"], mc["Student-t"]["losses"]], [0.001, 0.999]))
    ax.set_title("Monte Carlo One-Day Loss Distribution")
    ax.set_xlabel("Loss ($M; gains are negative losses)")
    ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(figure_dir / "monte_carlo_loss_distributions.png", dpi=160); plt.close(fig)

    # 4. Liquidity scenarios
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(liquidity_table["scenario"], liquidity_table["liquidity_adjusted_var"] / 1e6)
    ax.set_title("Scenario-Based Liquidity-Adjusted VaR")
    ax.set_ylabel("$ millions")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(figure_dir / "liquidity_scenarios.png", dpi=160); plt.close(fig)
