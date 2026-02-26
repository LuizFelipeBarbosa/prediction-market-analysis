"""Analyze Kalshi political speech mentions by sub-group and strategy profitability."""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.kalshi.util.categories import CATEGORY_SQL, get_group
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class SpeechMentionSubgroupAccuracyAnalysis(Analysis):
    """Analyze calibration and EV of political speech mention markets by subgroup."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="speech_mention_subgroup_accuracy",
            description="Calibration curves and profitability calculations for political speech mention subgroups",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def _kalshi_fee(self, price_cents: float) -> int:
        """Calculate Kalshi taker fee per contract in cents based on price."""
        return math.ceil(0.07 * price_cents * (100.0 - price_cents) / 100.0)

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Get base markets and 2-hour VWAPs
        df = con.execute(
            f"""
            WITH market_base AS (
                SELECT 
                    ticker,
                    title,
                    result,
                    close_time,
                    {CATEGORY_SQL} AS category
                FROM '{self.markets_dir}/*.parquet'
                WHERE result IN ('yes', 'no')
                  AND (ticker LIKE '%MENTION%' OR event_ticker LIKE '%MENTION%')
            ),
            relevant_trades AS (
                SELECT 
                    t.ticker,
                    SUM(t.yes_price * t.count) / NULLIF(SUM(t.count), 0) as vwap_yes,
                    SUM(t.count) as total_volume
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN market_base m ON t.ticker = m.ticker
                WHERE t.created_time <= m.close_time - INTERVAL 2 HOUR
                GROUP BY t.ticker
            )
            SELECT 
                m.ticker,
                m.title,
                m.result,
                m.category,
                r.vwap_yes as pre_speech_prob,
                r.total_volume
            FROM market_base m
            INNER JOIN relevant_trades r ON m.ticker = r.ticker
            WHERE r.vwap_yes IS NOT NULL
            """
        ).df()

        # Strict political filtering
        df["group_cat"] = df["category"].apply(get_group)
        df = df[df["group_cat"] == "Politics"].copy()

        # Parse speaker and event
        pat1 = r"What will (.+?) say during (.+?)\?"
        pat2 = r"Will the (White House Press Secretary) say .+ at (her next press briefing)\?"
        
        # New pattern for Mayoral announcements: 
        # "Will Zohran Mamdani say ... at his next NYC Mayor's Office announcement?"
        pat3 = r"Will ([^s]+ [^s]+) say .+ at ((?:his|her) next NYC Mayor\'s Office announcement)\?"
        
        speaker1 = df["title"].str.extract(pat1, expand=True)[0]
        event1 = df["title"].str.extract(pat1, expand=True)[1]
        
        speaker2 = df["title"].str.extract(pat2, expand=True)[0]
        event2 = df["title"].str.extract(pat2, expand=True)[1]

        speaker3 = df["title"].str.extract(pat3, expand=True)[0]
        event3 = df["title"].str.extract(pat3, expand=True)[1]
        
        df["speaker"] = speaker1.fillna(speaker2).fillna(speaker3).fillna("Unknown")
        df["event"] = event1.fillna(event2).fillna(event3).fillna("Unknown")

        # Map logic to subgroups
        def assign_subgroup(row):
            if row["speaker"] in ["Donald Trump", "Trump", "President Trump"]:
                return "Group 1: Trump Markets"
            elif row["speaker"] in ["White House Press Secretary", "Zohran Mamdani"] or "NYC Mayor's Office" in row["event"]:
                return "Group 2: Press & Mayoral"
            else:
                return "Group 3: Niche / One-offs"

        df["subgroup"] = df.apply(assign_subgroup, axis=1)
        
        df["actual_yes"] = df["result"] == "yes"
        df["predicted_yes"] = df["pre_speech_prob"] > 50.0  # pre_speech_prob is in cents here (1-99)
        df["correct"] = df["predicted_yes"] == df["actual_yes"]

        # Expected Profit calculation
        # Buy Yes > 70%, Buy No < 30%
        def calc_trade_profit(row):
            price = row["pre_speech_prob"]  # cents
            fee = self._kalshi_fee(price)
            resolved_yes = row["actual_yes"]
            
            if price > 70.0:
                cost = price + fee
                payout = 100.0 if resolved_yes else 0.0
                return pd.Series({"traded": 1, "cost": cost, "payout": payout})
            elif price < 30.0:
                cost = (100.0 - price) + fee
                payout = 100.0 if not resolved_yes else 0.0
                return pd.Series({"traded": 1, "cost": cost, "payout": payout})
            else:
                return pd.Series({"traded": 0, "cost": 0.0, "payout": 0.0})

        trade_stats = df.apply(calc_trade_profit, axis=1)
        df["traded"] = trade_stats["traded"]
        df["trade_cost"] = trade_stats["cost"]
        df["trade_payout"] = trade_stats["payout"]

        # Run subgroup aggregations
        subgroups = ["Group 1: Trump Markets", "Group 2: Press & Mayoral", "Group 3: Niche / One-offs"]
        
        bins = np.linspace(0, 100, 11) # 0, 10, ..., 100
        labels = [f"{(b)}-{(b+10)}%" for b in bins[:-1]]
        
        df["prob_bucket"] = pd.cut(df["pre_speech_prob"], bins=bins, labels=labels, include_lowest=True)
        
        profit_breakdown = []
        calibration_dfs = {}
        
        for g in subgroups:
            g_df = df[df["subgroup"] == g]
            acc = g_df["correct"].mean()
            tot = len(g_df)
            
            # Calibration
            cal = g_df.groupby("prob_bucket", observed=False).agg(
                n=("ticker", "count"),
                actual_yes_count=("actual_yes", "sum")
            ).reset_index()
            
            cal["actual_yes_pct"] = np.where(cal["n"] > 0, cal["actual_yes_count"] / cal["n"], np.nan)
            calibration_dfs[g] = cal
            
             # EV Profitability
            g_traded_df = g_df[g_df["traded"] == 1]
            total_cost = g_traded_df["trade_cost"].sum()
            total_payout = g_traded_df["trade_payout"].sum()
            
            expected_profit_per_dollar = (total_payout - total_cost) / total_cost if total_cost > 0 else 0.0
            profit_breakdown.append({
                "subgroup": g,
                "total_markets": tot,
                "accuracy": acc,
                "total_extremes_traded": len(g_traded_df),
                "profit_per_dollar": expected_profit_per_dollar
            })

        profit_df = pd.DataFrame(profit_breakdown)
        print("\n--- Buy Extremes Strategy Expected Profit ---")
        print(profit_df.to_string(index=False))

        # Generate figures and charts
        fig, axes = self._create_figure(calibration_dfs, profit_df)
        
        # Combine all calibration dfs into one large chart config if returning to web, or just the profit table
        # We'll return the profit breakdown table as the `data` for the UI
        chart = self._create_chart(profit_df)

        return AnalysisOutput(figure=fig, data=profit_df, chart=chart)

    def _create_figure(self, calibration_dfs: dict, profit_df: pd.DataFrame) -> tuple[plt.Figure, any]:
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        
        groups = ["Group 1: Trump Markets", "Group 2: Press & Mayoral", "Group 3: Niche / One-offs"]
        colors = ["#d62728", "#1f77b4", "#ff7f0e"]
        
        for i, g in enumerate(groups):
            ax = axes[i]
            cal = calibration_dfs[g]
            
            p_info = profit_df[profit_df["subgroup"] == g].iloc[0]
            tot_markets = p_info["total_markets"]
            acc = float(p_info["accuracy"] * 100)
            
            x = np.arange(len(cal))
            bars = ax.bar(x, cal["actual_yes_pct"].fillna(0) * 100, color=colors[i], edgecolor="black", alpha=0.7)
            
            # Perfect calibration line
            ideal_cal = np.linspace(5, 95, 10) 
            ax.plot(x, ideal_cal, marker="o", color="black", linestyle="--", label="Perfect Calibration")
            
            # Add N counts above bars
            for j, bar in enumerate(bars):
                n_count = cal.iloc[j]["n"]
                if n_count > 0:
                    yval = bar.get_height()
                    
                    # Flag n < 10 for Group 3
                    if g == "Group 3: Niche / One-offs" and n_count < 10:
                        ax.text(bar.get_x() + bar.get_width() / 2, yval + 2, f"n={n_count}\n(warn)", 
                                ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
                    else:
                        ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"n={n_count}", 
                                ha='center', va='bottom', fontsize=9)

            ax.set_xticks(x)
            ax.set_xticklabels(cal["prob_bucket"], rotation=45, ha="right")
            ax.set_ylim(0, 110)
            
            ax.set_xlabel("Market Implied Probability (2 hours before speech)")
            ax.set_ylabel("Actual % Resolved 'Yes'")
            ax.set_title(f"{g}\nTotal Markets: {tot_markets} | Overall Accuracy: {acc:.1f}%")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig, axes

    def _create_chart(self, profit_df: pd.DataFrame) -> ChartConfig:
        chart_data = []
        for _, row in profit_df.iterrows():
            chart_data.append({
                "subgroup": row["subgroup"],
                "profit": float(round(row["profit_per_dollar"] * 100, 2)),
                "accuracy": float(round(row["accuracy"] * 100, 2))
            })
                
        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="subgroup",
            yKeys=["profit", "accuracy"],
            title="Buy Extremes Expected Profit vs Overall Accuracy",
            yUnit=UnitType.PERCENT,
            xLabel="Speech Mention Subgroup",
            yLabel="Percentage (%)",
            colors={"profit": "#2ca02c", "accuracy": "#1f77b4"}
        )
