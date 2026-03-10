"""Analyze NBA game market accuracy and calibration."""
from __future__ import annotations
from pathlib import Path
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.analysis.kalshi.util.categories import CATEGORY_SQL, get_group
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType
class NbaGameAccuracyAnalysis(Analysis):
    """Analyze Kalshi prediction market accuracy on NBA games."""
    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="nba_game_accuracy",
            description="Overall accuracy, calibration, and breakdowns for NBA game markets",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")
    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()
        # Step 1: Identify Target Markets
        # Step 2: Calculate VWAP from trades up to 2 hours before market closed
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
                  AND ticker LIKE 'KXNBAGAME-%'
            ),
            relevant_trades AS (
                SELECT 
                    t.ticker,
                    -- Yes VWAP
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
                r.vwap_yes as pre_game_prob,
                r.total_volume
            FROM market_base m
            INNER JOIN relevant_trades r ON m.ticker = r.ticker
            WHERE r.vwap_yes IS NOT NULL
            """
        ).df()
        # Extract team from ticker (format: KXNBAGAME-DATE-MATCHUP-TEAM)
        df["team"] = df["ticker"].str.split("-").str[-1]
        # Normalize probabilities to 0-1 scale instead of 1-99 cents
        df["pre_game_prob"] = df["pre_game_prob"] / 100.0
        
        # Overall Accuracy
        # If prob > 0.5 and result == 'yes', correct. If prob < 0.5 and result == 'no', correct.
        df["predicted_yes"] = df["pre_game_prob"] > 0.5
        df["actual_yes"] = df["result"] == "yes"
        df["correct"] = df["predicted_yes"] == df["actual_yes"]
        
        overall_accuracy = df["correct"].mean() if not df.empty else 0.0
        overall_resolution_rate = df["actual_yes"].mean() if not df.empty else 0.0
        print(f"Overall Accuracy: {overall_accuracy:.2%}")
        print(f"Overall Resolution Rate (Yes): {overall_resolution_rate:.2%}")
        # Calibration Buckets
        bins = np.linspace(0, 1.0, 11) # 0.0, 0.1, ..., 1.0
        labels = [f"{int(b*100)}-{int((b+0.1)*100)}%" for b in bins[:-1]]
        
        df["prob_bucket"] = pd.cut(df["pre_game_prob"], bins=bins, labels=labels, include_lowest=True)
        
        calibration = df.groupby("prob_bucket", observed=False).agg(
            total=("ticker", "count"),
            actual_yes_count=("actual_yes", "sum")
        ).reset_index()
        
        calibration["actual_yes_pct"] = np.where(
            calibration["total"] > 0, 
            calibration["actual_yes_count"] / calibration["total"], 
            np.nan
        )
        # Breakdowns
        team_breakdown = df.groupby("team").agg(
            markets=("ticker", "count"),
            accuracy=("correct", "mean")
        ).reset_index().sort_values("markets", ascending=False).head(15)
        # Build output data
        print("\n--- Team Breakdown (Top 15) ---")
        print(team_breakdown.to_string(index=False))
        
        # Build markdown text
        markdown_text = f"""# NBA Game Market Accuracy

## Overall Performance
* **Overall Accuracy**: {overall_accuracy:.2%}
* **Overall Resolution Rate**: {overall_resolution_rate:.2%}

## Team Breakdown (Top 15)
{team_breakdown.to_markdown(index=False)}
"""
        
        # Build figure
        fig = self._create_figure(calibration)
        
        # Build chart
        chart = self._create_chart(calibration)
        
        # Return merged outputs in data
        return AnalysisOutput(figure=fig, data=calibration, chart=chart, markdown=markdown_text)
    def _create_figure(self, calibration: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cal_plot = calibration.dropna(subset=["actual_yes_pct"])
        x = np.arange(len(cal_plot))
        
        ax.bar(x, cal_plot["actual_yes_pct"] * 100, color="#1f77b4", edgecolor="black", alpha=0.7)
        
        # Perfect calibration line
        ideal_cal = np.linspace(5, 95, 10) # midpoints of bins: 5%, 15%... 95%
        # Filter ideal_cal to match dropped NAs
        valid_indices = cal_plot.index.tolist()
        ideal_cal_filtered = [ideal_cal[i] for i in valid_indices]
        
        ax.plot(x, ideal_cal_filtered, marker="o", color="red", linestyle="--", label="Perfect Calibration")
        
        ax.set_xticks(x)
        ax.set_xticklabels(cal_plot["prob_bucket"], rotation=45, ha="right")
        
        ax.set_xlabel("Market Implied Probability (2 hours before game)")
        ax.set_ylabel("Actual % Resolved 'Yes'")
        ax.set_title("Calibration Curve for NBA Game Markets")
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    def _create_chart(self, calibration: pd.DataFrame) -> ChartConfig:
        chart_data = []
        # Calculate midpoints for ideal calibration
        midpoints = np.linspace(5, 95, 10)
        for i, row in calibration.iterrows():
            if pd.notna(row["actual_yes_pct"]):
                chart_data.append({
                    "bucket": row["prob_bucket"],
                    "actual_yes": float(round(row["actual_yes_pct"] * 100, 1)),
                    "ideal": float(round(midpoints[i], 1))
                })
                
        return ChartConfig(
            type=ChartType.BAR, 
            data=chart_data,
            xKey="bucket",
            yKeys=["actual_yes", "ideal"],
            title="Calibration of NBA Game Markets",
            yUnit=UnitType.PERCENT,
            xLabel="Market Prediction Confidence",
            yLabel="Actual Resolution Rate",
            colors={"actual_yes": "#1f77b4", "ideal": "#ff0000"}
        )
