"""Analyze political speech mention market accuracy and calibration."""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.kalshi.util.categories import CATEGORY_SQL, get_group
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class PoliticalSpeechMentionAccuracyAnalysis(Analysis):
    """Analyze Kalshi prediction market accuracy on political speech mentions."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="political_speech_mention_accuracy",
            description="Overall accuracy, calibration, and breakdowns for political speech mention markets",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Step 1: Identify Target Markets & Extract Text Attributes
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
                  AND (ticker LIKE '%MENTION%' OR event_ticker LIKE '%MENTION%')
            ),
            political_markets AS (
                SELECT * FROM market_base
                -- pandas filtering later handles the 'Politics' group precisely, 
                -- but we filter out obviously non-political ones here where possible if desired.
            ),
            relevant_trades AS (
                SELECT 
                    t.ticker,
                    -- Yes VWAP
                    SUM(t.yes_price * t.count) / NULLIF(SUM(t.count), 0) as vwap_yes,
                    SUM(t.count) as total_volume
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN political_markets m ON t.ticker = m.ticker
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
            FROM political_markets m
            INNER JOIN relevant_trades r ON m.ticker = r.ticker
            WHERE r.vwap_yes IS NOT NULL
            """
        ).df()

        # Filter strictly for political mention markets
        df["group"] = df["category"].apply(get_group)
        df = df[df["group"] == "Politics"].copy()

        # Parse speaker and event from title
        # Case 1: "What will Donald Trump say during Working Breakfast with Governors?"
        # Pattern: What will (.+?) say during (.+)\?
        pat1 = r"What will (.+?) say during (.+?)\?"
        
        # Case 2: "Will the White House Press Secretary say Economy at her next press briefing?"
        pat2 = r"Will the (White House Press Secretary) say .+ at (her next press briefing)\?"
        
        # We extract speaker and event using regex
        speaker1 = df["title"].str.extract(pat1, expand=True)[0]
        event1 = df["title"].str.extract(pat1, expand=True)[1]
        
        speaker2 = df["title"].str.extract(pat2, expand=True)[0]
        event2 = df["title"].str.extract(pat2, expand=True)[1]
        
        df["speaker"] = speaker1.fillna(speaker2).fillna("Unknown")
        df["event"] = event1.fillna(event2).fillna("Unknown")

        # Normalize probabilities to 0-1 scale instead of 1-99 cents
        df["pre_speech_prob"] = df["pre_speech_prob"] / 100.0
        
        # Overall Accuracy
        # If prob > 0.5 and result == 'yes', correct. If prob < 0.5 and result == 'no', correct.
        df["predicted_yes"] = df["pre_speech_prob"] > 0.5
        df["actual_yes"] = df["result"] == "yes"
        df["correct"] = df["predicted_yes"] == df["actual_yes"]
        
        overall_accuracy = df["correct"].mean() if not df.empty else 0.0
        overall_resolution_rate = df["actual_yes"].mean() if not df.empty else 0.0

        print(f"Overall Accuracy: {overall_accuracy:.2%}")
        print(f"Overall Resolution Rate (Yes): {overall_resolution_rate:.2%}")

        # Calibration Buckets
        bins = np.linspace(0, 1.0, 11) # 0.0, 0.1, ..., 1.0
        labels = [f"{int(b*100)}-{int((b+0.1)*100)}%" for b in bins[:-1]]
        
        df["prob_bucket"] = pd.cut(df["pre_speech_prob"], bins=bins, labels=labels, include_lowest=True)
        
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
        speaker_breakdown = df.groupby("speaker").agg(
            markets=("ticker", "count"),
            accuracy=("correct", "mean")
        ).reset_index().sort_values("markets", ascending=False).head(10)
        
        event_breakdown = df.groupby("event").agg(
            markets=("ticker", "count"),
            accuracy=("correct", "mean")
        ).reset_index().sort_values("markets", ascending=False).head(10)

        # Build output data
        print("\n--- Speaker Breakdown (Top 10) ---")
        print(speaker_breakdown.to_string(index=False))
        print("\n--- Event Breakdown (Top 10) ---")
        print(event_breakdown.to_string(index=False))
        
        # Build figure
        fig = self._create_figure(calibration)
        
        # Build chart
        chart = self._create_chart(calibration)
        
        # Return merged outputs in data
        # To conform to AnalysisOutput taking a DataFrame for `data`, we return the calibration table
        return AnalysisOutput(figure=fig, data=calibration, chart=chart)

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
        
        ax.set_xlabel("Market Implied Probability (2 hours before speech)")
        ax.set_ylabel("Actual % Resolved 'Yes'")
        ax.set_title("Calibration Curve for Political Speech Mention Markets")
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
            type=ChartType.BAR, # Line+Bar combo if possible, using BAR as base
            data=chart_data,
            xKey="bucket",
            yKeys=["actual_yes", "ideal"],
            title="Calibration of Political Speech Mention Markets",
            yUnit=UnitType.PERCENT,
            xLabel="Market Prediction Confidence",
            yLabel="Actual Resolution Rate",
            colors={"actual_yes": "#1f77b4", "ideal": "#ff0000"}
        )
