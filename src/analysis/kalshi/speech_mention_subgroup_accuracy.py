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

        # Get base markets and VWAPs at different intervals before close
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
            trade_intervals AS (
                SELECT 
                    t.ticker,
                    m.close_time,
                    t.created_time,
                    t.yes_price,
                    t.count
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN market_base m ON t.ticker = m.ticker
            ),
            vwaps AS (
                SELECT 
                    ticker,
                    SUM(CASE WHEN created_time <= close_time - INTERVAL 7 DAY THEN yes_price * count ELSE 0 END) / 
                        NULLIF(SUM(CASE WHEN created_time <= close_time - INTERVAL 7 DAY THEN count ELSE 0 END), 0) as vwap_7d,
                    
                    SUM(CASE WHEN created_time <= close_time - INTERVAL 3 DAY THEN yes_price * count ELSE 0 END) / 
                        NULLIF(SUM(CASE WHEN created_time <= close_time - INTERVAL 3 DAY THEN count ELSE 0 END), 0) as vwap_3d,
                    
                    SUM(CASE WHEN created_time <= close_time - INTERVAL 1 DAY THEN yes_price * count ELSE 0 END) / 
                        NULLIF(SUM(CASE WHEN created_time <= close_time - INTERVAL 1 DAY THEN count ELSE 0 END), 0) as vwap_1d,
                    
                    SUM(CASE WHEN created_time <= close_time - INTERVAL 6 HOUR THEN yes_price * count ELSE 0 END) / 
                        NULLIF(SUM(CASE WHEN created_time <= close_time - INTERVAL 6 HOUR THEN count ELSE 0 END), 0) as vwap_6h,
                    
                    SUM(CASE WHEN created_time <= close_time - INTERVAL 2 HOUR THEN yes_price * count ELSE 0 END) / 
                        NULLIF(SUM(CASE WHEN created_time <= close_time - INTERVAL 2 HOUR THEN count ELSE 0 END), 0) as vwap_2h,
                    
                    SUM(CASE WHEN created_time <= close_time - INTERVAL 1 HOUR THEN yes_price * count ELSE 0 END) / 
                        NULLIF(SUM(CASE WHEN created_time <= close_time - INTERVAL 1 HOUR THEN count ELSE 0 END), 0) as vwap_1h,
                    
                    SUM(CASE WHEN created_time <= close_time - INTERVAL 30 MINUTE THEN yes_price * count ELSE 0 END) / 
                        NULLIF(SUM(CASE WHEN created_time <= close_time - INTERVAL 30 MINUTE THEN count ELSE 0 END), 0) as vwap_30m,
                    
                    SUM(CASE WHEN created_time <= close_time - INTERVAL 10 MINUTE THEN yes_price * count ELSE 0 END) / 
                        NULLIF(SUM(CASE WHEN created_time <= close_time - INTERVAL 10 MINUTE THEN count ELSE 0 END), 0) as vwap_10m,
                    
                    SUM(count) as total_volume
                FROM trade_intervals
                GROUP BY ticker
            )
            SELECT 
                m.ticker,
                m.title,
                m.result,
                m.category,
                v.vwap_7d, v.vwap_3d, v.vwap_1d, v.vwap_6h,
                v.vwap_2h, v.vwap_1h, v.vwap_30m, v.vwap_10m,
                -- Legacy backwards compatibility for EV calculation
                v.vwap_2h as pre_speech_prob,
                v.total_volume
            FROM market_base m
            INNER JOIN vwaps v ON m.ticker = v.ticker
            """
        ).df()

        # Strict political filtering
        df["group_cat"] = df["category"].apply(get_group)
        df = df[df["group_cat"] == "Politics"].copy()

        # Parse speaker and event
        pat1 = r"What will (.+?) say during (.+?)\?"
        pat2 = r"Will the (White House Press Secretary) say .+ at (her next press briefing)\?"
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
            elif row["speaker"] == "White House Press Secretary":
                return "Group 2: Press Briefings"
            elif row["speaker"] == "Zohran Mamdani" or "NYC Mayor's Office" in row["event"]:
                return "Group 3: Mayoral (Zohran Mamdani)"
            else:
                return "Group 4: Niche / One-offs"

        df["subgroup"] = df.apply(assign_subgroup, axis=1)
        
        df["actual_yes"] = df["result"] == "yes"
        # We need pre_speech_prob (which maps to 2h VWAP) to exist for EV compatibility, 
        # dropping rows where it's NaN so the legacy metrics don't break.
        df = df.dropna(subset=["pre_speech_prob"]).copy()
        
        df["predicted_yes"] = df["pre_speech_prob"] > 50.0  # pre_speech_prob is in cents here (1-99)
        df["correct"] = df["predicted_yes"] == df["actual_yes"]

        # Expected Profit calculation
        def calc_trade_profits(row):
            price = row["pre_speech_prob"]  # cents
            fee = self._kalshi_fee(price)
            resolved_yes = row["actual_yes"]
            
            res = {}
            
            # Extremes (>70 Yes, <30 No)
            if price > 70.0:
                cost = price + fee
                payout = 100.0 if resolved_yes else 0.0
                res["ext_traded"] = 1
                res["ext_cost"] = cost
                res["ext_payout"] = payout
            elif price < 30.0:
                cost = (100.0 - price) + fee
                payout = 100.0 if not resolved_yes else 0.0
                res["ext_traded"] = 1
                res["ext_cost"] = cost
                res["ext_payout"] = payout
            else:
                res["ext_traded"] = 0
                res["ext_cost"] = 0.0
                res["ext_payout"] = 0.0
                
            # > 60 Yes
            if price > 60.0:
                res["gt60_traded"] = 1
                res["gt60_cost"] = price + fee
                res["gt60_payout"] = 100.0 if resolved_yes else 0.0
            else:
                res["gt60_traded"] = 0
                res["gt60_cost"] = 0.0
                res["gt60_payout"] = 0.0

            # > 70 Yes
            if price > 70.0:
                res["gt70_traded"] = 1
                res["gt70_cost"] = price + fee
                res["gt70_payout"] = 100.0 if resolved_yes else 0.0
            else:
                res["gt70_traded"] = 0
                res["gt70_cost"] = 0.0
                res["gt70_payout"] = 0.0

            # > 80 Yes
            if price > 80.0:
                res["gt80_traded"] = 1
                res["gt80_cost"] = price + fee
                res["gt80_payout"] = 100.0 if resolved_yes else 0.0
            else:
                res["gt80_traded"] = 0
                res["gt80_cost"] = 0.0
                res["gt80_payout"] = 0.0

            # > 90 Yes
            if price > 90.0:
                res["gt90_traded"] = 1
                res["gt90_cost"] = price + fee
                res["gt90_payout"] = 100.0 if resolved_yes else 0.0
            else:
                res["gt90_traded"] = 0
                res["gt90_cost"] = 0.0
                res["gt90_payout"] = 0.0

            return pd.Series(res)

        trade_stats = df.apply(calc_trade_profits, axis=1)
        for col in trade_stats.columns:
            df[col] = trade_stats[col]

        # Run subgroup aggregations
        subgroups = [
            "Group 1: Trump Markets", 
            "Group 2: Press Briefings", 
            "Group 3: Mayoral (Zohran Mamdani)", 
            "Group 4: Niche / One-offs"
        ]
        
        bins_10c = np.linspace(0, 100, 11) # 0, 10, ..., 100
        labels_10c = [f"{(b)}-{(b+10)}%" for b in bins_10c[:-1]]
        
        bins_5c = np.linspace(0, 100, 21) # 0, 5, ..., 100
        labels_5c = [f"{(b)}-{(b+5)}%" for b in bins_5c[:-1]]
        
        bins_1c = np.arange(0, 101)  # 0, 1, 2, ..., 100
        labels_1c = [f"{int(b)}%" for b in bins_1c[:-1]]
        
        profit_breakdown = []
        calibration_dfs_10c = {}
        calibration_dfs_5c = {}
        calibration_dfs_1c = {}
        mad_over_time = {}
        
        # Time intervals in chronological order for plotting
        time_intervals = ["1d", "6h", "2h", "1h", "30m", "10m"]
        
        def calculate_calibration(g_df, bins, labels, midpoints):
            cal_by_time = {}
            mad_by_time = []
            
            for interval in time_intervals:
                col = f"vwap_{interval}"
                # Bucket probabilities
                g_df_interval = g_df.copy()
                g_df_interval[f"prob_bucket_{interval}"] = pd.cut(
                    g_df_interval[col], bins=bins, labels=labels, include_lowest=True
                )
                
                # Aggregate
                cal = g_df_interval.groupby(f"prob_bucket_{interval}", observed=False).agg(
                    n=("ticker", "count"),
                    actual_yes_count=("actual_yes", "sum")
                ).reset_index()
                
                cal["actual_yes_pct"] = np.where(cal["n"] > 0, cal["actual_yes_count"] / cal["n"], np.nan)
                
                # Assign representative probability (midpoint of bucket) to calculate MAD
                cal["midpoint"] = midpoints
                
                # Calculate True Mean Absolute Deviation for buckets with enough data (n >= 5) to remove noise
                valid_cal = cal[cal["n"] >= 5].copy()
                if not valid_cal.empty:
                    valid_cal["abs_err"] = np.abs((valid_cal["actual_yes_pct"] * 100) - valid_cal["midpoint"])
                    mad_by_time.append(valid_cal["abs_err"].mean())
                else:
                    mad_by_time.append(np.nan)
                
                # Rename the probabilty bucket so X axis works consistently
                cal = cal.rename(columns={f"prob_bucket_{interval}": "prob_bucket"})
                cal_by_time[interval] = cal
                
            return cal_by_time, mad_by_time
        
        for g in subgroups:
            g_df = df[df["subgroup"] == g]
            acc = g_df["correct"].mean()
            tot = len(g_df)
            
            # Calibration per time interval (10c)
            midpoints_10c = np.linspace(5, 95, 10)
            cal_by_time_10c, mad_by_time = calculate_calibration(g_df, bins_10c, labels_10c, midpoints_10c)
            calibration_dfs_10c[g] = cal_by_time_10c
            mad_over_time[g] = mad_by_time
            
            # Calibration per time interval (5c)
            midpoints_5c = np.linspace(2.5, 97.5, 20)
            cal_by_time_5c, _ = calculate_calibration(g_df, bins_5c, labels_5c, midpoints_5c)
            calibration_dfs_5c[g] = cal_by_time_5c
            
            # Calibration per time interval (1c)
            midpoints_1c = np.arange(0.5, 100, 1)  # 0.5, 1.5, ..., 99.5
            cal_by_time_1c, _ = calculate_calibration(g_df, bins_1c, labels_1c, midpoints_1c)
            calibration_dfs_1c[g] = cal_by_time_1c
            
            # EV Profitability (using 2-hour mark for canonical profitability mapping)
            def get_counts_profit(prefix):
                g_traded_df = g_df[g_df[f"{prefix}_traded"] == 1]
                total_cost = g_traded_df[f"{prefix}_cost"].sum()
                total_payout = g_traded_df[f"{prefix}_payout"].sum()
                prof = (total_payout - total_cost) / total_cost if total_cost > 0 else 0.0
                return len(g_traded_df), prof
            
            trades_ext, prof_ext = get_counts_profit("ext")
            trades_60, prof_60 = get_counts_profit("gt60")
            trades_70, prof_70 = get_counts_profit("gt70")
            trades_80, prof_80 = get_counts_profit("gt80")
            trades_90, prof_90 = get_counts_profit("gt90")
            
            profit_breakdown.append({
                "subgroup": g,
                "total_mention_contracts": tot,
                "accuracy": acc,
                "ext_trades": trades_ext,
                "ext_profit": prof_ext,
                "gt60_trades": trades_60,
                "gt60_profit": prof_60,
                "gt70_trades": trades_70,
                "gt70_profit": prof_70,
                "gt80_trades": trades_80,
                "gt80_profit": prof_80,
                "gt90_trades": trades_90,
                "gt90_profit": prof_90,
            })

        profit_df = pd.DataFrame(profit_breakdown)
        print("\n--- Buy Extremes & Favorites Strategy Expected Profits ---")
        print(profit_df.to_string(index=False))

        # Build markdown text
        markdown_text = f"""# Speech Mention Subgroup Accuracy

## Buy Extremes & Favorites Strategy Expected Profits
{profit_df.to_markdown(index=False)}
"""

        # Generate figures and charts
        figures_10c, mad_fig = self._create_figures(calibration_dfs_10c, mad_over_time, profit_df, "10c")
        figures_5c, _ = self._create_figures(calibration_dfs_5c, mad_over_time, profit_df, "5c")
        figures_1c, _ = self._create_figures(calibration_dfs_1c, mad_over_time, profit_df, "1c")
        vwap_trace_figures = self._create_vwap_trace_figures(df)
        
        # Combine figures to return for explicit handling in save
        # We need to restructure the output somewhat so the saving mechanism can 
        # bundle the multiple pages into a single PDF per group.
        # So we'll pass them in a special structured format.
        
        structured_figures = {}
        # Keep MAD summary standalone
        structured_figures["__mad_summary"] = mad_fig
        
        groups = [
            "Group 1: Trump Markets", 
            "Group 2: Press Briefings", 
            "Group 3: Mayoral (Zohran Mamdani)", 
            "Group 4: Niche / One-offs"
        ]
        
        for g in groups:
            group_key = g.split(":")[0].replace(" ", "_").lower()
            if group_key in figures_10c:
                structured_figures[group_key] = [
                    figures_10c.get(group_key),
                    figures_5c.get(group_key),
                    figures_1c.get(group_key),
                    vwap_trace_figures.get(group_key)
                ]

        chart = self._create_chart(mad_over_time)

        # HACK: The Analysis framework expects `figures` to be dict[str, Figure], but here 
        # to simplify custom saving in `save`, we bend the type of `figures` 
        # to hold lists of Figures for the multi-page behavior.
        return AnalysisOutput(figures=structured_figures, data=profit_df, chart=chart, markdown=markdown_text)

    def save(self, output_dir: Path | str, formats: list[str] | None = None, dpi: int = 300) -> dict[str, Path]:
        """Override save to ensure output is written to a dedicated subdirectory and supports multipage-pdf."""
        from matplotlib.backends.backend_pdf import PdfPages
        
        custom_output_dir = Path(output_dir) / "speech_mention_subgroups"
        custom_output_dir.mkdir(parents=True, exist_ok=True)
        
        output = self.run()
        saved: dict[str, Path] = {}
        
        if formats is None:
            formats = ["png", "pdf", "csv", "md"]
            
        fig_formats = [f for f in formats if f in ("png", "pdf", "svg", "gif")]
        
        if output.figures is not None:
            for name, figs in output.figures.items():
                if name == "__mad_summary":
                    fig = figs
                    if fig is not None:
                        for fmt in fig_formats:
                            path = custom_output_dir / f"{name}.{fmt}"
                            fig.savefig(path, dpi=dpi, bbox_inches="tight")
                            saved[f"{name}.{fmt}"] = path
                        plt.close(fig)
                    continue

                if not isinstance(figs, list):
                    continue
                    
                # We have a list of figures for a particular group
                figs = [f for f in figs if f is not None]
                if not figs:
                    continue
                    
                # Save multi-page PDF
                if "pdf" in fig_formats:
                    pdf_path = custom_output_dir / f"{name}_speech_mention_subgroup_accuracy.pdf"
                    with PdfPages(pdf_path) as pdf:
                        for fig in figs:
                            pdf.savefig(fig, bbox_inches="tight")
                    saved[f"{name}_pdf"] = pdf_path
                
                # Save individual PNGs for safety
                if "png" in fig_formats:
                    suffixes = ["10c_calibration", "5c_calibration", "1c_calibration", "vwap_trace"]
                    for idx, fig in enumerate(figs):
                        if idx < len(suffixes):
                            png_path = custom_output_dir / f"{name}_speech_mention_subgroup_accuracy_{suffixes[idx]}.png"
                            fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
                            saved[f"{name}_{idx}_png"] = png_path
                            
                # Cleanup memory
                for fig in figs:
                    plt.close(fig)

        # Save CSV
        if output.data is not None and "csv" in formats:
            path = custom_output_dir / f"speech_mention_subgroup_accuracy.csv"
            output.data.to_csv(path, index=False)
            saved["csv"] = path

        # Save JSON chart config
        if output.chart is not None and "json" in formats:
            path = custom_output_dir / f"speech_mention_subgroup_accuracy.json"
            path.write_text(output.chart.to_json())
            saved["json"] = path
            
        # Save Markdown text
        if output.markdown is not None and "md" in formats:
            path = custom_output_dir / f"speech_mention_subgroup_accuracy.md"
            path.write_text(output.markdown)
            saved["md"] = path

        return saved

    def _create_figures(self, calibration_dfs: dict, mad_over_time: dict, profit_df: pd.DataFrame, bucket_type: str) -> tuple[dict[str, plt.Figure], plt.Figure]:
        figures = {}
        groups = [
            "Group 1: Trump Markets", 
            "Group 2: Press Briefings", 
            "Group 3: Mayoral (Zohran Mamdani)", 
            "Group 4: Niche / One-offs"
        ]
        time_intervals = ["1d", "6h", "2h", "1h", "30m", "10m"]
        
        # Define a perceptually uniform colormap corresponding to time closing
        # Start faint/cool (7d), end strong/warm (10m)
        cmap = plt.get_cmap("viridis_r")
        colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(time_intervals))]
        
        if bucket_type == "10c":
            x_bins = np.arange(10)
            ideal_cal = np.linspace(5, 95, 10)
        elif bucket_type == "5c":
            x_bins = np.arange(20)
            ideal_cal = np.linspace(2.5, 97.5, 20)
        else:  # 1c
            x_bins = np.arange(100)
            ideal_cal = np.arange(0.5, 100, 1)
        
        for g in groups:
            fig, ax = plt.subplots(figsize=(14, 6) if bucket_type == "1c" else (10, 6))
            # Handle cases where subgroup may have no data (e.g. empty)
            subgroup_data = profit_df[profit_df["subgroup"] == g]
            if subgroup_data.empty:
                plt.close(fig)
                continue
                
            p_info = subgroup_data.iloc[0]
            tot_markets = p_info["total_mention_contracts"]
            acc = float(p_info["accuracy"] * 100)
            
            # Plot perfect calibration
            ax.plot(x_bins, ideal_cal, marker="", color="black", linestyle="--", linewidth=1.5, label="Perfect Calibration")
            
            cal_by_time = calibration_dfs.get(g, {})
            if not cal_by_time:
                plt.close(fig)
                continue
                
            prob_labels = cal_by_time["1d"]["prob_bucket"].values
            
            for j, interval in enumerate(time_intervals):
                cal = cal_by_time[interval]
                y = cal["actual_yes_pct"].fillna(np.nan) * 100
                marker = "" if bucket_type == "1c" else "o"
                lw = 1.2 if bucket_type == "1c" else 2
                ax.plot(x_bins, y, marker=marker, color=colors[j], linewidth=lw, label=f"{interval} before Close")
                
                # To prevent charting chaos, we only annotate 'n' counts for the 2h bucket (legacy view)
                # Skip annotations entirely for 1c since there are too many points
                if interval == "2h" and bucket_type != "1c":
                    for k, val in enumerate(y):
                        if not np.isnan(val):
                            n_count = cal.iloc[k]["n"]
                            if n_count > 0:
                                ax.text(x_bins[k], val + 3, f"n={n_count}", ha="center", va="bottom", fontsize=8)

            if bucket_type == "1c":
                # Show every 10th tick for readability
                tick_positions = np.arange(0, 100, 10)
                tick_labels = [prob_labels[i] for i in tick_positions]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=45, ha="right")
            else:
                ax.set_xticks(x_bins)
                ax.set_xticklabels(prob_labels, rotation=45, ha="right")
            ax.set_ylim(-5, 110)
            
            ax.set_xlabel(f"Market Implied Probability ({bucket_type} buckets)")
            ax.set_ylabel("Actual % Resolved 'Yes'")
            ax.set_title(f"{g}\nTotal Markets: {tot_markets} | Overall Accuracy: {acc:.1f}%")
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
            ax.grid(True, linestyle="--", alpha=0.3)
            
            plt.tight_layout()
            
            # Slugify for filename suffix
            group_key = g.split(":")[0].replace(" ", "_").lower()
            figures[group_key] = fig
            
        # Draw Summary MAD Graph
        fig_mad, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Calibration Bias Reduction Near Close (Mean Absolute Deviation)")
        
        subgroup_colors = {
            "Group 1: Trump Markets": "#d62728", 
            "Group 2: Press Briefings": "#1f77b4", 
            "Group 3: Mayoral (Zohran Mamdani)": "#2ca02c", 
            "Group 4: Niche / One-offs": "#ff7f0e"
        }
        x_time = np.arange(len(time_intervals))
        
        for g in groups:
            if g in mad_over_time:
                ax.plot(x_time, mad_over_time[g], marker="o", linewidth=2, color=subgroup_colors.get(g, "black"), label=g)
            
        ax.set_xticks(x_time)
        ax.set_xticklabels(time_intervals)
        ax.set_ylabel("Mean Absolute Deviation (%)")
        ax.set_xlabel("Time Before Market Close")
        ax.invert_xaxis()  # So time gets closer to 0 moving right
        
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        
        return figures, fig_mad

    def _create_vwap_trace_figures(self, df: pd.DataFrame) -> dict[str, plt.Figure]:
        figures = {}
        groups = [
            "Group 1: Trump Markets", 
            "Group 2: Press Briefings", 
            "Group 3: Mayoral (Zohran Mamdani)", 
            "Group 4: Niche / One-offs"
        ]
        time_intervals = ["7d", "3d", "1d", "6h", "2h", "1h", "30m", "10m"]
        vwap_cols = [f"vwap_{i}" for i in time_intervals]
        x_time = np.arange(len(time_intervals))
        
        for g in groups:
            fig, ax = plt.subplots(figsize=(10, 6))
            subgroup_data = df[df["subgroup"] == g]
            if subgroup_data.empty:
                plt.close(fig)
                continue
                
            ax.set_title(f"VWAP Traces for Individual Contracts: {g}")
            
            # Plot Yes outcomes
            yes_data = subgroup_data[subgroup_data["actual_yes"] == True]
            if not yes_data.empty:
                ax.plot(x_time, yes_data[vwap_cols].T, color="#2ca02c", alpha=0.05, linewidth=1)
                # Plot a dummy line for the legend
                ax.plot([], [], color="#2ca02c", label="Resolved Yes")
                
            # Plot No outcomes
            no_data = subgroup_data[subgroup_data["actual_yes"] == False]
            if not no_data.empty:
                ax.plot(x_time, no_data[vwap_cols].T, color="#d62728", alpha=0.05, linewidth=1)
                # Plot a dummy line for the legend
                ax.plot([], [], color="#d62728", label="Resolved No")
                
            ax.set_xticks(x_time)
            ax.set_xticklabels(time_intervals)
            ax.set_ylabel("VWAP (Cents)")
            ax.set_xlabel("Time Before Market Close")
            ax.set_ylim(-5, 105)
            
            # Create a more visible legend
            leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
            for line in leg.get_lines():
                line.set_linewidth(3)
                line.set_alpha(1.0)
                
            ax.grid(True, linestyle="--", alpha=0.3)

            plt.tight_layout()
            
            group_key = g.split(":")[0].replace(" ", "_").lower()
            figures[group_key] = fig
            
        return figures

    def _create_chart(self, mad_over_time: dict) -> ChartConfig:
        time_intervals = ["1d", "6h", "2h", "1h", "30m", "10m"]
        chart_data = []
        for i, interval in enumerate(time_intervals):
            row = {"interval": interval}
            for g, mads in mad_over_time.items():
                if not np.isnan(mads[i]):
                     row[g] = float(round(mads[i], 2))
            chart_data.append(row)
            
        groups = list(mad_over_time.keys())
        colors = {
            "Group 1: Trump Markets": "#d62728",
            "Group 2: Press Briefings": "#1f77b4",
            "Group 3: Mayoral (Zohran Mamdani)": "#2ca02c",
            "Group 4: Niche / One-offs": "#ff7f0e"
        }
            
        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="interval",
            yKeys=groups,
            title="Calibration Bias Reduction Near Close (MAD)",
            yUnit=UnitType.PERCENT,
            xLabel="Time Before Market Close",
            yLabel="Mean Absolute Deviation (%)",
            colors=colors
        )
