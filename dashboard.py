import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Kalshi Analysis Dashboard", layout="wide")

st.title("Kalshi Prediction Market Analysis")

output_dir = Path("output")
if not output_dir.exists():
    st.error("Output directory not found. Please run the analyses first using `uv run main.py analyze all`.")
    st.stop()

# Find all markdown files as the base analyses
md_files = list(output_dir.glob("*.md"))
if not md_files:
    st.info("No analysis outputs found in the `output/` directory.")
    st.stop()

# Get base names
analyses = sorted([f.stem for f in md_files])

st.sidebar.header("Navigation")
selected_analysis = st.sidebar.selectbox("Select Analysis", analyses, format_func=lambda x: x.replace("_", " ").title())

if selected_analysis:
    st.header(selected_analysis.replace("_", " ").title())

    # We display markdown first on top if it's broad, or we use tabs
    import pandas as pd
    import altair as alt

    tab1, tab2, tab3 = st.tabs(["Interactive Charts & Report", "Raw Data Table", "Chart JSON/Config"])

    csv_path = output_dir / f"{selected_analysis}.csv"
    df = None
    if csv_path.exists():
        df = pd.read_csv(csv_path)

    with tab1:
        st.subheader("Interactive Visualizations")

        # Render specific interactive charts based on the analysis
        if df is not None:
            if selected_analysis == "speech_mention_subgroup_accuracy":
                # 1. EV Profitability Bar Chart
                # User config toggles
                available_groups = df["subgroup"].unique()
                selected_groups = st.multiselect("Filter Subgroups", available_groups, default=available_groups)

                if selected_groups:
                    filtered_df = df[df["subgroup"].isin(selected_groups)]

                    # Melt dataframe for grouped bar chart (Accuracy vs Profit)
                    melted = filtered_df.melt(
                        id_vars=["subgroup"],
                        value_vars=["accuracy", "ext_profit"],
                        var_name="Metric",
                        value_name="Percentage",
                    )
                    # Convert to strict % for display
                    melted["Percentage"] = melted["Percentage"] * 100

                    chart = (
                        alt.Chart(melted)
                        .mark_bar()
                        .encode(
                            x=alt.X("Metric:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
                            y=alt.Y("Percentage:Q", title="Percentage (%)"),
                            color=alt.Color("Metric:N", scale=alt.Scale(range=["#1f77b4", "#2ca02c"])),
                            column=alt.Column("subgroup:N", title="Speech Mention Subgroup"),
                            tooltip=["subgroup", "Metric", alt.Tooltip("Percentage:Q", format=".2f")],
                        )
                        .properties(width=200, title="Expected Extremes Profit vs Overall Accuracy")
                        .configure_view(stroke="transparent")
                    )

                    st.altair_chart(chart, use_container_width=False)

                st.divider()

                # 2. Calibration Bias Reduction Over Time
                import json

                json_path = output_dir / f"{selected_analysis}.json"
                if json_path.exists():
                    chart_config = json.loads(json_path.read_text())
                    jf = pd.DataFrame(chart_config["data"])

                    time_intervals = ["7d", "3d", "1d", "6h", "2h", "1h", "30m", "10m"]
                    if selected_groups:
                        # Melt to get Group, Interval, MAD format
                        melted_mad = jf.melt(
                            id_vars=["interval"],
                            value_vars=[g for g in chart_config["yKeys"] if g in selected_groups],
                            var_name="Subgroup",
                            value_name="MAD (%)",
                        )

                        # Preserve interval ordering by casting to categorical
                        melted_mad["interval"] = pd.Categorical(
                            melted_mad["interval"], categories=time_intervals, ordered=True
                        )

                        line_chart = (
                            alt.Chart(melted_mad)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X("interval:O", title="Time Before Market Close", sort=time_intervals),
                                y=alt.Y("MAD (%):Q", title="Mean Absolute Deviation (%)"),
                                color=alt.Color(
                                    "Subgroup:N",
                                    scale=alt.Scale(
                                        domain=[
                                            "Group 1: Trump Markets",
                                            "Group 2: Press & Mayoral",
                                            "Group 3: Niche / One-offs",
                                        ],
                                        range=["#d62728", "#1f77b4", "#ff7f0e"],
                                    ),
                                ),
                                tooltip=["interval", "Subgroup", alt.Tooltip("MAD (%):Q", format=".2f")],
                            )
                            .properties(height=450, title="Calibration Bias Reduction Near Close")
                        )

                        st.altair_chart(line_chart, use_container_width=True)

                st.divider()
                st.subheader("Subgroup Calibration Curves")
                img_path = output_dir / f"{selected_analysis}.png"
                if img_path.exists():
                    st.image(str(img_path))

            elif selected_analysis == "kalshi_political_mention_calibration_deviation_over_time":
                # Line chart of date vs mean_absolute_deviation
                if "date" in df.columns and "mean_absolute_deviation" in df.columns:
                    st.line_chart(
                        df, x="date", y="mean_absolute_deviation", y_label="Mean Absolute Deviation (%)", x_label="Date"
                    )

            elif selected_analysis in ["political_speech_mention_accuracy", "nba_game_accuracy"]:
                # Line chart for Calibration (Actual vs Implied)
                # The CSV might contain bucketed actual vs implied probabilities
                if "implied_prob_bucket" in df.columns and "actual_yes_pct" in df.columns:
                    # Let's map buckets to midpoints for a line chart
                    df_chart = df.copy()
                    # Strip 'bucket' strings like '0-10%' to numeric midpoints if needed, or just use as categorical x

                    chart = (
                        alt.Chart(df_chart)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("implied_prob_bucket:N", title="Market Implied Probability", sort=None),
                            y=alt.Y(
                                "actual_yes_pct:Q", title="Actual % Resolved 'Yes'", scale=alt.Scale(domain=[0, 1])
                            ),
                            tooltip=["implied_prob_bucket", alt.Tooltip("actual_yes_pct:Q", format=".1%"), "n"],
                        )
                        .properties(height=400)
                    )

                    st.altair_chart(chart, use_container_width=True)
                else:
                    # Fallback to PNG if the CSV doesn't have the expected calibration columns
                    img_path = output_dir / f"{selected_analysis}.png"
                    if img_path.exists():
                        st.image(str(img_path))
            else:
                # Generic fallback
                img_path = output_dir / f"{selected_analysis}.png"
                if img_path.exists():
                    st.image(str(img_path))
        else:
            # No CSV, fallback to generic PNG
            img_path = output_dir / f"{selected_analysis}.png"
            if img_path.exists():
                st.image(str(img_path))

        st.divider()
        st.subheader("Summary Report")
        md_path = output_dir / f"{selected_analysis}.md"
        if md_path.exists():
            st.markdown(md_path.read_text())

    with tab2:
        if df is not None:
            # Provide an interactive, styled dataframe
            st.subheader("Data Table")

            # Auto-format percentage columns to be more readable
            column_configs = {}
            for col in df.columns:
                if (
                    "accuracy" in col.lower()
                    or "profit" in col.lower()
                    or "pct" in col.lower()
                    or "rate" in col.lower()
                ):
                    # Assuming these are typically represented as 0.0 - 1.0 decimals based on prior scripts
                    column_configs[col] = st.column_config.NumberColumn(
                        col.replace("_", " ").title(), format="%.2f", help=f"{col} represented as a decimal"
                    )
                else:
                    column_configs[col] = st.column_config.Column(col.replace("_", " ").title())

            st.dataframe(df, use_container_width=True, hide_index=True, column_config=column_configs)

            st.download_button(
                "Download CSV", csv_path.read_bytes(), file_name=f"{selected_analysis}.csv", mime="text/csv"
            )
        else:
            st.info("No CSV data available for this analysis.")

    with tab3:
        json_path = output_dir / f"{selected_analysis}.json"
        if json_path.exists():
            st.json(json_path.read_text())
            st.download_button(
                "Download JSON", json_path.read_bytes(), file_name=f"{selected_analysis}.json", mime="application/json"
            )
        else:
            st.info("No JSON chart config available for this analysis.")
