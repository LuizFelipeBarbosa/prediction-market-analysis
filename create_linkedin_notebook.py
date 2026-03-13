#!/usr/bin/env python3
"""Create a condensed ~15-page LinkedIn version of analysis.ipynb."""

import json
import re
import copy


def load_notebook(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_notebook(nb: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)


def get_source(cell: dict) -> str:
    return "".join(cell["source"])


def set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(True)
    # Ensure last line ends with newline if non-empty
    if cell["source"] and not cell["source"][-1].endswith("\n"):
        cell["source"][-1] += "\n"


def strip_leading_hr(cell: dict) -> dict:
    """Remove a leading '---\\n' (horizontal rule) from a markdown cell."""
    src = get_source(cell)
    if src.startswith("---\n"):
        set_source(cell, src[4:])
    return cell


def trim_cell_0(cell: dict) -> dict:
    """Keep title, code availability, abstract, and first intro paragraph.
    Remove second intro paragraph and sections 1.1–1.4."""
    src = get_source(cell)
    # Cut from second intro paragraph onward
    marker = "Prediction markets have also shifted"
    idx = src.find(marker)
    if idx != -1:
        src = src[:idx].rstrip() + "\n"
    set_source(cell, src)
    return cell


def trim_methodology(cell: dict) -> dict:
    """Remove ### Methodology section and everything after it in the cell."""
    src = get_source(cell)
    marker = "### Methodology"
    idx = src.find(marker)
    if idx != -1:
        src = src[:idx].rstrip() + "\n"
    set_source(cell, src)
    return cell


def trim_cell_7(cell: dict) -> dict:
    """Remove ### Methodology from Section 2."""
    return trim_methodology(cell)


def trim_cell_22(cell: dict) -> dict:
    """Remove ### Methodology from Section 6."""
    return trim_methodology(cell)


def trim_cell_27(cell: dict) -> dict:
    """Remove ### Data Construction through ### Speaker and Event Subgroups table."""
    src = get_source(cell)
    marker = "### Data Construction"
    idx = src.find(marker)
    if idx != -1:
        src = src[:idx].rstrip() + "\n"
    set_source(cell, src)
    return cell


def trim_cell_28(cell: dict) -> dict:
    """Trim formal equations from Section 7a Directional Accuracy.
    Keep intro and plain-English explanation, remove math blocks."""
    src = get_source(cell)
    # Remove the formal math blocks ($$...$$) but keep surrounding text
    # Remove "Formally," paragraph through the accuracy equation
    # Strategy: remove from "Formally," to the line before "In short,"
    formally_idx = src.find("Formally,")
    in_short_idx = src.find("In short,")
    if formally_idx != -1 and in_short_idx != -1:
        src = src[:formally_idx].rstrip() + "\n\n" + src[in_short_idx:]
    set_source(cell, src)
    return cell


def trim_cell_32(cell: dict) -> dict:
    """Remove ### Methodology and #### 1. Bucket construction from Section 7a Trump."""
    src = get_source(cell)
    marker = "### Methodology"
    idx = src.find(marker)
    if idx != -1:
        src = src[:idx].rstrip() + "\n"
    set_source(cell, src)
    return cell


def trim_cell_4(cell: dict) -> dict:
    """Remove ### Methodology from Section 1."""
    return trim_methodology(cell)


def trim_cell_45(cell: dict) -> dict:
    """Trim verbose setup text from Section 8, keep fee formula and strategies."""
    src = get_source(cell)
    # Remove the last two paragraphs ("This separates two questions..." through end)
    marker = "This separates two questions"
    idx = src.find(marker)
    if idx != -1:
        src = src[:idx].rstrip() + "\n"
    set_source(cell, src)
    return cell


def trim_cell_6(cell: dict) -> dict:
    """Trim Section 1 interpretation: keep summary, remove equation and bullet explanations."""
    set_source(cell, """### Interpretation

The results are plotted as **actual win rate versus contract price**, with a 45-degree reference line representing perfect calibration.

### Summary

The red-shaded region at the left confirms classic longshot bias: events priced below ~20% win less often than their price implies. The green-shaded region at the right shows the mirror effect — favorites are underpriced, winning more often than the market suggests.

This is not a small effect. At 10¢, the actual win rate is ~9.2% vs. the 10% implied. But at 5¢, the actual win rate is ~4.2% vs. 5% implied, and the gap widens further at the extremes. On the favorite side, events priced at 90¢ win ~90.8% of the time close to calibrated, but consistently above the line.
""")
    return cell


def trim_cell_9(cell: dict) -> dict:
    """Trim Section 2 interpretation: condense to key insight."""
    set_source(cell, """### Interpretation

If low-probability YES contracts are overpriced, then makers who buy **YES** at low prices should underperform, while makers who buy **NO** against those same contracts should outperform.

### Summary

The figure shows that YES makers who provide liquidity at low prices are consistently below break-even — they are buying overpriced longshots. NO makers across most of the price range hover near or above break-even. The structural bias favors selling longshots (providing NO liquidity at low prices) and buying favorites.
""")
    return cell


def trim_cell_22_v2(cell: dict) -> dict:
    """Trim Section 6 intro: remove NBA benchmark detail and methodology."""
    set_source(cell, """## Section 6: Calibration in Political Speech Mention Contracts

**Political speech mention contracts** are contracts like *"Will Trump mention immigration in his State of the Union address?"* or *"Will the White House press briefing mention Ukraine?"*

This section constructs a calibration curve to evaluate the accuracy of Kalshi's implied probabilities. Contracts are grouped into 10-percentage-point probability buckets based on their implied price. Under perfect calibration, every point would lie on the 45-degree diagonal. The central question is whether a contract priced at 70% two hours before a speech resolves YES approximately 70% of the time.

The shaded regions highlight the tails where exploitable divergence is most likely: the low-probability area (where longshot overpricing would appear) and the high-probability area (where favorite underpricing — the principal finding of this paper — is concentrated). Sample sizes (`n=`) are annotated at each point to indicate statistical power.
""")
    return cell


def trim_cell_24(cell: dict) -> dict:
    """Trim Section 6 interpretation: condense to key finding."""
    set_source(cell, """### Interpretation

The calibration curve for political speech mention contracts diverges dramatically from the diagonal in the high-probability zone. Speech mention contracts priced above 60¢ systematically resolve YES far more often than their prices imply. The gap widens from +12% at the 60–70% bucket to a peak of +17% at 70–80%, before narrowing at the extreme tail (90–100%).

This pattern is consistent with classic longshot bias: longshot contracts are overpriced while favorites are significantly underpriced. The large sample sizes (hundreds of contracts per bucket) lend statistical significance to these deviations. This is the paper's headline result: a structural, persistent, and economically large mispricing in a specific Kalshi market category.
""")
    return cell


def trim_cell_28_v2(cell: dict) -> dict:
    """Trim Section 7a directional accuracy: keep only plain-English explanation."""
    set_source(cell, """## Section 7a: Directional Accuracy

The first summary statistic is **overall directional accuracy**, based on the market's 2-hour pre-event probability. If the market price 2 hours before close is above 50¢, it's predicting the event will be mentioned; if below 50¢, predicting it won't. Accuracy is the percentage of times this naive threshold prediction was right.

This statistic answers a simple question: **how often was the market directionally correct two hours before resolution?** It is useful as a descriptive benchmark, but it should not be confused with calibration. A subgroup can have high directional accuracy while still being systematically overconfident or underconfident in probability terms.
""")
    return cell


def trim_cell_36(cell: dict) -> dict:
    """Trim Section 7b Press Briefings intro: condense."""
    set_source(cell, """## Section 7b: Press Briefing Contracts

White House press briefing mention contracts form the second subgroup. These contracts ask whether a specific topic (e.g., "border security," "Ukraine") will be mentioned during the daily White House press briefing. This subgroup has fewer observations and a lower win rate than Trump markets, which may translate to a smaller but still positive edge. If the mispricing structure is similar in shape but smaller in magnitude, it suggests the bias is systematic across political speech contracts rather than idiosyncratic to Trump-related events.
""")
    return cell


def trim_cell_39(cell: dict) -> dict:
    """Trim Section 7c Mamdani intro: condense."""
    set_source(cell, """## Section 7c: Mamdani Markets (NYC Mayoral Race)

The Zohran Mamdani NYC mayoral race contracts are hyper-specific local political contracts that retail traders likely price poorly due to low information availability. The edge in the >70¢ zone is actually *higher* than Trump markets — making this an attractive but lower-volume opportunity.

The analytical interest here lies in *information asymmetry*. Local political events attract fewer sophisticated traders, thinner order books, and less media pre-analysis — all conditions that theory predicts should produce larger and more persistent mispricings.
""")
    return cell


def trim_cell_44(cell: dict) -> dict:
    """Trim Group 4 Niche interpretation: condense significantly."""
    set_source(cell, """The Group 4 "Niche / One-offs" calibration chart shows that high-probability YES outcomes are consistently **underpriced**, especially as the event gets closer to resolution. When the price falls between 60 and 80 cents, actual resolution rates substantially exceed market-implied probabilities, with calibration gaps of about 10 to 20 percentage points. The pattern strengthens as resolution approaches, indicating that even late in trading, these contracts do not fully reflect the true probability of resolution.

This subgroup exhibits characteristics of an "attention-friction market": one-off contracts are harder to price consistently due to their lack of repeatable structure and thinner participation. From a trading perspective, these contracts present more dispersed opportunities than the headline political subgroups, but the repeated underpricing of likely YES outcomes late in the trading window suggests a meaningful behavioral edge.
""")
    return cell


def trim_cell_56(cell: dict) -> dict:
    """Trim Conclusions: remove Next Steps section."""
    src = get_source(cell)
    marker = "### Next steps"
    idx = src.find(marker)
    if idx != -1:
        src = src[:idx].rstrip() + "\n"
    set_source(cell, src)
    return cell


def main():
    nb = load_notebook("analysis.ipynb")
    cells = nb["cells"]

    # Cells to keep (in order), with optional trimming functions
    keep_spec = [
        # (cell_index, trim_function_or_None, strip_hr)
        (0, trim_cell_0, False),        # Title, abstract, intro (trimmed)
        (4, trim_cell_4, True),         # Section 1 intro (strip HR, no methodology)
        (5, None, False),               # Section 1 chart (code)
        (6, trim_cell_6, False),        # Section 1 interpretation (condensed)
        (7, trim_cell_7, True),         # Section 2 intro (strip HR, no methodology)
        (8, None, False),               # Section 2 chart (code)
        (9, trim_cell_9, False),        # Section 2 interpretation (condensed)
        (22, trim_cell_22_v2, True),    # Section 6 intro (condensed)
        (23, None, False),              # Section 6 chart (code)
        (24, trim_cell_24, False),      # Section 6 interpretation (condensed)
        (26, None, False),              # Section 6 comparison tables
        (27, trim_cell_27, True),       # Section 7 intro (strip HR, no data construction)
        (28, trim_cell_28_v2, True),    # Section 7a directional accuracy (condensed)
        (29, None, False),              # Subgroup accuracy code
        (30, None, False),              # Subgroup accuracy bar chart
        (31, None, False),              # Section 7a commentary
        (32, trim_cell_32, True),       # Section 7a Trump intro (strip HR, no methodology)
        (33, None, False),              # Plot function definition (code)
        (34, None, False),              # Trump chart (code)
        (35, None, False),              # Trump interpretation
        (36, trim_cell_36, True),       # Section 7b Press Briefings (condensed)
        (37, None, False),              # Press Briefings chart
        (38, None, False),              # Press Briefings interpretation
        (39, trim_cell_39, True),       # Section 7c Mamdani (condensed)
        (40, None, False),              # Mamdani chart
        (41, None, False),              # Mamdani interpretation
        (42, None, True),               # Section 7d Niche (strip HR)
        (43, None, False),              # Niche chart
        (44, trim_cell_44, False),      # Niche interpretation (condensed)
        (45, trim_cell_45, True),       # Section 8 intro (strip HR, trimmed)
        (46, None, False),              # Strategy backtest code
        (47, None, False),              # Fee model markdown
        (48, None, False),              # Strategy chart (code)
        (49, None, False),              # Strategy interpretation
        (51, None, True),               # Section 9 intro (strip HR)
        (54, None, False),              # Section 9 heatmap (code)
        (55, None, False),              # Section 9 heatmap interpretation
        (56, trim_cell_56, True),       # Conclusions (strip HR, no next steps)
        (57, None, False),              # Research & Citations
    ]

    new_cells = []
    for cell_idx, trim_fn, do_strip_hr in keep_spec:
        cell = copy.deepcopy(cells[cell_idx])
        if trim_fn:
            cell = trim_fn(cell)
        if do_strip_hr and cell["cell_type"] == "markdown":
            cell = strip_leading_hr(cell)
        new_cells.append(cell)

    # Build new notebook
    new_nb = copy.deepcopy(nb)
    new_nb["cells"] = new_cells

    save_notebook(new_nb, "analysis_linkedin.ipynb")
    print(f"Created analysis_linkedin.ipynb with {len(new_cells)} cells")

    # Print a summary of what was included
    for i, (cell_idx, _, _) in enumerate(keep_spec):
        src = get_source(new_cells[i])
        preview = src[:80].replace("\n", "\\n")
        print(f"  New cell {i:2d} (orig {cell_idx:2d}): {preview}")


if __name__ == "__main__":
    main()
