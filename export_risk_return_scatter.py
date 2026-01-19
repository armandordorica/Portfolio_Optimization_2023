import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio

from returns import get_returns_table, get_stddev_table


def plot_risk_return_scatter_by_horizon(
    returns_wide,
    stddev_wide,
    horizons,
    title=None,
    width=900,
    height=600,
):
    horizons = list(horizons)

    ret_long = returns_wide[horizons].reset_index()
    ret_long = ret_long.rename(columns={ret_long.columns[0]: "ticker"}).melt(
        id_vars="ticker", var_name="horizon", value_name="return"
    )

    vol_long = stddev_wide[horizons].reset_index()
    vol_long = vol_long.rename(columns={vol_long.columns[0]: "ticker"}).melt(
        id_vars="ticker", var_name="horizon", value_name="stddev"
    )

    plot_df = ret_long.merge(vol_long, on=["ticker", "horizon"]).dropna()

    fig = px.scatter(
        plot_df,
        x="stddev",
        y="return",
        color="horizon",
        text="ticker",
        title=title or "Risk/Return Scatter by Horizon",
    )
    fig.update_traces(textposition="top center", marker=dict(size=10, opacity=0.85))
    fig.update_layout(
        width=width,
        height=height,
        xaxis_title="Std Dev of Daily Returns",
        yaxis_title="Return",
        legend_title="Horizon",
    )
    return fig


def _parse_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export risk/return scatter plot to docs/ as HTML.")
    parser.add_argument("--parquet", default="ETFs_data.parquet.gzip", help="Input parquet file")
    parser.add_argument("--out", default="docs/risk_return_scatter.html", help="Output HTML path")
    parser.add_argument("--horizons", default="6M,1Y,3Y,5Y", help="Comma-separated horizons")
    parser.add_argument(
        "--patterns",
        default="VFV,XUU,VOO,BRK,QQ,XEF",
        help="Comma-separated substrings to select tickers (matched against Stock_name)",
    )
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=650)
    args = parser.parse_args()

    horizons = _parse_list(args.horizons)
    patterns = _parse_list(args.patterns)

    df = pd.read_parquet(args.parquet)
    df.rename_axis("date", inplace=True)

    all_tickers = sorted(df["Stock_name"].unique())
    tickers = sorted({t for t in all_tickers if any(p in t for p in patterns)})
    if not tickers:
        raise SystemExit(f"No tickers matched patterns={patterns}")

    subset_df = df[df["Stock_name"].isin(tickers)]
    returns_wide = get_returns_table(subset_df, tickers, horizons)
    stddev_wide = get_stddev_table(subset_df, tickers, horizons)

    fig = plot_risk_return_scatter_by_horizon(
        returns_wide,
        stddev_wide,
        horizons,
        width=args.width,
        height=args.height,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
