import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio

from returns import get_sharpe_table


def plot_sharpe_ratio_grouped_bar(sharpe_wide, title=None, width=1100, height=550):
    tmp = sharpe_wide.reset_index()
    ticker_col = tmp.columns[0]

    plot_df = (
        tmp.rename(columns={ticker_col: "ticker"})
        .melt(id_vars="ticker", var_name="horizon", value_name="sharpe")
        .dropna()
    )

    fig = px.bar(
        plot_df,
        x="ticker",
        y="sharpe",
        color="horizon",
        barmode="group",
        title=title or "Sharpe Ratio by Ticker and Horizon",
    )
    fig.update_layout(
        width=width,
        height=height,
        xaxis_title="Ticker",
        yaxis_title="Sharpe Ratio",
        legend_title="Horizon",
    )
    return fig


def _parse_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Sharpe ratio bar chart to docs/ as HTML.")
    parser.add_argument("--parquet", default="ETFs_data.parquet.gzip", help="Input parquet file")
    parser.add_argument("--out", default="docs/sharpe_ratio.html", help="Output HTML path")
    parser.add_argument("--horizons", default="6M,1Y,3Y,5Y", help="Comma-separated horizons")
    parser.add_argument(
        "--patterns",
        default="VFV,XUU,VOO,BRK,QQ,XEF,XEQ",
        help="Comma-separated substrings to select tickers (matched against Stock_name)",
    )
    parser.add_argument("--risk-free", type=float, default=0.0, help="Annual risk-free rate (e.g. 0.03)")
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=550)
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
    sharpe_wide = get_sharpe_table(
        subset_df,
        tickers,
        horizons,
        risk_free_rate_annual=args.risk_free,
    )

    fig = plot_sharpe_ratio_grouped_bar(
        sharpe_wide,
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
