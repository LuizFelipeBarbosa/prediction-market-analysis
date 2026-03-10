import duckdb
from pathlib import Path
from src.analysis.kalshi.util.categories import CATEGORY_SQL

base_dir = Path("/Users/lfpmb/Documents/prediction-market-analysis")
trades_dir = base_dir / "data" / "kalshi" / "trades"
markets_dir = base_dir / "data" / "kalshi" / "markets"

con = duckdb.connect()

df = con.execute(f"""
    WITH market_base AS (
        SELECT 
            ticker,
            title,
            result,
            close_time,
            {CATEGORY_SQL} AS category
        FROM '{markets_dir}/*.parquet'
        WHERE result IN ('yes', 'no')
            AND (ticker LIKE '%MENTION%' OR event_ticker LIKE '%MENTION%')
        LIMIT 100
    )
    SELECT 
        m.ticker,
        m.title,
        m.result,
        m.category,
        max_by(t.yes_price, t.created_time) FILTER (WHERE t.created_time <= m.close_time - INTERVAL 7 DAY) as price_7d,
        max_by(t.yes_price, t.created_time) FILTER (WHERE t.created_time <= m.close_time - INTERVAL 3 DAY) as price_3d,
        max_by(t.yes_price, t.created_time) FILTER (WHERE t.created_time <= m.close_time - INTERVAL 1 DAY) as price_1d,
        max_by(t.yes_price, t.created_time) FILTER (WHERE t.created_time <= m.close_time - INTERVAL 6 HOUR) as price_6h,
        max_by(t.yes_price, t.created_time) FILTER (WHERE t.created_time <= m.close_time - INTERVAL 1 HOUR) as price_1h,
        max_by(t.yes_price, t.created_time) FILTER (WHERE t.created_time <= m.close_time - INTERVAL 10 MINUTE) as price_10m
    FROM market_base m
    LEFT JOIN '{trades_dir}/*.parquet' t ON m.ticker = t.ticker
    GROUP BY m.ticker, m.title, m.result, m.category
""").df()

print(df.head())
print(df.describe())
