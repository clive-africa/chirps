import polars as pl
from pathlib import Path


def load_weather_station_csvs(directory: str = "weather_station_dta") -> pl.DataFrame:
	base_dir = Path(__file__).resolve().parent
	csv_dir = base_dir / directory

	csv_files = sorted(csv_dir.glob("*.csv"))
	if not csv_files:
		return pl.DataFrame()

	dataframes: list[pl.DataFrame] = []
	expected_columns: list[str] | None = None

	for file_path in csv_files:
		df = pl.read_csv(file_path)

		if expected_columns is None:
			expected_columns = df.columns
			dataframes.append(df)
			continue

		if df.columns == expected_columns:
			dataframes.append(df)

	if not dataframes:
		return pl.DataFrame()

	return pl.concat(dataframes, how="vertical")


weather_station_df = load_weather_station_csvs()



