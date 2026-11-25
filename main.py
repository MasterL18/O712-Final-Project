"""
F1 Undercut Strategy & Tire Degradation Decision Support Suite.

Uses the Kaggle Formula 1 Race Data dataset (jtrotman/formula-1-race-data) and
FastF1 to model fuel-corrected tire degradation, detect tire cliffs, simulate
optimal undercut pit windows, backtest against historical results, and render
an interactive Plotly dashboard for any user-selected circuit and tire compound.

Requirements:
    pip install kagglehub fastf1 pandas numpy scikit-learn plotly

Usage:
    # Interactive mode (prompts for circuit and compound):
    python f1_undercut_strategy.py

    # Command-line mode:
    python f1_undercut_strategy.py --circuit spa --compound Hard --year 2023
"""

import argparse
import os
import sys

import fastf1
import kagglehub
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# =============================================================================
# Configuration & Constants
# =============================================================================
FUEL_START_KG = 110.0
FUEL_EFFECT_SEC_PER_KG = 0.035
GREEN_FLAG_PACE_MULTIPLIER = 1.07
CLIFF_DEG_RATE_THRESHOLD = 0.15
UNDERCUT_GAIN_THRESHOLD_SEC = 0.5

FEATURES = [
    "tire_age",
    "fuel_remaining_kg",
    "lap",
    "position",
    "circuitId",
    "compound_code",
]

COMPOUND_MAPPING = {"Soft": 0, "Medium": 1, "Hard": 2}
COMPOUND_NAMES = {v: k for k, v in COMPOUND_MAPPING.items()}

POPULAR_CIRCUITS = [
    "silverstone",
    "monaco",
    "spa",
    "monza",
    "red_bull_ring",
    "suzuka",
    "marina_bay",
    "americas",
    "interlagos",
    "albert_park",
]

# Maps Kaggle circuitRef values to FastF1 event lookup names
FASTF1_EVENT_MAP = {
    "silverstone": "Silverstone",
    "monaco": "Monaco",
    "spa": "Spa-Francorchamps",
    "monza": "Monza",
    "red_bull_ring": "Austria",
    "suzuka": "Suzuka",
    "marina_bay": "Singapore",
    "americas": "United States",
    "interlagos": "Sao Paulo",
    "albert_park": "Australia",
}


# =============================================================================
# Step 1: Data Loading & Preprocessing (Kaggle Dataset)
# =============================================================================
def get_kaggle_dataset_path(fallback_dir: str = "data/f1") -> str:
    """Download the latest Kaggle F1 dataset or fall back to a local folder."""
    try:
        dataset_path = kagglehub.dataset_download("jtrotman/formula-1-race-data")
        print(f"Path to Kaggle dataset files: {dataset_path}")
        return dataset_path
    except Exception as exc:
        print(f"Kaggle download unavailable ({exc}). Falling back to '{fallback_dir}'.")
        return fallback_dir


def load_and_prepare_data(data_dir: str):
    """
    Load Kaggle relational tables, filter to modern F1 era (2018+), remove
    pit/SC/VSC laps, and engineer fuel-corrected degradation features.
    """
    lap_times = pd.read_csv(os.path.join(data_dir, "lap_times.csv"))
    pit_stops = pd.read_csv(os.path.join(data_dir, "pit_stops.csv"))
    races = pd.read_csv(os.path.join(data_dir, "races.csv"))
    circuits = pd.read_csv(os.path.join(data_dir, "circuits.csv"))
    drivers = pd.read_csv(os.path.join(data_dir, "drivers.csv"))
    results = pd.read_csv(os.path.join(data_dir, "results.csv"))

    # Filter to modern hybrid/ground-effect era (2018+)
    races_modern = races[races["year"] >= 2018][
        ["raceId", "year", "round", "circuitId", "name"]
    ].merge(
        circuits[["circuitId", "circuitRef", "name"]],
        on="circuitId",
        suffixes=("_race", "_circuit"),
    )

    drivers["driver_name"] = drivers["forename"] + " " + drivers["surname"]

    df = (
        lap_times.merge(races_modern, on="raceId", how="inner")
        .merge(drivers[["driverId", "code", "driver_name"]], on="driverId", how="left")
    )
    df["lap_time_sec"] = df["milliseconds"] / 1000.0

    # Identify pit in-laps and out-laps
    pit_stops_clean = pit_stops[
        ["raceId", "driverId", "lap", "stop", "milliseconds"]
    ].rename(columns={"milliseconds": "pit_duration_ms"})

    df = df.merge(pit_stops_clean, on=["raceId", "driverId", "lap"], how="left")
    df["is_pit_inlap"] = df["stop"].notna()

    df = df.sort_values(["raceId", "driverId", "lap"]).reset_index(drop=True)
    df["is_pit_outlap"] = (
        df.groupby(["raceId", "driverId"])["is_pit_inlap"]
        .shift(1, fill_value=False)
        .astype(bool)
    )

    # Exclude standing starts, pit laps, and SC/VSC/rain spikes (>107% median pace)
    race_median_pace = df.groupby("raceId")["lap_time_sec"].transform("median")
    df["is_green_flag"] = (
        (df["lap"] > 1)
        & (~df["is_pit_inlap"])
        & (~df["is_pit_outlap"])
        & (df["lap_time_sec"] <= race_median_pace * GREEN_FLAG_PACE_MULTIPLIER)
    )

    # Stint number and tire age
    df["stint_number"] = df.groupby(["raceId", "driverId"])["is_pit_outlap"].cumsum() + 1
    df["tire_age"] = df.groupby(["raceId", "driverId", "stint_number"]).cumcount() + 1

    # Fuel-corrected (zero-fuel equivalent) lap times
    total_laps_per_race = df.groupby("raceId")["lap"].transform("max")
    df["fuel_remaining_kg"] = FUEL_START_KG * (1.0 - (df["lap"] / total_laps_per_race))
    df["fuel_corrected_lap_sec"] = (
        df["lap_time_sec"] - df["fuel_remaining_kg"] * FUEL_EFFECT_SEC_PER_KG
    )

    # Proxy tire compound from stint length (Kaggle lacks Pirelli compound labels)
    stint_lengths = df.groupby(["raceId", "driverId", "stint_number"])[
        "tire_age"
    ].transform("max")
    df["inferred_compound"] = pd.cut(
        stint_lengths,
        bins=[0, 15, 28, 100],
        labels=["Soft (Short Stint)", "Medium (Mid Stint)", "Hard (Long Stint)"],
    )

    # Clean green-flag laps & degradation delta relative to driver's race baseline
    clean_df = df[df["is_green_flag"]].copy()
    race_driver_baseline = clean_df.groupby(["raceId", "driverId"])[
        "fuel_corrected_lap_sec"
    ].transform("min")
    clean_df["deg_delta_sec"] = clean_df["fuel_corrected_lap_sec"] - race_driver_baseline

    print(
        f"Total modern laps: {len(df):,} | "
        f"Clean green-flag laps: {len(clean_df):,}"
    )
    return df, clean_df, races_modern, circuits, results


# =============================================================================
# Step 2: FastF1 Session Enrichment (Real Compounds & Weather)
# =============================================================================
def load_fastf1_context(
    circuit_ref: str,
    circuit_name: str,
    year: int = 2023,
    cache_dir: str = "data/fastf1_cache",
) -> dict:
    """
    Fetch modern race lap/compound and weather data from FastF1 for the
    selected circuit. Returns an empty dict gracefully if unavailable.
    """
    os.makedirs(cache_dir, exist_ok=True)
    if hasattr(fastf1.Cache, "configure"):
        fastf1.Cache.configure(cache_dir=cache_dir)
    else:
        fastf1.Cache.enable_cache(cache_dir)

    event_query = FASTF1_EVENT_MAP.get(circuit_ref, circuit_name)
    context = {}

    try:
        print(f"Loading FastF1 {year} Race session for '{event_query}'...")
        session = fastf1.get_session(year, event_query, "R")
        session.load(laps=True, telemetry=False, weather=True)

        laps = session.laps
        if laps is not None and not laps.empty:
            cols = ["LapNumber", "Driver", "Team", "Compound", "TyreLife", "LapTime"]
            available_cols = [c for c in cols if c in laps.columns]
            context["laps"] = laps[available_cols]

            if "Compound" in laps.columns and "TyreLife" in laps.columns:
                compound_summary = (
                    laps.dropna(subset=["Compound", "TyreLife"])
                    .groupby("Compound")["TyreLife"]
                    .agg(["count", "mean", "max"])
                )
                context["compound_summary"] = compound_summary

        weather = session.weather_data
        if weather is not None and not weather.empty:
            context["weather"] = {
                "avg_air_temp": float(weather["AirTemp"].mean()),
                "avg_track_temp": float(weather["TrackTemp"].mean()),
                "rainfall": bool(weather["Rainfall"].any()),
            }
    except Exception as exc:
        print(f"[FastF1] Skipping live session enrichment ({exc}).")

    return context


# =============================================================================
# Step 3: Predictive Modeling (Quantile Gradient Boosting)
# =============================================================================
def train_degradation_models(clean_df: pd.DataFrame) -> dict:
    """
    Train lower (5%), median (50%), and upper (95%) quantile
    HistGradientBoostingRegressor models to predict fuel-corrected pace loss.
    """
    model_df = (
        clean_df[
            [
                "tire_age",
                "fuel_remaining_kg",
                "lap",
                "position",
                "circuitId",
                "inferred_compound",
                "deg_delta_sec",
            ]
        ]
        .dropna()
        .copy()
    )
    model_df["compound_code"] = model_df["inferred_compound"].cat.codes

    X = model_df[FEATURES]
    y = model_df["deg_delta_sec"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {}
    for name, quantile in [("lower", 0.05), ("median", 0.50), ("upper", 0.95)]:
        reg = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_iter=200,
            max_depth=6,
            random_state=42,
        )
        reg.fit(X_train, y_train)
        models[name] = reg

    y_pred = models["median"].predict(X_test)
    print(
        f"Quantile Degradation Model -> "
        f"MAE: {mean_absolute_error(y_test, y_pred):.3f}s | "
        f"R²: {r2_score(y_test, y_pred):.3f}"
    )
    return models


# =============================================================================
# Step 4: Stint Forecast, Undercut Simulator & Historical Backtest
# =============================================================================
def forecast_stint_with_ci(
    models: dict,
    circuit_id: int,
    compound_code: int,
    total_race_laps: int,
    start_lap: int = 1,
    max_age: int = 35,
):
    """Forecast tire degradation with 90% CI and detect the tire cliff lap."""
    laps = np.arange(start_lap, start_lap + max_age)
    sim_laps = pd.DataFrame(
        {
            "tire_age": np.arange(1, max_age + 1),
            "fuel_remaining_kg": FUEL_START_KG * (1.0 - (laps / total_race_laps)),
            "lap": laps,
            "position": 5,  # Clean/semi-clean air baseline
            "circuitId": circuit_id,
            "compound_code": compound_code,
        }
    )

    sim_laps["pred_median"] = models["median"].predict(sim_laps[FEATURES])
    sim_laps["pred_lower"] = models["lower"].predict(sim_laps[FEATURES])
    sim_laps["pred_upper"] = models["upper"].predict(sim_laps[FEATURES])
    sim_laps["deg_rate"] = sim_laps["pred_median"].diff()

    cliff_candidates = sim_laps[
        (sim_laps["tire_age"] > 5) & (sim_laps["deg_rate"] > CLIFF_DEG_RATE_THRESHOLD)
    ]
    detected_cliff = (
        int(cliff_candidates["tire_age"].iloc[0])
        if not cliff_candidates.empty
        else None
    )
    return sim_laps, detected_cliff


def simulate_undercut_window(
    models: dict,
    circuit_id: int,
    compound_code: int,
    total_race_laps: int,
    traffic_penalty_sec: float = 1.5,
) -> pd.DataFrame:
    """
    Compare pitting on Lap L (fresh tires, harder compound, traffic penalty)
    against staying out 2 extra laps on old tires.
    """
    next_compound = min(compound_code + 1, 2)
    results_sim = []

    for pit_lap in range(10, max(11, total_race_laps - 10)):
        fuel_pair = [
            FUEL_START_KG * (1.0 - pit_lap / total_race_laps),
            FUEL_START_KG * (1.0 - (pit_lap + 1) / total_race_laps),
        ]
        lap_pair = [pit_lap, pit_lap + 1]

        old_tire_laps = pd.DataFrame(
            {
                "tire_age": [pit_lap, pit_lap + 1],
                "fuel_remaining_kg": fuel_pair,
                "lap": lap_pair,
                "position": [5, 5],
                "circuitId": [circuit_id, circuit_id],
                "compound_code": [compound_code, compound_code],
            }
        )
        fresh_tire_laps = pd.DataFrame(
            {
                "tire_age": [1, 2],
                "fuel_remaining_kg": fuel_pair,
                "lap": lap_pair,
                "position": [10, 10],  # Rejoins further back in traffic
                "circuitId": [circuit_id, circuit_id],
                "compound_code": [next_compound, next_compound],
            }
        )

        time_old = models["median"].predict(old_tire_laps[FEATURES]).sum()
        time_fresh = (
            models["median"].predict(fresh_tire_laps[FEATURES]).sum()
            + traffic_penalty_sec
        )
        undercut_gain = float(time_old - time_fresh)

        results_sim.append(
            {
                "pit_lap": pit_lap,
                "undercut_gain_sec": undercut_gain,
                "recommended": undercut_gain > UNDERCUT_GAIN_THRESHOLD_SEC,
            }
        )

    return pd.DataFrame(results_sim)


def run_historical_backtest(
    df: pd.DataFrame,
    results: pd.DataFrame,
    circuit_id: int,
    win_start: int,
    win_end: int,
):
    """Compare average positions gained for drivers pitting inside vs. outside the window."""
    circuit_pits = df[(df["circuitId"] == circuit_id) & (df["is_pit_inlap"])].copy()
    if circuit_pits.empty or win_start is None:
        return None

    first_pit = circuit_pits.groupby(["raceId", "driverId"])["lap"].min().reset_index()
    first_pit = first_pit.merge(
        results[["raceId", "driverId", "grid", "positionOrder"]],
        on=["raceId", "driverId"],
    )
    first_pit["grid"] = pd.to_numeric(first_pit["grid"], errors="coerce")
    first_pit["positionOrder"] = pd.to_numeric(
        first_pit["positionOrder"], errors="coerce"
    )
    first_pit["positions_gained"] = first_pit["grid"] - first_pit["positionOrder"]

    first_pit["strategy_group"] = np.where(
        (first_pit["lap"] >= win_start) & (first_pit["lap"] <= win_end),
        "Optimal Window (Model)",
        "Outside Window",
    )
    return (
        first_pit.dropna(subset=["positions_gained"])
        .groupby("strategy_group")["positions_gained"]
        .agg(["count", "mean"])
    )


# =============================================================================
# Step 5: Interactive Plotly Dashboard
# =============================================================================
def plot_strategy_dashboard(
    sim_laps: pd.DataFrame,
    undercut_df: pd.DataFrame,
    circuit_name: str,
    compound_name: str,
    detected_cliff: int | None,
    save_html: str | None = None,
    show_plot: bool = True,
) -> go.Figure:
    """Build and display (or save) the two-panel Plotly strategy dashboard."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"<b>Tire Degradation Forecast (90% CI)</b><br>"
            f"<span style='font-size:11px;color:#5f6368'>{circuit_name} {compound_name} Proxy</span>",
            "<b>Undercut Advantage Window</b><br>"
            "<span style='font-size:11px;color:#5f6368'>Net Gain/Loss vs. Pit Stop Lap</span>",
        ),
        horizontal_spacing=0.20,
    )

    # Panel 1: 90% Confidence Interval band
    fig.add_trace(
        go.Scatter(
            x=list(sim_laps["tire_age"]) + list(sim_laps["tire_age"])[::-1],
            y=list(sim_laps["pred_upper"]) + list(sim_laps["pred_lower"])[::-1],
            fill="toself",
            fillcolor="rgba(26, 115, 232, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="90% Confidence Interval",
        ),
        row=1,
        col=1,
    )

    # Panel 1: Median degradation forecast
    fig.add_trace(
        go.Scatter(
            x=sim_laps["tire_age"],
            y=sim_laps["pred_median"],
            mode="lines+markers",
            line=dict(color="#1a73e8", width=3),
            marker=dict(size=6),
            name="Median Forecast (P5 pace)",
            customdata=sim_laps["deg_rate"],
            hovertemplate=(
                "<b>Tire Age:</b> %{x} laps<br>"
                "<b>Pace Loss:</b> %{y:.2f}s<br>"
                "<b>Deg/Lap:</b> %{customdata:.2f}s<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    if detected_cliff is not None:
        fig.add_vline(
            x=detected_cliff,
            line_width=2,
            line_dash="dash",
            line_color="#ea4335",
            annotation_text=f"{compound_name} Cliff (Lap {detected_cliff})",
            annotation_position="top left",
            annotation_yshift=15,
            row=1,
            col=1,
        )

    # Panel 2: Undercut advantage window
    for rec_status, group in undercut_df.groupby("recommended"):
        color = "#34a853" if rec_status else "#9aa0a6"
        name = "Optimal Undercut Window" if rec_status else "Suboptimal Window"
        fig.add_trace(
            go.Scatter(
                x=group["pit_lap"],
                y=group["undercut_gain_sec"],
                mode="markers+lines",
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color),
                name=name,
                hovertemplate=(
                    "<b>Pit on Lap:</b> %{x}<br>"
                    "<b>Net Undercut Advantage:</b> %{y:+.2f}s<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    fig.add_hline(
        y=UNDERCUT_GAIN_THRESHOLD_SEC,
        line_width=1.5,
        line_dash="dot",
        line_color="#34a853",
        annotation_text=f"Optimal Trigger (>{UNDERCUT_GAIN_THRESHOLD_SEC}s Gained)",
        annotation_position="bottom right",
        annotation_xshift=-15,
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Tire Age (Laps Completed)", row=1, col=1, title_standoff=15)
    fig.update_yaxes(title_text="Seconds Lost vs. Baseline Pace", row=1, col=1, title_standoff=15)
    fig.update_xaxes(title_text="Pit Stop Lap (Trigger Lap)", row=1, col=2, title_standoff=15)
    fig.update_yaxes(title_text="Estimated Net Undercut Gain (Seconds)", row=1, col=2, title_standoff=15)

    fig.update_layout(
        title=f"<b>F1 Strategy Decision Support Suite: {circuit_name} ({compound_name} Compound)</b>",
        title_font=dict(size=18, family="Arial"),
        template="plotly_white",
        height=580,
        width=1100,
        margin=dict(t=140, b=120, l=80, r=40),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
        ),
    )

    if save_html:
        fig.write_html(save_html)
        print(f"Dashboard saved to: {save_html}")
    if show_plot:
        fig.show()

    return fig


# =============================================================================
# Step 6: Universal Strategy Pipeline
# =============================================================================
def run_race_strategy_pipeline(
    circuit_ref: str,
    compound_name: str,
    df: pd.DataFrame,
    races_modern: pd.DataFrame,
    circuits: pd.DataFrame,
    results: pd.DataFrame,
    models: dict,
    fastf1_year: int = 2023,
    use_fastf1: bool = True,
    traffic_penalty_sec: float = 1.5,
    save_html: str | None = None,
    show_plot: bool = True,
):
    """Execute the full predictive, prescriptive, backtest, and visual pipeline."""
    circuit_subset = circuits[circuits["circuitRef"] == circuit_ref]
    if circuit_subset.empty:
        print(f"Circuit '{circuit_ref}' not found in database.")
        return

    circuit_id = int(circuit_subset["circuitId"].values[0])
    circuit_name = str(circuit_subset["name"].values[0])

    circuit_races = races_modern[races_modern["circuitId"] == circuit_id]
    if circuit_races.empty:
        print(f"No modern race records (2018+) found for {circuit_name}.")
        return

    latest_race_id = circuit_races["raceId"].max()
    total_race_laps = int(df[df["raceId"] == latest_race_id]["lap"].max())
    compound_code = COMPOUND_MAPPING[compound_name]

    print(f"\n=== F1 Undercut Strategy Engine: {circuit_name} ===")
    print(
        f"Circuit Ref: {circuit_ref} | Circuit ID: {circuit_id} | "
        f"Typical Race Laps: {total_race_laps}"
    )
    print(f"Selected Compound: {compound_name} (Code: {compound_code})\n")

    # Optional FastF1 real-world compound & weather context
    if use_fastf1:
        f1_ctx = load_fastf1_context(circuit_ref, circuit_name, year=fastf1_year)
        if "weather" in f1_ctx:
            w = f1_ctx["weather"]
            print(
                f"[FastF1 {fastf1_year}] Avg Air Temp: {w['avg_air_temp']:.1f}°C | "
                f"Avg Track Temp: {w['avg_track_temp']:.1f}°C | "
                f"Rain: {'Yes' if w['rainfall'] else 'No'}"
            )
        if "compound_summary" in f1_ctx:
            print(f"\n[FastF1 {fastf1_year}] Observed Pirelli Compound Stint Lengths:")
            print(f1_ctx["compound_summary"].round(1))
            print()

    # Predictive stint forecast & cliff detection
    sim_laps, detected_cliff = forecast_stint_with_ci(
        models=models,
        circuit_id=circuit_id,
        compound_code=compound_code,
        total_race_laps=total_race_laps,
    )
    if detected_cliff is not None:
        print(f"[Predictive] {compound_name} Tire Cliff detected at: Lap {detected_cliff}")
    else:
        print(
            f"[Predictive] No clear tire cliff signature detected on "
            f"{compound_name}s within 35 laps."
        )

    # Prescriptive undercut window simulation
    undercut_df = simulate_undercut_window(
        models=models,
        circuit_id=circuit_id,
        compound_code=compound_code,
        total_race_laps=total_race_laps,
        traffic_penalty_sec=traffic_penalty_sec,
    )
    optimal_window = undercut_df[undercut_df["recommended"]]
    best_pit_lap = None
    best_gain_sec = None

    if not optimal_window.empty:
        win_start = int(optimal_window["pit_lap"].min())
        win_end = int(optimal_window["pit_lap"].max())
        best_lap = optimal_window.loc[optimal_window["undercut_gain_sec"].idxmax()]
        best_pit_lap = int(best_lap["pit_lap"])
        best_gain_sec = float(best_lap["undercut_gain_sec"])
        print(f"[Prescriptive] Target Undercut Window: Laps {win_start} to {win_end}")
        print(
            f"[Prescriptive] Peak Trigger Lap: Lap {best_pit_lap} "
            f"(+{best_gain_sec:.2f}s gained)"
        )
    else:
        win_start, win_end = None, None
        print(
            "[Prescriptive] Undercut benefit is low. "
            "No recommended strategic early pit window found."
        )

    # Historical backtest validation
    verification = run_historical_backtest(
        df=df,
        results=results,
        circuit_id=circuit_id,
        win_start=win_start,
        win_end=win_end,
    )
    if verification is not None:
        print("\n--- Historical Backtesting Results (2018+) ---")
        print(verification)
    else:
        print(
            "\n[Validation] Historical backtesting skipped "
            "(insufficient pit stop records or no strategy window)."
        )

    # Interactive Plotly dashboard
    figure = plot_strategy_dashboard(
        sim_laps=sim_laps,
        undercut_df=undercut_df,
        circuit_name=circuit_name,
        compound_name=compound_name,
        detected_cliff=detected_cliff,
        save_html=save_html,
        show_plot=show_plot,
    )
    return {
        "figure": figure,
        "undercut": undercut_df,
        "verification": verification,
        "detected_cliff": detected_cliff,
        "total_race_laps": total_race_laps,
        "win_start": win_start,
        "win_end": win_end,
        "best_pit_lap": best_pit_lap,
        "best_gain_sec": best_gain_sec,
    }


# =============================================================================
# CLI & Interactive Entry Point
# =============================================================================
def prompt_user_selection(valid_circuits: list[str]) -> tuple[str, str]:
    """Prompt the user to pick a circuit and tire compound interactively."""
    featured = [c for c in POPULAR_CIRCUITS if c in valid_circuits]
    print("\nPopular modern circuits:", ", ".join(featured))
    print(f"(Total modern circuits available: {len(valid_circuits)} — type 'list' to see all)")

    while True:
        choice = input("Select circuit (default: silverstone): ").strip().lower()
        if not choice:
            circuit_ref = "silverstone"
            break
        if choice == "list":
            print(", ".join(sorted(valid_circuits)))
            continue
        if choice in valid_circuits:
            circuit_ref = choice
            break
        print(f"Invalid circuit '{choice}'. Please try again.")

    compound_lookup = {k.lower(): k for k in COMPOUND_MAPPING}
    while True:
        comp_input = input("Select tire compound [Soft / Medium / Hard] (default: Soft): ").strip().lower()
        if not comp_input:
            compound_name = "Soft"
            break
        if comp_input in compound_lookup:
            compound_name = compound_lookup[comp_input]
            break
        print("Invalid compound. Choose Soft, Medium, or Hard.")

    return circuit_ref, compound_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate F1 undercut pit windows by circuit and tire compound."
    )
    parser.add_argument(
        "--circuit",
        type=str,
        default=None,
        help="Circuit reference (e.g., silverstone, spa, monaco, monza).",
    )
    parser.add_argument(
        "--compound",
        type=str,
        choices=["Soft", "Medium", "Hard"],
        default=None,
        help="Starting tire compound proxy.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="FastF1 season year for live compound/weather context (default: 2023).",
    )
    parser.add_argument(
        "--no-fastf1",
        action="store_true",
        help="Skip FastF1 API fetch and use Kaggle data only.",
    )
    parser.add_argument(
        "--traffic-penalty",
        type=float,
        default=1.5,
        help="Estimated traffic penalty in seconds when rejoining after a pit stop (default: 1.5).",
    )
    parser.add_argument(
        "--save-html",
        type=str,
        default=None,
        help="Optional path to save the interactive Plotly dashboard as an HTML file.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the Plotly dashboard in a browser window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = get_kaggle_dataset_path()
    df, clean_df, races_modern, circuits, results = load_and_prepare_data(data_dir)
    models = train_degradation_models(clean_df)

    valid_circuits = sorted(races_modern["circuitRef"].dropna().unique().tolist())

    # CLI single-run mode
    if args.circuit is not None or args.compound is not None:
        circuit_ref = (args.circuit or "silverstone").lower()
        compound_name = args.compound or "Soft"
        run_race_strategy_pipeline(
            circuit_ref=circuit_ref,
            compound_name=compound_name,
            df=df,
            races_modern=races_modern,
            circuits=circuits,
            results=results,
            models=models,
            fastf1_year=args.year,
            use_fastf1=not args.no_fastf1,
            traffic_penalty_sec=args.traffic_penalty,
            save_html=args.save_html,
            show_plot=not args.no_show,
        )
        return

    # Interactive loop so users can test multiple tracks/compounds without retraining
    while True:
        circuit_ref, compound_name = prompt_user_selection(valid_circuits)
        run_race_strategy_pipeline(
            circuit_ref=circuit_ref,
            compound_name=compound_name,
            df=df,
            races_modern=races_modern,
            circuits=circuits,
            results=results,
            models=models,
            fastf1_year=args.year,
            use_fastf1=not args.no_fastf1,
            traffic_penalty_sec=args.traffic_penalty,
            save_html=args.save_html,
            show_plot=not args.no_show,
        )
        again = input("\nRun another simulation? [y/N]: ").strip().lower()
        if again not in ("y", "yes"):
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited by user.")
        sys.exit(0)

