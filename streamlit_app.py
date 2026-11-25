import contextlib
import io

import streamlit as st

import main as strategy


st.set_page_config(
    page_title="F1 Undercut Strategy",
    page_icon="🏁",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_race_data():
    data_dir = strategy.get_kaggle_dataset_path()
    return strategy.load_and_prepare_data(data_dir)


if "race_data" not in st.session_state:
    st.session_state.race_data = None
if "models" not in st.session_state:
    st.session_state.models = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analysis_output" not in st.session_state:
    st.session_state.analysis_output = ""
if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = None
if "data_error" not in st.session_state:
    st.session_state.data_error = None


st.title("F1 Undercut Strategy")
st.caption("Tire degradation forecasts, pit-window simulation, and historical checks")

with st.sidebar:
    st.header("Race data")
    st.caption("The app loads the Kaggle race tables on request and reuses them for this session.")
    if st.button("Load race data", type="primary", use_container_width=True):
        captured = io.StringIO()
        try:
            with st.spinner("Loading historical race data..."):
                with contextlib.redirect_stdout(captured):
                    st.session_state.race_data = load_race_data()
            st.session_state.models = None
            st.session_state.analysis_result = None
            st.session_state.analysis_output = ""
            st.session_state.analysis_error = None
            st.session_state.data_error = None
        except Exception as exc:
            st.session_state.data_error = str(exc)
        st.session_state.data_output = captured.getvalue()

    if st.session_state.race_data is not None:
        _, clean_df, _, _, _ = st.session_state.race_data
        st.success(f"Loaded {len(clean_df):,} clean green-flag laps")
    elif st.session_state.data_error:
        st.error(st.session_state.data_error)
        st.caption("Check Kaggle access or place the required CSV tables in data/f1.")
    else:
        st.info("Load the race data to configure an analysis.")

if st.session_state.race_data is None:
    st.info("Load the historical dataset from the sidebar to begin.")
    if st.session_state.get("data_output"):
        with st.expander("Data loading details"):
            st.code(st.session_state.data_output)
    st.stop()


df, clean_df, races_modern, circuits, results = st.session_state.race_data
valid_circuits = sorted(races_modern["circuitRef"].dropna().unique().tolist())
if not valid_circuits:
    st.error("No circuits with race data from 2018 onward were found.")
    st.stop()

circuit_names = dict(zip(circuits["circuitRef"], circuits["name"]))
default_circuit = "silverstone" if "silverstone" in valid_circuits else valid_circuits[0]

with st.form("strategy_controls"):
    st.subheader("Simulation settings")
    control_left, control_right = st.columns(2)
    with control_left:
        circuit_ref = st.selectbox(
            "Circuit",
            valid_circuits,
            index=valid_circuits.index(default_circuit),
            format_func=lambda value: circuit_names.get(value, value),
        )
        compound_name = st.selectbox("Starting tire compound", ["Soft", "Medium", "Hard"])
    with control_right:
        traffic_penalty = st.number_input(
            "Pit rejoin traffic penalty (seconds)",
            min_value=0.0,
            value=1.5,
            step=0.1,
        )
    run_analysis = st.form_submit_button("Run strategy analysis", type="primary")

if run_analysis:
    captured = io.StringIO()
    st.session_state.analysis_result = None
    st.session_state.analysis_error = None
    try:
        if st.session_state.models is None:
            with st.spinner("Training degradation models. The first run may take a little while..."):
                with contextlib.redirect_stdout(captured):
                    st.session_state.models = strategy.train_degradation_models(clean_df)

        with st.spinner("Simulating pit windows and preparing the dashboard..."):
            with contextlib.redirect_stdout(captured):
                st.session_state.analysis_result = strategy.run_race_strategy_pipeline(
                    circuit_ref=circuit_ref,
                    compound_name=compound_name,
                    df=df,
                    races_modern=races_modern,
                    circuits=circuits,
                    results=results,
                    models=st.session_state.models,
                    traffic_penalty_sec=traffic_penalty,
                    show_plot=False,
                )
        if st.session_state.analysis_result is None:
            st.session_state.analysis_error = "The selected circuit could not be analyzed with this dataset."
    except Exception as exc:
        st.session_state.analysis_error = str(exc)
    st.session_state.analysis_output = captured.getvalue()


if st.session_state.analysis_error:
    st.error(st.session_state.analysis_error)

analysis_result = st.session_state.analysis_result
if analysis_result is not None:
    best_gain = analysis_result["best_gain_sec"]
    best_gain_value = f"{best_gain:+.2f} s" if best_gain is not None else "None found"
    window_start = analysis_result["win_start"]
    window_end = analysis_result["win_end"]
    window_value = (
        f"Laps {window_start}-{window_end}"
        if window_start is not None
        else "No window"
    )

    metric_laps, metric_cliff, metric_window, metric_gain = st.columns(4)
    metric_laps.metric("Race distance", f"{analysis_result['total_race_laps']} laps")
    metric_cliff.metric(
        "Tire cliff",
        f"Lap {analysis_result['detected_cliff']}"
        if analysis_result["detected_cliff"] is not None
        else "Not detected",
    )
    metric_window.metric("Recommended window", window_value)
    metric_gain.metric("Peak undercut gain", best_gain_value)

    st.plotly_chart(analysis_result["figure"], use_container_width=True)
    st.download_button(
        "Download interactive chart",
        data=analysis_result["figure"].to_html(full_html=True, include_plotlyjs="cdn"),
        file_name=f"{circuit_ref}_{compound_name.lower()}_strategy.html",
        mime="text/html",
    )

    detail_tab, backtest_tab = st.tabs(["Pit-window estimates", "Historical backtest"])
    with detail_tab:
        st.dataframe(analysis_result["undercut"], use_container_width=True, hide_index=True)
    with backtest_tab:
        verification = analysis_result["verification"]
        if verification is None:
            st.info("Backtest skipped because pit-stop records or a recommended window were unavailable.")
        else:
            st.dataframe(verification, use_container_width=True)

if st.session_state.analysis_output:
    with st.expander("Analysis details"):
        st.code(st.session_state.analysis_output)