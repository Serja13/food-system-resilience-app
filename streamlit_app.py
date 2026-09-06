import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import snowflake.connector
import streamlit as st


st.set_page_config(
    page_title="After the Shock",
    page_icon="🌾",
    layout="wide",
)

COLORS = {
    "plum": "#4B1735",
    "rose": "#B77A91",
    "cream": "#FAF4F6",
    "gold": "#D7A84B",
    "green": "#4F7A65",
    "red": "#B94A48",
    "gray": "#64748B",
}

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {COLORS['cream']}; }}
    .block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; }}
    h1, h2, h3 {{ color: {COLORS['plum']}; }}
    [data-testid="stMetric"] {{
        background: white;
        border: 1px solid #eadce2;
        border-radius: 12px;
        padding: 14px;
    }}
    .callout {{
        background: white;
        border-left: 5px solid {COLORS['plum']};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: .5rem 0 1rem 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


EVENT_SQL = """
SELECT
    ISO3, AREA_CODE_M49, COUNTRY, FAOSTAT_AREA, EVENT_YEAR, EVENT_TYPES,
    TOTAL_FOCUS_EVENTS, DROUGHT_EVENTS, WILDFIRE_EVENTS,
    EXTREME_TEMPERATURE_EVENTS, FLOOD_EVENTS, STORM_EVENTS,
    TOTAL_AFFECTED_REPORTED, TOTAL_DEATHS_REPORTED,
    TOTAL_DAMAGE_ADJUSTED_000_USD, BASELINE_YEAR_COUNT, BASELINE_INDEX,
    TREND_YEAR_COUNT, PRE_EVENT_TREND_SLOPE, PRE_EVENT_TREND_INTERCEPT,
    EXPECTED_EVENT_YEAR_INDEX, EXPECTED_YEAR_1_INDEX,
    EVENT_YEAR_INDEX, YEAR_1_INDEX, YEAR_2_INDEX, YEAR_3_INDEX,
    WORST_INDEX_T_TO_T1, MAX_PRODUCTION_YEAR, EVENT_YEAR_CHANGE_PERCENT,
    WORST_CHANGE_PERCENT_T_TO_T1, EVENT_YEAR_DETRENDED_CHANGE_PERCENT,
    YEAR_1_DETRENDED_CHANGE_PERCENT, WORST_DETRENDED_CHANGE_PERCENT_T_TO_T1,
    HAS_FULL_3_YEAR_FOLLOWUP,
    RECOVERY_YEARS, RECOVERY_STATUS, RESILIENCE_CATEGORY,
    FOOD_INSECURITY_PERCENT, UNDERNOURISHMENT_PERCENT,
    FOOD_INSECURITY_IS_UPPER_BOUND, UNDERNOURISHMENT_IS_UPPER_BOUND
FROM FAOSTAT_DB.ANALYTICS.VW_EVENT_PRODUCTION_RESILIENCE
WHERE BASELINE_YEAR_COUNT >= 2
  AND EVENT_YEAR_INDEX IS NOT NULL
"""

PRODUCTION_SQL = """
SELECT AREA_CODE_M49, AREA, YEAR, FOOD_PRODUCTION_INDEX
FROM FAOSTAT_DB.ANALYTICS.VW_FOOD_PRODUCTION_YEAR
"""

POPULATION_SQL = """
SELECT AREA_CODE_M49, YEAR, VALUE * 1000 AS POPULATION
FROM FAOSTAT_DB.CLEAN.VW_POPULATION
WHERE ITEM = 'Population - Est. & Proj.'
  AND ELEMENT = 'Total Population - Both sexes'
  AND YEAR BETWEEN 2010 AND 2023
  AND VALUE IS NOT NULL
"""

HAZARD_SQL = """
SELECT DISASTER_NUMBER, ISO3, COUNTRY, START_YEAR, DISASTER_TYPE,
       TOTAL_AFFECTED, TOTAL_DEATHS, TOTAL_DAMAGE_ADJUSTED_000_USD
FROM FAOSTAT_DB.CLEAN.VW_EMDAT_EVENTS
WHERE IS_ANALYSIS_YEAR = TRUE
  AND IS_FOCUS_EVENT = TRUE
"""

HAZARDS = {
    "Drought": "DROUGHT_EVENTS",
    "Wildfire": "WILDFIRE_EVENTS",
    "Extreme temperature": "EXTREME_TEMPERATURE_EVENTS",
    "Flood": "FLOOD_EVENTS",
    "Storm": "STORM_EVENTS",
}


def secret_value(key: str, default: str = "") -> str:
    try:
        section = st.secrets.get("snowflake", {})
        return str(section.get(key, default))
    except Exception:
        return default


@st.cache_resource(show_spinner=False)
def connect_snowflake(
    account: str,
    user: str,
    password: str,
    warehouse: str,
    database: str,
    role: str,
):
    return snowflake.connector.connect(
        account=account,
        user=user,
        password=password,
        warehouse=warehouse,
        database=database,
        schema="ANALYTICS",
        role=role,
        client_session_keep_alive=True,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def query_dataframe(_connection: Any, sql: str) -> pd.DataFrame:
    cursor = _connection.cursor()
    try:
        cursor.execute(sql)
        columns = [column[0] for column in cursor.description]
        return pd.DataFrame(cursor.fetchall(), columns=columns)
    finally:
        cursor.close()


def vulnerability_band(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    value = float(value)
    if value < 2.5:
        return "Extremely low (<2.5%)"
    if value < 5:
        return "Very low (2.5% to <5%)"
    if value < 20:
        return "Moderately low (5% to <20%)"
    if value < 35:
        return "Moderately high (20% to <35%)"
    return "Very high (35%+)"


def percent(value: Any, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "No data"
    return f"{float(value):.{digits}f}%"


def fmt_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "No data"
    return f"{float(value):,.0f}"


def friendly_status(value: Any) -> str:
    if value is None or pd.isna(value):
        return "No data"
    return str(value).replace("_", " ").title()


def select_country_focus_event(
    events: pd.DataFrame,
    iso3: str,
    shock_column: str,
) -> Any:
    country_events = events[events["ISO3"].eq(iso3)].copy()
    if country_events.empty:
        return None
    measured_events = country_events.dropna(subset=[shock_column])
    if measured_events.empty:
        return country_events.sort_values("EVENT_YEAR").iloc[-1]
    # The lowest value is the deepest production drop versus the selected benchmark.
    return measured_events.loc[measured_events[shock_column].idxmin()]


def render_country_snapshot(
    events: pd.DataFrame,
    iso3: str,
    shock_column: str,
    shock_label: str,
) -> None:
    row = select_country_focus_event(events, iso3, shock_column)
    if row is None:
        return

    country = str(row["COUNTRY"])
    event_year = int(row["EVENT_YEAR"])
    event_types = str(row["EVENT_TYPES"])
    shock = row[shock_column]
    recovery_status = friendly_status(row["RECOVERY_STATUS"])

    st.markdown(f"### {country}")
    st.caption(
        "Most severe measured production shock among this country's event years "
        "that match the current filters."
    )
    c1, c2, c3, c4, c5 = st.columns([0.8, 1.5, 1.2, 1.3, 1.1])
    c1.metric("Year", str(event_year))
    c2.metric("Climate event(s)", event_types)
    c3.metric(shock_label, percent(shock))
    c4.metric("Recovery outcome", recovery_status)
    c5.metric("Undernourishment", percent(row["UNDERNOURISHMENT_PERCENT"]))

    if pd.isna(shock):
        shock_text = "does not have enough information to calculate the selected production measure"
    elif float(shock) < 0:
        shock_text = f"was {abs(float(shock)):.1f}% below its comparison level"
    else:
        shock_text = f"was {float(shock):.1f}% above its comparison level"

    status = str(row["RECOVERY_STATUS"])
    if status == "MAINTAINED":
        recovery_text = "Production remained at or above the project's recovery threshold."
    elif status == "RECOVERED" and pd.notna(row["RECOVERY_YEARS"]):
        recovery_text = f"It returned to the recovery threshold within {int(row['RECOVERY_YEARS'])} year(s)."
    elif status == "NOT_RECOVERED_WITHIN_3_YEARS":
        recovery_text = "It had not returned to the recovery threshold within three years."
    else:
        recovery_text = "A complete three-year recovery outcome is not available."

    st.markdown(
        f'<div class="callout"><b>What happened:</b> During or immediately after '
        f"{event_types.lower()} in {event_year}, {country}'s food-production index {shock_text}. "
        f"{recovery_text}<br><br><b>Reported event impact:</b> "
        f"{fmt_number(row['TOTAL_AFFECTED_REPORTED'])} people affected and "
        f"{fmt_number(row['TOTAL_DEATHS_REPORTED'])} deaths.</div>",
        unsafe_allow_html=True,
    )


def render_production_timeline(
    row: Any,
    production: pd.DataFrame,
    chart_key: str,
) -> None:
    event_year = int(row["EVENT_YEAR"])
    series = production[
        production["AREA_CODE_M49"].astype(str).eq(str(row["AREA_CODE_M49"]))
        & production["YEAR"].between(event_year - 5, event_year + 3)
    ].copy()
    series = series.sort_values("YEAR")
    if series.empty:
        st.info("No annual production series is available for this selected case.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series["YEAR"],
            y=series["FOOD_PRODUCTION_INDEX"],
            mode="lines+markers",
            name="Actual production",
            line=dict(color=COLORS["plum"], width=3),
            marker=dict(size=8),
        )
    )

    has_trend = (
        pd.notna(row["PRE_EVENT_TREND_SLOPE"])
        and pd.notna(row["PRE_EVENT_TREND_INTERCEPT"])
        and pd.notna(row["TREND_YEAR_COUNT"])
        and float(row["TREND_YEAR_COUNT"]) >= 4
    )
    if has_trend:
        series["EXPECTED_PRODUCTION"] = (
            float(row["PRE_EVENT_TREND_INTERCEPT"])
            + float(row["PRE_EVENT_TREND_SLOPE"]) * series["YEAR"]
        )
        fig.add_trace(
            go.Scatter(
                x=series["YEAR"],
                y=series["EXPECTED_PRODUCTION"],
                mode="lines",
                name="Expected from pre-event trend",
                line=dict(color=COLORS["gold"], width=3, dash="dash"),
            )
        )

    if pd.notna(row["BASELINE_INDEX"]):
        fig.add_hline(
            y=float(row["BASELINE_INDEX"]) * 0.95,
            line_dash="dot",
            line_color=COLORS["gray"],
            annotation_text="95% recovery threshold",
            annotation_position="bottom right",
        )
    fig.add_vrect(
        x0=event_year,
        x1=event_year + 3,
        fillcolor=COLORS["rose"],
        opacity=0.10,
        line_width=0,
        annotation_text="Event and recovery window",
        annotation_position="top left",
    )
    fig.add_vline(x=event_year, line_dash="dash", line_color=COLORS["red"])
    fig.update_layout(
        title=f"Actual production versus expected path: {row['COUNTRY']}, {event_year}",
        xaxis_title="Year",
        yaxis_title="Food-production index (2014–2016 = 100)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        margin=dict(t=95),
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)
    if has_trend:
        st.caption(
            "The expected path extends the five-year pre-event trend. The shaded area is the "
            "project's three-year recovery window; association does not by itself prove causation."
        )
    else:
        st.caption(
            "The expected trend is unavailable because fewer than four usable pre-event observations "
            "were available. The actual production series and recovery threshold are still shown."
        )


def render_country_takeaway(row: Any, shock_column: str) -> None:
    country = str(row["COUNTRY"])
    event_year = int(row["EVENT_YEAR"])
    event_types = str(row["EVENT_TYPES"]).lower()
    shock = row[shock_column]
    undernourishment = row["UNDERNOURISHMENT_PERCENT"]
    status = str(row["RECOVERY_STATUS"])

    st.markdown("#### What this result suggests")
    if pd.isna(shock):
        st.info(
            f"The available production history is not sufficient to classify {country}'s response "
            f"to {event_types} in {event_year}."
        )
    elif float(shock) <= -10 and pd.notna(undernourishment) and float(undernourishment) >= 20 \
            and status == "NOT_RECOVERED_WITHIN_3_YEARS":
        st.error(
            f"**High-concern pattern.** Production fell {abs(float(shock)):.1f}% below its comparison "
            f"level, undernourishment was {float(undernourishment):.1f}%, and production had not "
            "returned to the recovery threshold within three years. This combination suggests a "
            "food system that may have limited capacity to absorb the shock."
        )
    elif float(shock) <= -10 and status == "RECOVERED":
        recovery_years = (
            f" within {int(row['RECOVERY_YEARS'])} year(s)"
            if pd.notna(row["RECOVERY_YEARS"])
            else ""
        )
        st.warning(
            f"**Significant but temporary production shock.** Production fell "
            f"{abs(float(shock)):.1f}% below its comparison level but returned to the recovery "
            f"threshold{recovery_years}."
        )
    elif float(shock) <= -10 and status == "NOT_RECOVERED_WITHIN_3_YEARS":
        st.warning(
            f"**Persistent production disruption.** Production fell {abs(float(shock)):.1f}% below "
            "its comparison level and had not returned to the recovery threshold within three years. "
            "Food vulnerability should be considered separately when interpreting the level of concern."
        )
    elif pd.notna(undernourishment) and float(undernourishment) >= 20:
        st.warning(
            f"**High underlying food vulnerability, but no major measured production shock.** "
            f"Undernourishment was {float(undernourishment):.1f}%, while the selected production "
            "measure did not fall at least 10% below its comparison level."
        )
    elif status == "MAINTAINED":
        st.success(
            "**Production held up relatively well.** This event-year remained above the project's "
            "recovery threshold and did not meet the high-concern rule."
        )
    else:
        st.info(
            "**Mixed result.** The case does not meet the high-concern rule, but it should be read "
            "alongside the production timeline and available food-vulnerability information."
        )

    caveats = ["This is an observed pattern and does not prove that the disaster caused the production change."]
    if not (pd.notna(row["HAS_FULL_3_YEAR_FOLLOWUP"]) and bool(row["HAS_FULL_3_YEAR_FOLLOWUP"])):
        caveats.append("A complete three-year follow-up period is not available.")
    if pd.notna(row["UNDERNOURISHMENT_IS_UPPER_BOUND"]) and bool(row["UNDERNOURISHMENT_IS_UPPER_BOUND"]):
        caveats.append("The undernourishment value is an upper-bound estimate rather than an exact value.")
    st.caption(" ".join(caveats))


def render_priority_explorer(
    events: pd.DataFrame,
    production: pd.DataFrame,
    shock_column: str,
    shock_label: str,
) -> None:
    shock_threshold = -10
    undernourishment_threshold = 20
    priority = events[
        events[shock_column].le(shock_threshold)
        & events["UNDERNOURISHMENT_PERCENT"].ge(undernourishment_threshold)
        & events["RECOVERY_STATUS"].eq("NOT_RECOVERED_WITHIN_3_YEARS")
    ].copy()
    priority = priority.sort_values(
        [shock_column, "UNDERNOURISHMENT_PERCENT", "COUNTRY", "EVENT_YEAR"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)

    st.markdown("### Priority Country Explorer")
    st.write(
        "These cases combine a major production shock, substantial undernourishment, "
        "and no observed recovery within three years."
    )
    with st.expander("How a priority case is defined"):
        st.markdown(
            f"- **Production shock:** at least 10% below the selected comparison measure  \n"
            f"- **Food vulnerability:** undernourishment of at least {undernourishment_threshold}%  \n"
            "- **Recovery:** did not return to 95% of the pre-event baseline within three years  \n\n"
            f"The production rule currently uses **{shock_label.lower()}**. These are transparent "
            "project rules, not an agency-defined risk score."
        )

    c1, c2 = st.columns(2)
    c1.metric("Priority country-event cases", f"{len(priority):,}")
    c2.metric("Countries represented", f"{priority['ISO3'].nunique():,}")

    if priority.empty:
        st.info("No cases meet all three priority rules under the current dashboard filters.")
        return

    display = pd.DataFrame(
        {
            "Country": priority["COUNTRY"],
            "Year": priority["EVENT_YEAR"].astype(int),
            "Climate event(s)": priority["EVENT_TYPES"],
            "Production shock (%)": priority[shock_column],
            "Undernourishment (%)": priority["UNDERNOURISHMENT_PERCENT"],
            "Recovery": priority["RECOVERY_STATUS"].map(friendly_status),
            "People affected": priority["TOTAL_AFFECTED_REPORTED"],
        }
    )
    table_event = st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        height=min(420, 38 + len(display) * 35),
        column_config={
            "Production shock (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Undernourishment (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "People affected": st.column_config.NumberColumn(format="localized"),
        },
        key=f"priority_country_table_{shock_column}",
        on_select="rerun",
        selection_mode="single-row",
    )
    st.caption("Select a row to open that event's summary.")

    selected_rows = table_event.selection.rows
    if not selected_rows:
        return

    row = priority.iloc[selected_rows[0]]
    event_year = int(row["EVENT_YEAR"])
    st.markdown(f"#### Selected case: {row['COUNTRY']}, {event_year}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Climate event(s)", str(row["EVENT_TYPES"]))
    c2.metric(shock_label, percent(row[shock_column]))
    c3.metric("Undernourishment", percent(row["UNDERNOURISHMENT_PERCENT"]))
    c4.metric("People affected", fmt_number(row["TOTAL_AFFECTED_REPORTED"]))
    render_production_timeline(
        row,
        production,
        chart_key=f"priority_timeline_{row['ISO3']}_{event_year}",
    )
    render_country_takeaway(row, shock_column)




def aid_priority_tab(events: pd.DataFrame, production: pd.DataFrame) -> None:
    shock_column = "WORST_DETRENDED_CHANGE_PERCENT_T_TO_T1"
    shock_label = "Detrended production shock"

    st.subheader("Humanitarian Aid Priority Review")
    st.markdown(
        '<div class="callout"><b>Decision supported:</b> Which historical country-disaster '
        "cases should humanitarian and agricultural recovery teams examine first?</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "This is a historical screening tool, not a live emergency alert or an automatic aid-allocation system."
    )

    threshold_col1, threshold_col2 = st.columns(2)
    with threshold_col1:
        minimum_drop = st.slider(
            "Severe production drop", 5, 30, 10, 1, format="%d%%",
            help="A case receives a production warning when its detrended drop reaches this level.",
        )
    with threshold_col2:
        minimum_undernourishment = st.slider(
            "High undernourishment", 5, 40, 20, 1, format="%d%%",
            help="A case receives a food-vulnerability warning at or above this prevalence.",
        )

    screened = events.copy()
    screened["SEVERE_SHOCK"] = screened[shock_column].le(-minimum_drop)
    screened["HIGH_UNDERNOURISHMENT"] = screened["UNDERNOURISHMENT_PERCENT"].ge(
        minimum_undernourishment
    )
    screened["PERSISTENT_NON_RECOVERY"] = screened["RECOVERY_STATUS"].eq(
        "NOT_RECOVERED_WITHIN_3_YEARS"
    )
    screened["WARNING_COUNT"] = screened[
        ["SEVERE_SHOCK", "HIGH_UNDERNOURISHMENT", "PERSISTENT_NON_RECOVERY"]
    ].sum(axis=1)
    screened["PRIORITY_TIER"] = np.select(
        [screened["WARNING_COUNT"].eq(3), screened["WARNING_COUNT"].eq(2)],
        ["Priority 1", "Priority 2"],
        default="Monitor",
    )

    def explain_case(row: pd.Series) -> str:
        reasons = []
        if row["SEVERE_SHOCK"]:
            reasons.append(
                f"production was {abs(float(row[shock_column])):.1f}% below its expected trend"
            )
        if row["HIGH_UNDERNOURISHMENT"]:
            reasons.append(
                f"undernourishment was {float(row['UNDERNOURISHMENT_PERCENT']):.1f}%"
            )
        if row["PERSISTENT_NON_RECOVERY"]:
            reasons.append("production did not recover within three years")
        return "; ".join(reasons) if reasons else "no selected priority thresholds were met"

    def assessment_area(row: pd.Series) -> str:
        if row["WARNING_COUNT"] == 3:
            action = "Food assistance and agricultural recovery assessment"
        elif row["SEVERE_SHOCK"] and row["PERSISTENT_NON_RECOVERY"]:
            action = "Longer-term production recovery and input-needs assessment"
        elif row["SEVERE_SHOCK"] and row["HIGH_UNDERNOURISHMENT"]:
            action = "Food assistance and seed/equipment needs assessment"
        elif row["HIGH_UNDERNOURISHMENT"] and row["PERSISTENT_NON_RECOVERY"]:
            action = "Food-security and agricultural recovery assessment"
        elif row["SEVERE_SHOCK"]:
            action = "Agricultural damage and production assessment"
        elif row["HIGH_UNDERNOURISHMENT"]:
            action = "Food-security monitoring"
        elif row["PERSISTENT_NON_RECOVERY"]:
            action = "Production recovery review"
        else:
            action = "Continue monitoring"

        affected_share = row.get("AFFECTED_POPULATION_PERCENT")
        if pd.notna(affected_share) and float(affected_share) >= 10:
            action = f"Rapid needs and logistics review; {action.lower()}"
        return action

    def data_caution(row: pd.Series) -> str:
        cautions = []
        if pd.isna(row[shock_column]):
            cautions.append("detrended shock unavailable")
        if pd.isna(row["UNDERNOURISHMENT_PERCENT"]):
            cautions.append("undernourishment unavailable")
        elif pd.notna(row["UNDERNOURISHMENT_IS_UPPER_BOUND"]) and bool(
            row["UNDERNOURISHMENT_IS_UPPER_BOUND"]
        ):
            cautions.append("undernourishment is an upper-bound estimate")
        if not (
            pd.notna(row["HAS_FULL_3_YEAR_FOLLOWUP"])
            and bool(row["HAS_FULL_3_YEAR_FOLLOWUP"])
        ):
            cautions.append("follow-up incomplete")
        if pd.isna(row["TOTAL_AFFECTED_REPORTED"]):
            cautions.append("people affected not reported")
        return "; ".join(cautions) if cautions else "no major completeness flags"

    screened["WHY_FLAGGED"] = screened.apply(explain_case, axis=1)
    screened["ASSESSMENT_AREA"] = screened.apply(assessment_area, axis=1)
    screened["DATA_CAUTION"] = screened.apply(data_caution, axis=1)
    tier_order = {"Priority 1": 1, "Priority 2": 2, "Monitor": 3}
    screened["TIER_ORDER"] = screened["PRIORITY_TIER"].map(tier_order)
    screened = screened.sort_values(
        ["TIER_ORDER", "WARNING_COUNT", shock_column, "UNDERNOURISHMENT_PERCENT",
         "AFFECTED_POPULATION_PERCENT"],
        ascending=[True, False, True, False, False],
        na_position="last",
    ).reset_index(drop=True)

    p1 = screened["PRIORITY_TIER"].eq("Priority 1")
    p2 = screened["PRIORITY_TIER"].eq("Priority 2")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Priority 1 cases", f"{p1.sum():,}")
    c2.metric("Priority 2 cases", f"{p2.sum():,}")
    c3.metric("Priority countries", f"{screened.loc[p1 | p2, 'ISO3'].nunique():,}")
    c4.metric(
        "Cases with affected-share data",
        f"{screened['AFFECTED_POPULATION_PERCENT'].notna().sum():,}",
    )

    filter_col, row_col = st.columns([2, 1])
    with filter_col:
        selected_tiers = st.multiselect(
            "Cases to include",
            ["Priority 1", "Priority 2", "Monitor"],
            default=["Priority 1", "Priority 2"],
        )
    with row_col:
        maximum_rows = st.selectbox("Maximum rows", [10, 25, 50, 100], index=1)

    queue = screened[screened["PRIORITY_TIER"].isin(selected_tiers)].copy()
    visible_queue = queue.head(maximum_rows)
    if queue.empty:
        st.info("No cases match the selected tiers and dashboard filters.")
        return

    chart_counts = (
        screened.groupby("PRIORITY_TIER", as_index=False)
        .size()
        .rename(columns={"size": "Cases"})
    )
    chart_counts["PRIORITY_TIER"] = pd.Categorical(
        chart_counts["PRIORITY_TIER"],
        ["Priority 1", "Priority 2", "Monitor"],
        ordered=True,
    )
    chart_counts = chart_counts.sort_values("PRIORITY_TIER")
    fig = px.bar(
        chart_counts,
        x="PRIORITY_TIER",
        y="Cases",
        text="Cases",
        color="PRIORITY_TIER",
        color_discrete_map={
            "Priority 1": COLORS["red"],
            "Priority 2": COLORS["gold"],
            "Monitor": COLORS["green"],
        },
        labels={"PRIORITY_TIER": ""},
        title="Historical cases by review tier",
    )
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    display = pd.DataFrame(
        {
            "Tier": visible_queue["PRIORITY_TIER"],
            "Country": visible_queue["COUNTRY"],
            "Year": visible_queue["EVENT_YEAR"].astype(int),
            "Climate event(s)": visible_queue["EVENT_TYPES"],
            "Detrended shock": visible_queue[shock_column],
            "Undernourishment": visible_queue["UNDERNOURISHMENT_PERCENT"],
            "No recovery in 3 years": visible_queue["PERSISTENT_NON_RECOVERY"],
            "People affected": visible_queue["TOTAL_AFFECTED_REPORTED"],
            "Affected share": visible_queue["AFFECTED_POPULATION_PERCENT"],
            "Deaths reported": visible_queue["TOTAL_DEATHS_REPORTED"],
            "Adjusted damage (000 USD)": visible_queue["TOTAL_DAMAGE_ADJUSTED_000_USD"],
            "Why flagged": visible_queue["WHY_FLAGGED"],
            "Suggested next assessment": visible_queue["ASSESSMENT_AREA"],
            "Data caution": visible_queue["DATA_CAUTION"],
        }
    )
    table_event = st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        height=min(610, 38 + len(display) * 35),
        column_config={
            "Detrended shock": st.column_config.NumberColumn(format="%.1f%%"),
            "Undernourishment": st.column_config.NumberColumn(format="%.1f%%"),
            "People affected": st.column_config.NumberColumn(format="localized"),
            "Affected share": st.column_config.NumberColumn(format="%.1f%%"),
            "Deaths reported": st.column_config.NumberColumn(format="localized"),
            "Adjusted damage (000 USD)": st.column_config.NumberColumn(format="localized"),
        },
        key="aid_priority_queue",
        on_select="rerun",
        selection_mode="single-row",
    )
    st.caption(
        "Select a row for its evidence and production timeline. Reported affected share may exceed "
        "100% when multiple disasters occurred in one year or the same people were counted more than once."
    )

    download_columns = [
        "PRIORITY_TIER", "COUNTRY", "ISO3", "EVENT_YEAR", "EVENT_TYPES",
        shock_column, "UNDERNOURISHMENT_PERCENT", "RECOVERY_STATUS",
        "TOTAL_AFFECTED_REPORTED", "AFFECTED_POPULATION_PERCENT",
        "TOTAL_DEATHS_REPORTED", "TOTAL_DAMAGE_ADJUSTED_000_USD",
        "WHY_FLAGGED", "ASSESSMENT_AREA", "DATA_CAUTION",
    ]
    st.download_button(
        "Download priority review list (CSV)",
        data=queue[download_columns].to_csv(index=False).encode("utf-8"),
        file_name="humanitarian_aid_priority_review.csv",
        mime="text/csv",
    )

    selected_rows = table_event.selection.rows
    if not selected_rows:
        st.info("Select a case in the table to see why it was flagged.")
        return

    row = visible_queue.iloc[selected_rows[0]]
    st.markdown(f"### {row['COUNTRY']}, {int(row['EVENT_YEAR'])}")
    st.markdown(
        f"""<div class="callout"><strong>{row['PRIORITY_TIER']}:</strong>
        {row['WHY_FLAGGED'].capitalize()}.<br><br>
        <strong>Suggested next assessment:</strong> {row['ASSESSMENT_AREA']}.<br>
        <strong>Data caution:</strong> {row['DATA_CAUTION']}.</div>""",
        unsafe_allow_html=True,
    )
    d1, d2, d3 = st.columns(3)
    d1.metric(shock_label, percent(row[shock_column]))
    d2.metric("Undernourishment", percent(row["UNDERNOURISHMENT_PERCENT"]))
    d3.metric("Recovery", friendly_status(row["RECOVERY_STATUS"]))
    d4, d5, d6, d7 = st.columns(4)
    d4.metric("People affected", fmt_number(row["TOTAL_AFFECTED_REPORTED"]))
    d5.metric("Affected share", percent(row["AFFECTED_POPULATION_PERCENT"]))
    d6.metric("Deaths reported", fmt_number(row["TOTAL_DEATHS_REPORTED"]))
    d7.metric("Adjusted damage (000 USD)", fmt_number(row["TOTAL_DAMAGE_ADJUSTED_000_USD"]))

    render_production_timeline(
        row,
        production,
        chart_key=f"aid_timeline_{row['ISO3']}_{int(row['EVENT_YEAR'])}",
    )
    st.warning(
        "This shortlist supports human review. It should be combined with current field assessments, "
        "local knowledge, logistics, and verified humanitarian needs before resources are allocated."
    )


def apply_filters(events: pd.DataFrame) -> pd.DataFrame:
    filtered = events.copy()
    selected_years = st.sidebar.slider(
        "Event years",
        int(events["EVENT_YEAR"].min()),
        int(events["EVENT_YEAR"].max()),
        (int(events["EVENT_YEAR"].min()), int(events["EVENT_YEAR"].max())),
    )
    selected_hazards = st.sidebar.multiselect(
        "Hazards",
        list(HAZARDS),
        default=list(HAZARDS),
    )
    vulnerability_options = list(events["POU_CATEGORY"].drop_duplicates())
    selected_vulnerability = st.sidebar.multiselect(
        "Undernourishment category",
        vulnerability_options,
        default=vulnerability_options,
    )
    full_follow_up = st.sidebar.checkbox("Full 3-year follow-up only", value=False)

    filtered = filtered[filtered["EVENT_YEAR"].between(*selected_years)]
    filtered = filtered[filtered["POU_CATEGORY"].isin(selected_vulnerability)]
    if selected_hazards:
        hazard_mask = np.zeros(len(filtered), dtype=bool)
        for hazard in selected_hazards:
            hazard_mask |= filtered[HAZARDS[hazard]].fillna(0).astype(float).gt(0).to_numpy()
        filtered = filtered[hazard_mask]
    else:
        filtered = filtered.iloc[0:0]
    if full_follow_up:
        filtered = filtered[filtered["HAS_FULL_3_YEAR_FOLLOWUP"].fillna(False)]
    return filtered


def overview_tab(
    events: pd.DataFrame,
    hazards: pd.DataFrame,
    production: pd.DataFrame,
    shock_column: str,
    shock_label: str,
) -> None:
    st.subheader("Global overview")
    st.markdown(
        '<div class="callout"><b>Question:</b> Where were climate events followed by '
        "food-production declines, and which food-vulnerable countries had the most difficulty recovering?</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Country disaster-years", f"{len(events):,}")
    c2.metric("Countries", f"{events['ISO3'].nunique():,}")
    decline_count = int(events[shock_column].lt(0).sum())
    c3.metric("Cases with a decline", f"{decline_count:,}")
    full = events[events["HAS_FULL_3_YEAR_FOLLOWUP"].fillna(False)]
    non_recovered = int(full["RECOVERY_STATUS"].eq("NOT_RECOVERED_WITHIN_3_YEARS").sum())
    c4.metric("No recovery in 3 years", f"{non_recovered:,}")

    country_summary = (
        events.groupby(["ISO3", "COUNTRY"], as_index=False)
        .agg(
            MEDIAN_SHOCK=(shock_column, "median"),
            EVENT_YEARS=("EVENT_YEAR", "nunique"),
            POU=("UNDERNOURISHMENT_PERCENT", "median"),
        )
    )
    fig = px.choropleth(
        country_summary,
        locations="ISO3",
        color="MEDIAN_SHOCK",
        hover_name="COUNTRY",
        custom_data=["ISO3"],
        hover_data={"EVENT_YEARS": True, "POU": ":.1f", "ISO3": False},
        color_continuous_scale=[COLORS["red"], "#F5E6E8", COLORS["green"]],
        color_continuous_midpoint=0,
        labels={
            "MEDIAN_SHOCK": f"Median {shock_label.lower()} (%)",
            "EVENT_YEARS": "Disaster-years",
            "POU": "Undernourishment (%)",
        },
        title=f"Median {shock_label.lower()} following focus events",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=45, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        clickmode="event+select",
    )
    st.caption("Click a country. Its snapshot will open immediately below the map.")
    map_event = st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"overview_country_map_{shock_column}",
        on_select="rerun",
        selection_mode="points",
    )

    selected_points = map_event.selection.points
    if selected_points:
        point = selected_points[0]
        selected_iso3 = point.get("location")
        if not selected_iso3 and point.get("customdata"):
            selected_iso3 = point["customdata"][0]
        if selected_iso3:
            st.session_state["overview_selected_iso3"] = selected_iso3

    selected_iso3 = st.session_state.get("overview_selected_iso3")
    valid_iso3 = set(events["ISO3"].dropna().astype(str))
    if selected_iso3 in valid_iso3:
        render_country_snapshot(events, selected_iso3, shock_column, shock_label)
        selected_row = select_country_focus_event(events, selected_iso3, shock_column)
        if selected_row is not None:
            render_production_timeline(
                selected_row,
                production,
                chart_key=f"map_timeline_{selected_iso3}_{int(selected_row['EVENT_YEAR'])}",
            )
            render_country_takeaway(selected_row, shock_column)
    else:
        st.info("Select a country on the map to open its country snapshot here.")

    st.divider()
    st.markdown("### Supporting context")
    left, right = st.columns(2)
    with left:
        hazard_counts = (
            hazards[hazards["ISO3"].isin(events["ISO3"].unique())]
            .groupby("DISASTER_TYPE", as_index=False)
            .size()
            .rename(columns={"size": "EVENT_COUNT"})
            .sort_values("EVENT_COUNT")
        )
        fig = px.bar(
            hazard_counts,
            x="EVENT_COUNT",
            y="DISASTER_TYPE",
            orientation="h",
            text="EVENT_COUNT",
            title="Recorded disasters by type",
            color_discrete_sequence=[COLORS["rose"]],
            labels={"EVENT_COUNT": "Events", "DISASTER_TYPE": ""},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=30, t=45, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_showgrid=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        recovery = (
            events[events["HAS_FULL_3_YEAR_FOLLOWUP"].fillna(False)]
            .groupby("RECOVERY_STATUS", as_index=False)
            .size()
            .rename(columns={"size": "CASES"})
        )
        status_order = ["MAINTAINED", "RECOVERED", "NOT_RECOVERED_WITHIN_3_YEARS"]
        recovery["RECOVERY_STATUS"] = pd.Categorical(
            recovery["RECOVERY_STATUS"], categories=status_order, ordered=True
        )
        recovery = recovery.sort_values("RECOVERY_STATUS")
        fig = px.bar(
            recovery,
            x="RECOVERY_STATUS",
            y="CASES",
            text="CASES",
            title="Recovery outcomes with complete follow-up",
            color="RECOVERY_STATUS",
            color_discrete_map={
                "MAINTAINED": COLORS["green"],
                "RECOVERED": COLORS["gold"],
                "NOT_RECOVERED_WITHIN_3_YEARS": COLORS["red"],
            },
            labels={"RECOVERY_STATUS": "", "CASES": "Country disaster-years"},
        )
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    scatter = events.dropna(subset=["UNDERNOURISHMENT_PERCENT", shock_column])
    fig = px.scatter(
        scatter,
        x="UNDERNOURISHMENT_PERCENT",
        y=shock_column,
        color="RECOVERY_STATUS",
        hover_name="COUNTRY",
        hover_data=["EVENT_YEAR", "EVENT_TYPES"],
        opacity=0.65,
        title="Undernourishment and production change",
        labels={
            "UNDERNOURISHMENT_PERCENT": "Undernourishment (%)",
            shock_column: f"{shock_label} (%)",
            "RECOVERY_STATUS": "Outcome",
        },
        color_discrete_map={
            "MAINTAINED": COLORS["green"],
            "RECOVERED": COLORS["gold"],
            "NOT_RECOVERED_WITHIN_3_YEARS": COLORS["red"],
            "FOLLOW_UP_INCOMPLETE": COLORS["gray"],
        },
    )
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["gray"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


def country_tab(
    events: pd.DataFrame,
    production: pd.DataFrame,
    shock_column: str,
    shock_label: str,
) -> None:
    st.subheader("Country deep dive")
    countries = sorted(events["COUNTRY"].dropna().unique())
    country = st.selectbox("Choose a country", countries)
    country_events = events[events["COUNTRY"].eq(country)].sort_values("EVENT_YEAR")
    event_year = st.selectbox("Choose an event year", country_events["EVENT_YEAR"].astype(int).tolist())
    row = country_events[country_events["EVENT_YEAR"].eq(event_year)].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Event types", row["EVENT_TYPES"])
    c2.metric(shock_label, percent(row[shock_column]))
    c3.metric("Recovery status", str(row["RECOVERY_STATUS"]).replace("_", " ").title())
    c4.metric("Undernourishment", percent(row["UNDERNOURISHMENT_PERCENT"]))

    render_production_timeline(
        row,
        production,
        chart_key=f"deep_dive_timeline_{row['ISO3']}_{int(event_year)}",
    )
    render_country_takeaway(row, shock_column)

    details = pd.DataFrame(
        {
            "Measure": [
                "Country disaster-year",
                "Pre-event baseline",
                "Event-year index",
                "Worst index in T to T+1",
                "Raw shock vs. 3-year baseline",
                "Detrended shock vs. expected trend",
                "Pre-event trend observations",
                "Recovery time",
                "People affected, reported",
                "Deaths, reported",
                "Adjusted damage (000 USD)",
            ],
            "Value": [
                f"{country}, {int(event_year)}",
                f"{float(row['BASELINE_INDEX']):.1f}",
                f"{float(row['EVENT_YEAR_INDEX']):.1f}",
                f"{float(row['WORST_INDEX_T_TO_T1']):.1f}" if pd.notna(row["WORST_INDEX_T_TO_T1"]) else "No data",
                percent(row["WORST_CHANGE_PERCENT_T_TO_T1"]),
                percent(row["WORST_DETRENDED_CHANGE_PERCENT_T_TO_T1"]),
                f"{int(row['TREND_YEAR_COUNT'])} years" if pd.notna(row["TREND_YEAR_COUNT"]) else "No data",
                f"{int(row['RECOVERY_YEARS'])} years" if pd.notna(row["RECOVERY_YEARS"]) else "Not observed",
                fmt_number(row["TOTAL_AFFECTED_REPORTED"]),
                fmt_number(row["TOTAL_DEATHS_REPORTED"]),
                fmt_number(row["TOTAL_DAMAGE_ADJUSTED_000_USD"]),
            ],
        }
    )
    st.dataframe(details, hide_index=True, use_container_width=True)


def assumptions_tab(events: pd.DataFrame) -> None:
    st.subheader("Assumptions Lab")
    st.caption("These settings are project-defined. Adjust them to see whether the main conclusion changes.")
    threshold = st.slider("Recovery threshold", 90, 100, 95, 1)
    window = st.slider("Recovery window (years)", 1, 3, 3, 1)
    include_bounds = st.checkbox("Include upper-bound undernourishment estimates", value=True)

    data = events.copy()
    cutoff = data["BASELINE_INDEX"] * threshold / 100
    available_years = [data[f"YEAR_{year}_INDEX"] for year in range(1, window + 1)]
    maintained = data["EVENT_YEAR_INDEX"].ge(cutoff)
    recovered = pd.Series(False, index=data.index)
    recovery_year = pd.Series(np.nan, index=data.index)
    for year, values in enumerate(available_years, start=1):
        newly_recovered = (~maintained) & (~recovered) & values.ge(cutoff)
        recovery_year.loc[newly_recovered] = year
        recovered |= newly_recovered

    full_window = data["EVENT_YEAR"].le(data["MAX_PRODUCTION_YEAR"] - window)
    data["LAB_STATUS"] = np.select(
        [maintained, recovered, full_window],
        ["Maintained", "Recovered", f"Not recovered within {window} years"],
        default="Follow-up incomplete",
    )
    data["LAB_RECOVERY_YEARS"] = recovery_year
    if not include_bounds:
        data = data[~data["UNDERNOURISHMENT_IS_UPPER_BOUND"].fillna(False)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Maintained", f"{data['LAB_STATUS'].eq('Maintained').sum():,}")
    c2.metric("Recovered", f"{data['LAB_STATUS'].eq('Recovered').sum():,}")
    c3.metric(
        f"No recovery in {window} years",
        f"{data['LAB_STATUS'].eq(f'Not recovered within {window} years').sum():,}",
    )

    counts = data.groupby("LAB_STATUS", as_index=False).size().rename(columns={"size": "CASES"})
    fig = px.bar(
        counts,
        x="LAB_STATUS",
        y="CASES",
        text="CASES",
        color="LAB_STATUS",
        color_discrete_sequence=[COLORS["green"], COLORS["gold"], COLORS["red"], COLORS["gray"]],
        labels={"LAB_STATUS": "", "CASES": "Country disaster-years"},
        title=f"Outcomes using a {threshold}% threshold and {window}-year window",
    )
    fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Current limitation: the maintained classification begins with event-year production. "
        "The separate T to T+1 metric should be reviewed for delayed impacts before final conclusions."
    )


def guided_explorer_tab(
    events: pd.DataFrame,
    production: pd.DataFrame,
    shock_column: str,
    shock_label: str,
) -> None:
    st.subheader("Guided Data Explorer")
    st.write(
        "Choose a question and adjust the controls. The answer is calculated directly from "
        "the filtered Snowflake data, so every result can be traced back to the underlying rows."
    )
    st.caption(
        "Results follow the event years, hazards, undernourishment categories, follow-up setting, "
        "and shock measure selected in the dashboard sidebar."
    )

    questions = [
        "Which countries had the largest production shocks?",
        "Which countries meet the priority rules?",
        "How do outcomes compare by disaster type?",
        "What happened in one country's event history?",
    ]
    question = st.selectbox("What would you like to explore?", questions)

    if question == questions[0]:
        top_n = st.slider("Number of countries to show", 5, 20, 10)
        ranked = (
            events.dropna(subset=[shock_column])
            .sort_values(shock_column)
            .drop_duplicates("ISO3")
            .head(top_n)
            .copy()
        )
        if ranked.empty:
            st.warning("No measured production shocks match the current dashboard filters.")
            return

        worst = ranked.iloc[0]
        st.markdown(
            f"""<div class="callout"><strong>Answer:</strong> Among the current results,
            <strong>{worst['COUNTRY']}</strong> had the deepest measured shock in
            <strong>{int(worst['EVENT_YEAR'])}</strong>: {percent(worst[shock_column])}
            ({worst['EVENT_TYPES']}). More negative values indicate production farther below
            the selected benchmark.</div>""",
            unsafe_allow_html=True,
        )

        chart_data = ranked.sort_values(shock_column, ascending=False)
        fig = px.bar(
            chart_data,
            x=shock_column,
            y="COUNTRY",
            orientation="h",
            color=shock_column,
            color_continuous_scale=["#B94A48", "#E8C6CF", "#4F7A65"],
            text=shock_column,
            hover_data=["EVENT_YEAR", "EVENT_TYPES", "RECOVERY_STATUS"],
            labels={shock_column: f"{shock_label} (%)", "COUNTRY": ""},
            title=f"Deepest measured shock for each of the top {len(ranked)} countries",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=45, t=50, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        table = ranked[
            ["COUNTRY", "EVENT_YEAR", "EVENT_TYPES", shock_column, "RECOVERY_STATUS"]
        ].copy()
        table.columns = ["Country", "Event year", "Climate event(s)", shock_label, "Recovery outcome"]
        table["Recovery outcome"] = table["Recovery outcome"].map(friendly_status)
        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            column_config={shock_label: st.column_config.NumberColumn(format="%.1f%%")},
        )

    elif question == questions[1]:
        priority_shock_column = "WORST_DETRENDED_CHANGE_PERCENT_T_TO_T1"
        priority_shock_label = "Detrended shock, T to T+1"
        control_a, control_b = st.columns(2)
        with control_a:
            minimum_undernourishment = st.slider(
                "Minimum undernourishment", 0, 50, 20, 1, format="%d%%"
            )
        with control_b:
            minimum_drop = st.slider(
                "Minimum detrended production drop", 0, 40, 10, 1, format="%d%%"
            )

        priority = events[
            events["HAS_FULL_3_YEAR_FOLLOWUP"].fillna(False)
            & events["RECOVERY_STATUS"].eq("NOT_RECOVERED_WITHIN_3_YEARS")
            & events["UNDERNOURISHMENT_PERCENT"].ge(minimum_undernourishment)
            & events[priority_shock_column].le(-minimum_drop)
        ].copy()

        country_count = priority["ISO3"].nunique()
        st.markdown(
            f"""<div class="callout"><strong>Answer:</strong>
            <strong>{country_count:,} countries</strong> have {len(priority):,} country-event-year
            cases meeting all three selected rules: at least {minimum_undernourishment}%
            undernourishment, a production drop of at least {minimum_drop}%, and no recovery
            within three years.</div>""",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Countries", f"{country_count:,}")
        c2.metric("Country-event-year cases", f"{len(priority):,}")
        c3.metric(
            "Upper-bound hunger estimates",
            f"{priority['UNDERNOURISHMENT_IS_UPPER_BOUND'].fillna(False).sum():,}",
        )

        if priority.empty:
            st.info("No cases meet these settings. Lower one of the thresholds to broaden the search.")
            return

        fig = px.scatter(
            priority,
            x="UNDERNOURISHMENT_PERCENT",
            y=priority_shock_column,
            color="EVENT_TYPES",
            hover_name="COUNTRY",
            hover_data=["EVENT_YEAR", "RECOVERY_STATUS"],
            labels={
                "UNDERNOURISHMENT_PERCENT": "Undernourishment (%)",
                priority_shock_column: f"{priority_shock_label} (%)",
                "EVENT_TYPES": "Climate event(s)",
            },
            title="Priority cases: vulnerability and production shock",
        )
        fig.add_vline(x=minimum_undernourishment, line_dash="dot", line_color=COLORS["gray"])
        fig.add_hline(y=-minimum_drop, line_dash="dot", line_color=COLORS["gray"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        table = priority.sort_values(priority_shock_column)[
            [
                "COUNTRY",
                "EVENT_YEAR",
                "EVENT_TYPES",
                priority_shock_column,
                "UNDERNOURISHMENT_PERCENT",
                "UNDERNOURISHMENT_IS_UPPER_BOUND",
            ]
        ].copy()
        table.columns = [
            "Country",
            "Event year",
            "Climate event(s)",
            priority_shock_label,
            "Undernourishment",
            "Upper-bound estimate",
        ]
        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            column_config={
                priority_shock_label: st.column_config.NumberColumn(format="%.1f%%"),
                "Undernourishment": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

    elif question == questions[2]:
        rows = []
        for hazard, event_count_column in HAZARDS.items():
            hazard_cases = events[
                events[event_count_column].gt(0) & events[shock_column].notna()
            ]
            complete = hazard_cases[
                hazard_cases["HAS_FULL_3_YEAR_FOLLOWUP"].fillna(False)
            ]
            not_recovered_rate = (
                complete["RECOVERY_STATUS"].eq("NOT_RECOVERED_WITHIN_3_YEARS").mean() * 100
                if not complete.empty
                else np.nan
            )
            rows.append(
                {
                    "Disaster type": hazard,
                    "Country-event-year cases": len(hazard_cases),
                    "Countries": hazard_cases["ISO3"].nunique(),
                    "Median production shock": hazard_cases[shock_column].median(),
                    "Not recovered within 3 years": not_recovered_rate,
                }
            )

        comparison = pd.DataFrame(rows).sort_values("Median production shock")
        deepest = comparison.iloc[0]
        st.markdown(
            f"""<div class="callout"><strong>Answer:</strong>
            <strong>{deepest['Disaster type']}</strong> has the deepest median measured production
            shock under the current filters at {percent(deepest['Median production shock'])}.
            This is a comparison of associated country-event-year outcomes, not proof that the
            disaster type caused the change.</div>""",
            unsafe_allow_html=True,
        )

        fig = px.bar(
            comparison.sort_values("Median production shock", ascending=False),
            x="Median production shock",
            y="Disaster type",
            orientation="h",
            text="Median production shock",
            color="Median production shock",
            color_continuous_scale=["#B94A48", "#E8C6CF", "#4F7A65"],
            labels={"Median production shock": f"Median {shock_label.lower()} (%)"},
            title="Typical production shock by disaster type",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            comparison,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Median production shock": st.column_config.NumberColumn(format="%.1f%%"),
                "Not recovered within 3 years": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        st.caption(
            "A country-year containing more than one disaster type appears in each relevant row "
            "of this comparison."
        )

    else:
        countries = sorted(events["COUNTRY"].dropna().unique())
        selected_country = st.selectbox("Choose a country", countries)
        country_events = events[events["COUNTRY"].eq(selected_country)].sort_values("EVENT_YEAR")
        measured = country_events.dropna(subset=[shock_column])

        if measured.empty:
            st.info("This country has no measured shocks under the current filters.")
            return

        worst = measured.loc[measured[shock_column].idxmin()]
        st.markdown(
            f"""<div class="callout"><strong>Answer:</strong> The deepest measured shock for
            <strong>{selected_country}</strong> occurred in <strong>{int(worst['EVENT_YEAR'])}</strong>
            following {worst['EVENT_TYPES']}: {percent(worst[shock_column])}. Its recovery outcome
            was <strong>{friendly_status(worst['RECOVERY_STATUS'])}</strong>.</div>""",
            unsafe_allow_html=True,
        )

        fig = px.line(
            measured,
            x="EVENT_YEAR",
            y=shock_column,
            markers=True,
            color="RECOVERY_STATUS",
            hover_data=["EVENT_TYPES", "UNDERNOURISHMENT_PERCENT"],
            labels={
                "EVENT_YEAR": "Event year",
                shock_column: f"{shock_label} (%)",
                "RECOVERY_STATUS": "Recovery outcome",
            },
            title=f"{selected_country}: measured shocks across event years",
            color_discrete_map={
                "MAINTAINED": COLORS["green"],
                "RECOVERED": COLORS["gold"],
                "NOT_RECOVERED_WITHIN_3_YEARS": COLORS["red"],
                "FOLLOW_UP_INCOMPLETE": COLORS["gray"],
            },
        )
        fig.add_hline(y=0, line_dash="dot", line_color=COLORS["gray"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        event_year = st.selectbox(
            "Open the production timeline for an event year",
            measured["EVENT_YEAR"].astype(int).tolist(),
            index=measured.index.get_loc(worst.name),
        )
        selected_row = measured[measured["EVENT_YEAR"].eq(event_year)].iloc[0]
        render_production_timeline(
            selected_row,
            production,
            chart_key=f"guided_timeline_{selected_row['ISO3']}_{int(event_year)}",
        )
        render_country_takeaway(selected_row, shock_column)





def methodology_tab() -> None:
    st.subheader("Methodology and limitations")
    st.markdown(
        """
        **Analysis unit:** one country experiencing one or more focus events during one year.

        **Focus events:** drought, wildfire, extreme temperature, flood, and storm.

        **Production baseline:** the average food-production index during the three years before an event,
        with at least two years required.

        **Detrended shock:** a linear trend is fitted to the five pre-event years, with at least four
        observations required. The metric is the worse percentage deviation from trend-predicted production
        in the event year or following year. Negative values indicate production below its expected path.

        **Recovery:** a project-defined rule. The working definition is a return to at least 95% of baseline
        within three years. It is not an agency standard.

        **Food vulnerability:** prevalence of undernourishment is the primary measure. Survey-based food
        insecurity is retained as a separate secondary measure because the methodologies are different.

        **Aid priority tiers:** Priority 1 means all three project warnings are present: a detrended
        production drop at or beyond the selected threshold, undernourishment at or above the selected
        threshold, and no recovery within three years. Priority 2 means two warnings are present. These
        tiers identify cases for further review; they are not an agency standard or an aid-allocation decision.

        **Affected population share:** reported people affected is divided by FAOSTAT total population for
        the event year. The percentage can exceed 100% when multiple disasters are combined or people are
        counted more than once, so it is supporting context rather than a ranking rule.

        **Interpretation:** the results show associations and recovery patterns. They do not prove that a
        recorded disaster caused the observed production change.
        """
    )


st.title("After the Shock")
st.markdown("### Humanitarian Aid Prioritization Tool")
st.write(
    "Use historical production shocks, undernourishment, recovery outcomes, and reported disaster "
    "impact to identify country-disaster cases for closer humanitarian and agricultural review."
)

stored_password = secret_value("password", os.getenv("SNOWFLAKE_PASSWORD", ""))
cloud_connection = bool(stored_password)

with st.sidebar:
    if cloud_connection:
        account = secret_value("account", "biofiay-oi65812")
        user = secret_value("user", "")
        password = stored_password
        warehouse = secret_value("warehouse", "FAOSTAT_WH")
        database = secret_value("database", "FAOSTAT_DB")
        role = secret_value("role", "DATATHON_APP_ROLE")
        connect_clicked = False
        st.success("Secure Snowflake connection configured")
        st.caption("Credentials are stored server-side and are not visible to app visitors.")
    else:
        st.header("Snowflake connection")
        account = st.text_input("Account", value="biofiay-oi65812")
        user = st.text_input("Username", value="jrapson")
        password = st.text_input("Password or access token", value="", type="password")
        warehouse = st.text_input("Warehouse", value="FAOSTAT_WH")
        database = st.text_input("Database", value="FAOSTAT_DB")
        role = st.text_input("Role", value="ACCOUNTADMIN")
        connect_clicked = st.button("Connect", type="primary", use_container_width=True)
        st.caption("The credential is used for this session and is not written to the project files.")

if cloud_connection or connect_clicked:
    st.session_state["connect_requested"] = True

if not st.session_state.get("connect_requested"):
    st.info("Enter your Snowflake password in the sidebar and select **Connect** to load the dashboard.")
    st.stop()

if not password:
    st.error("Enter a Snowflake password to connect.")
    st.stop()

try:
    with st.spinner("Loading analysis data from Snowflake..."):
        connection = connect_snowflake(account, user, password, warehouse, database, role)
        events_df = query_dataframe(connection, EVENT_SQL)
        production_df = query_dataframe(connection, PRODUCTION_SQL)
        population_df = query_dataframe(connection, POPULATION_SQL)
        hazards_df = query_dataframe(connection, HAZARD_SQL)
except Exception as exc:
    st.error("Snowflake connection failed. Check the account, username, password, warehouse, and role.")
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

numeric_columns = [
    "EVENT_YEAR", "TOTAL_FOCUS_EVENTS", "DROUGHT_EVENTS", "WILDFIRE_EVENTS",
    "EXTREME_TEMPERATURE_EVENTS", "FLOOD_EVENTS", "STORM_EVENTS",
    "BASELINE_YEAR_COUNT", "BASELINE_INDEX", "EVENT_YEAR_INDEX", "YEAR_1_INDEX",
    "YEAR_2_INDEX", "YEAR_3_INDEX", "WORST_INDEX_T_TO_T1", "MAX_PRODUCTION_YEAR",
    "TREND_YEAR_COUNT", "PRE_EVENT_TREND_SLOPE", "PRE_EVENT_TREND_INTERCEPT",
    "EXPECTED_EVENT_YEAR_INDEX", "EXPECTED_YEAR_1_INDEX", "EVENT_YEAR_CHANGE_PERCENT",
    "WORST_CHANGE_PERCENT_T_TO_T1", "EVENT_YEAR_DETRENDED_CHANGE_PERCENT",
    "YEAR_1_DETRENDED_CHANGE_PERCENT", "WORST_DETRENDED_CHANGE_PERCENT_T_TO_T1",
    "RECOVERY_YEARS",
    "UNDERNOURISHMENT_PERCENT", "FOOD_INSECURITY_PERCENT", "TOTAL_AFFECTED_REPORTED",
    "TOTAL_DEATHS_REPORTED", "TOTAL_DAMAGE_ADJUSTED_000_USD",
]
for column in numeric_columns:
    events_df[column] = pd.to_numeric(events_df[column], errors="coerce")
production_df["YEAR"] = pd.to_numeric(production_df["YEAR"], errors="coerce")
production_df["FOOD_PRODUCTION_INDEX"] = pd.to_numeric(
    production_df["FOOD_PRODUCTION_INDEX"], errors="coerce"
)
hazards_df["START_YEAR"] = pd.to_numeric(hazards_df["START_YEAR"], errors="coerce")
population_df["YEAR"] = pd.to_numeric(population_df["YEAR"], errors="coerce")
population_df["POPULATION"] = pd.to_numeric(population_df["POPULATION"], errors="coerce")
events_df["AREA_CODE_M49_JOIN"] = (
    events_df["AREA_CODE_M49"].astype("string").str.extract(r"(\d+)", expand=False).str.zfill(3)
)
population_df["AREA_CODE_M49_JOIN"] = (
    population_df["AREA_CODE_M49"].astype("string").str.extract(r"(\d+)", expand=False).str.zfill(3)
)
population_year = (
    population_df.groupby(["AREA_CODE_M49_JOIN", "YEAR"], as_index=False)["POPULATION"].max()
)
events_df = events_df.merge(
    population_year,
    how="left",
    left_on=["AREA_CODE_M49_JOIN", "EVENT_YEAR"],
    right_on=["AREA_CODE_M49_JOIN", "YEAR"],
)
events_df["AFFECTED_POPULATION_PERCENT"] = (
    events_df["TOTAL_AFFECTED_REPORTED"] / events_df["POPULATION"] * 100
).replace([np.inf, -np.inf], np.nan)
events_df["POU_CATEGORY"] = events_df["UNDERNOURISHMENT_PERCENT"].apply(vulnerability_band)

st.sidebar.divider()
st.sidebar.header("Dashboard filters")
filtered_events = apply_filters(events_df)
shock_choice = st.sidebar.radio(
    "Shock measure",
    ["Detrended (recommended)", "Raw 3-year baseline"],
    help="Detrended compares actual production with the country's expected pre-event trend.",
)
if shock_choice == "Detrended (recommended)":
    shock_column = "WORST_DETRENDED_CHANGE_PERCENT_T_TO_T1"
    shock_label = "Detrended shock, T to T+1"
else:
    shock_column = "WORST_CHANGE_PERCENT_T_TO_T1"
    shock_label = "Raw shock, T to T+1"
filtered_hazards = hazards_df[
    hazards_df["START_YEAR"].between(
        filtered_events["EVENT_YEAR"].min() if not filtered_events.empty else 0,
        filtered_events["EVENT_YEAR"].max() if not filtered_events.empty else 0,
    )
]

if filtered_events.empty:
    st.warning("No cases match the current filters.")
    st.stop()

aid, overview, country, guided, assumptions, methods = st.tabs(
    ["Aid Priority", "Global overview", "Country deep dive", "Guided explorer", "Assumptions Lab", "Methodology"]
)
with aid:
    aid_priority_tab(filtered_events, production_df)
with overview:
    overview_tab(filtered_events, filtered_hazards, production_df, shock_column, shock_label)
with country:
    country_tab(filtered_events, production_df, shock_column, shock_label)
with guided:
    guided_explorer_tab(filtered_events, production_df, shock_column, shock_label)
with assumptions:
    assumptions_tab(filtered_events)
with methods:
    methodology_tab()
