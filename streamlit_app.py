import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
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


def snowflake_base_url(account: str) -> str:
    host = account.strip().replace("https://", "").replace("http://", "").rstrip("/")
    if not host.endswith(".snowflakecomputing.com"):
        host = f"{host}.snowflakecomputing.com"
    return f"https://{host}"


def collect_agent_artifacts(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]], list[str], list[str]]:
    response_text = "\n\n".join(
        item.get("text", "")
        for item in payload.get("content", [])
        if item.get("type") == "text" and item.get("text")
    ).strip()
    tables: list[dict[str, Any]] = []
    sql_statements: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            result_set = value.get("result_set")
            if isinstance(result_set, dict) and result_set.get("data") is not None:
                metadata = result_set.get("resultSetMetaData", {})
                columns = [column.get("name", "Column") for column in metadata.get("rowType", [])]
                tables.append({"columns": columns, "rows": result_set.get("data", [])})
            sql = value.get("sql")
            if isinstance(sql, str) and sql.strip() and sql not in sql_statements:
                sql_statements.append(sql)
            for key, child in value.items():
                if key != "result_set":
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload.get("content", []))
    warning_messages = [
        warning.get("message", "")
        for warning in payload.get("warnings", [])
        if warning.get("message")
    ]
    if not response_text:
        response_text = "The agent completed the request but did not return a written summary."
    return response_text, tables, sql_statements, warning_messages


def run_resilience_agent(
    account: str,
    token: str,
    conversation: list[dict[str, Any]],
) -> dict[str, Any]:
    endpoint = (
        f"{snowflake_base_url(account)}/api/v2/databases/FAOSTAT_DB/schemas/ANALYTICS/"
        "agents/FOOD_SYSTEM_RESILIENCE_AGENT:run"
    )
    api_messages = [
        {
            "role": message["role"],
            "content": [{"type": "text", "text": message["text"]}],
        }
        for message in conversation[-8:]
    ]
    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {token}",
            "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={
            "messages": api_messages,
            "background": False,
            "stream": False,
            "tool_choice": {"type": "auto"},
        },
        timeout=180,
    )
    if not response.ok:
        try:
            error_detail = response.json().get("message", response.text)
        except ValueError:
            error_detail = response.text
        raise RuntimeError(f"Snowflake Agent request failed ({response.status_code}): {error_detail[:800]}")
    return response.json()


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
    render_priority_explorer(events, production, shock_column, shock_label)

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


def ask_data_tab(account: str, token: str) -> None:
    st.subheader("Ask the Data")
    st.write(
        "Ask questions in everyday language. The Snowflake agent uses the project's governed "
        "definitions for production shock, recovery, food vulnerability, and priority cases."
    )
    st.caption(
        "The assistant can summarize patterns in this dataset. Its answers show associations, not proof "
        "that a recorded disaster caused a production change."
    )

    if not token:
        st.info("Add the Snowflake programmatic access token to the app secrets to enable this tab.")
        return

    if "resilience_chat" not in st.session_state:
        st.session_state["resilience_chat"] = []

    header_left, header_right = st.columns([4, 1])
    with header_right:
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["resilience_chat"] = []
            st.rerun()

    for message in st.session_state["resilience_chat"]:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])
            for table in message.get("tables", []):
                if table["columns"]:
                    st.dataframe(
                        pd.DataFrame(table["rows"], columns=table["columns"]),
                        hide_index=True,
                        use_container_width=True,
                    )
            if message.get("sql"):
                with st.expander("SQL used by Snowflake"):
                    for statement in message["sql"]:
                        st.code(statement, language="sql")
            for warning in message.get("warnings", []):
                st.warning(warning)

    suggested_questions = [
        "Which countries meet all three priority rules?",
        "Which drought cases had the largest detrended production drops?",
        "How do recovery outcomes differ by disaster type?",
    ]
    selected_suggestion = None
    if not st.session_state["resilience_chat"]:
        st.markdown("**Try one of these:**")
        suggestion_columns = st.columns(3)
        for index, question in enumerate(suggested_questions):
            with suggestion_columns[index]:
                if st.button(question, key=f"suggested_question_{index}", use_container_width=True):
                    selected_suggestion = question

    typed_question = st.chat_input("Ask about countries, disasters, production shocks, or recovery")
    question = typed_question or selected_suggestion
    if not question:
        return

    st.session_state["resilience_chat"].append({"role": "user", "text": question})
    with st.spinner("Snowflake is analyzing the resilience data..."):
        try:
            payload = run_resilience_agent(
                account,
                token,
                st.session_state["resilience_chat"],
            )
            text, tables, sql_statements, warnings = collect_agent_artifacts(payload)
            st.session_state["resilience_chat"].append(
                {
                    "role": "assistant",
                    "text": text,
                    "tables": tables,
                    "sql": sql_statements,
                    "warnings": warnings,
                }
            )
        except Exception as exc:
            st.session_state["resilience_chat"].append(
                {
                    "role": "assistant",
                    "text": (
                        "I could not reach the Snowflake resilience agent. Confirm that "
                        "`18_create_resilience_semantic_agent.sql` completed successfully and that "
                        "the app secret contains the current programmatic access token."
                    ),
                    "warnings": [str(exc)],
                }
            )
    st.rerun()


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

        **Interpretation:** the results show associations and recovery patterns. They do not prove that a
        recorded disaster caused the observed production change.
        """
    )


st.title("After the Shock")
st.markdown("### Food System Resilience Explorer")
st.write(
    "Explore how national food production changed during and after climate-related events, "
    "and whether countries already facing undernourishment had greater difficulty recovering."
)

stored_password = secret_value("password", os.getenv("SNOWFLAKE_PASSWORD", ""))
agent_token = secret_value("token", stored_password)
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
        password = st.text_input("Password", value="", type="password")
        warehouse = st.text_input("Warehouse", value="FAOSTAT_WH")
        database = st.text_input("Database", value="FAOSTAT_DB")
        role = st.text_input("Role", value="ACCOUNTADMIN")
        connect_clicked = st.button("Connect", type="primary", use_container_width=True)
        st.caption("The password is used for this session and is not written to the project files.")

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

overview, country, ask_data, assumptions, methods = st.tabs(
    ["Global overview", "Country deep dive", "Ask the Data", "Assumptions Lab", "Methodology"]
)
with overview:
    overview_tab(filtered_events, filtered_hazards, production_df, shock_column, shock_label)
with country:
    country_tab(filtered_events, production_df, shock_column, shock_label)
with ask_data:
    ask_data_tab(account, agent_token)
with assumptions:
    assumptions_tab(filtered_events)
with methods:
    methodology_tab()
