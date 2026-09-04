-- Run as ACCOUNTADMIN after 17_add_detrended_shock.sql.
-- Creates the governed business layer used by the Streamlit "Ask the Data" tab.

USE ROLE ACCOUNTADMIN;
USE WAREHOUSE FAOSTAT_WH;
USE DATABASE FAOSTAT_DB;
USE SCHEMA ANALYTICS;

CREATE OR REPLACE SEMANTIC VIEW ANALYTICS.SV_FOOD_SYSTEM_RESILIENCE
  TABLES (
    resilience AS ANALYTICS.VW_EVENT_PRODUCTION_RESILIENCE
      PRIMARY KEY (ISO3, EVENT_YEAR)
      COMMENT = 'One row per country and year in which one or more focus climate disasters occurred'
  )
  FACTS (
    resilience.total_focus_events AS TOTAL_FOCUS_EVENTS
      COMMENT = 'Number of focus disaster records for the country-year',
    resilience.drought_events AS DROUGHT_EVENTS
      COMMENT = 'Number of drought records',
    resilience.wildfire_events AS WILDFIRE_EVENTS
      COMMENT = 'Number of wildfire records',
    resilience.extreme_temperature_events AS EXTREME_TEMPERATURE_EVENTS
      COMMENT = 'Number of extreme-temperature records',
    resilience.flood_events AS FLOOD_EVENTS
      COMMENT = 'Number of flood records',
    resilience.storm_events AS STORM_EVENTS
      COMMENT = 'Number of storm records',
    resilience.detrended_shock_percent AS WORST_DETRENDED_CHANGE_PERCENT_T_TO_T1
      WITH SYNONYMS = ('production shock', 'detrended production drop', 'change versus expected trend')
      COMMENT = 'Worse percentage deviation of event-year or following-year production from the five-year pre-event trend; more negative is worse',
    resilience.raw_shock_percent AS WORST_CHANGE_PERCENT_T_TO_T1
      WITH SYNONYMS = ('raw production shock', 'change from baseline')
      COMMENT = 'Worse percentage change in event-year or following-year production from the three-year pre-event average',
    resilience.undernourishment_percent AS UNDERNOURISHMENT_PERCENT
      WITH SYNONYMS = ('hunger', 'food vulnerability')
      COMMENT = 'Prevalence of undernourishment percentage used as the primary food-vulnerability measure',
    resilience.food_insecurity_percent AS FOOD_INSECURITY_PERCENT
      COMMENT = 'Survey-based moderate or severe food insecurity percentage; do not combine numerically with undernourishment',
    resilience.recovery_years AS RECOVERY_YEARS
      WITH SYNONYMS = ('time to recover', 'recovery time')
      COMMENT = 'Years required to return to at least 95 percent of the three-year pre-event production baseline',
    resilience.people_affected AS TOTAL_AFFECTED_REPORTED
      COMMENT = 'Total people affected across focus disasters with a reported value',
    resilience.deaths_reported AS TOTAL_DEATHS_REPORTED
      COMMENT = 'Total deaths across focus disasters with a reported value',
    resilience.adjusted_damage_thousand_usd AS TOTAL_DAMAGE_ADJUSTED_000_USD
      COMMENT = 'Inflation-adjusted reported disaster damage in thousands of US dollars'
  )
  DIMENSIONS (
    resilience.country AS COUNTRY
      WITH SYNONYMS = ('nation', 'country name')
      COMMENT = 'Country in which the disaster was recorded',
    resilience.iso3 AS ISO3
      COMMENT = 'Three-letter country code',
    resilience.event_year AS EVENT_YEAR
      WITH SYNONYMS = ('disaster year', 'climate event year')
      COMMENT = 'Year in which one or more focus disasters began',
    resilience.event_types AS EVENT_TYPES
      WITH SYNONYMS = ('hazards', 'disasters', 'climate events')
      COMMENT = 'Comma-separated focus disaster types recorded for this country and year',
    resilience.recovery_status AS RECOVERY_STATUS
      WITH SYNONYMS = ('recovery outcome', 'recovered')
      COMMENT = 'Project classification: maintained, recovered, not recovered within three years, or incomplete follow-up',
    resilience.resilience_category AS RESILIENCE_CATEGORY
      COMMENT = 'Project resilience category from the analytical view',
    resilience.has_full_followup AS HAS_FULL_3_YEAR_FOLLOWUP
      COMMENT = 'True when all three post-event production years are available',
    resilience.food_insecurity_upper_bound AS FOOD_INSECURITY_IS_UPPER_BOUND
      COMMENT = 'True when the food-insecurity percentage is reported as an upper-bound value',
    resilience.undernourishment_upper_bound AS UNDERNOURISHMENT_IS_UPPER_BOUND
      COMMENT = 'True when the undernourishment percentage is reported as an upper-bound value',
    resilience.priority_case AS IFF(
      WORST_DETRENDED_CHANGE_PERCENT_T_TO_T1 <= -10
      AND UNDERNOURISHMENT_PERCENT >= 20
      AND RECOVERY_STATUS = 'NOT_RECOVERED_WITHIN_3_YEARS',
      TRUE,
      FALSE
    )
      WITH SYNONYMS = ('high concern', 'highest risk', 'priority country')
      COMMENT = 'Project rule: detrended shock at or below -10 percent, undernourishment at or above 20 percent, and no recovery within three years'
  )
  METRICS (
    resilience.country_event_case_count AS COUNT(*)
      COMMENT = 'Number of country-event-year cases, not a count of distinct countries',
    resilience.country_count AS COUNT(DISTINCT ISO3)
      COMMENT = 'Number of distinct countries',
    resilience.average_detrended_shock_percent AS AVG(resilience.detrended_shock_percent)
      COMMENT = 'Average production change versus expected pre-event trend',
    resilience.median_detrended_shock_percent AS MEDIAN(resilience.detrended_shock_percent)
      COMMENT = 'Median production change versus expected pre-event trend',
    resilience.average_undernourishment_percent AS AVG(resilience.undernourishment_percent)
      COMMENT = 'Average prevalence of undernourishment',
    resilience.no_recovery_case_count AS SUM(IFF(RECOVERY_STATUS = 'NOT_RECOVERED_WITHIN_3_YEARS', 1, 0))
      COMMENT = 'Number of country-event-year cases that did not recover within three years',
    resilience.priority_case_count AS SUM(IFF(
      WORST_DETRENDED_CHANGE_PERCENT_T_TO_T1 <= -10
      AND UNDERNOURISHMENT_PERCENT >= 20
      AND RECOVERY_STATUS = 'NOT_RECOVERED_WITHIN_3_YEARS',
      1,
      0
    ))
      COMMENT = 'Number of cases meeting all three transparent priority rules'
  )
  COMMENT = 'Business layer for analyzing food-production shocks, recovery, disasters, and undernourishment by country and event year'
  AI_SQL_GENERATION 'Use detrended_shock_percent when a question says production shock unless raw or baseline change is explicitly requested. More negative shock values mean worse outcomes. A row is one country-event-year, so never describe country_event_case_count as a number of countries. Use country_count for distinct countries. Treat undernourishment_percent as the primary food-vulnerability measure. Never average or combine food_insecurity_percent and undernourishment_percent because they use different methodologies. The priority rule is detrended shock at or below -10 percent, undernourishment at or above 20 percent, and recovery status NOT_RECOVERED_WITHIN_3_YEARS. Recovery is a project rule: return to at least 95 percent of the three-year pre-event baseline within three years. Round percentages to one decimal place.'
  AI_QUESTION_CATEGORIZATION 'Answer questions about the food-system resilience data only. If biggest disaster is ambiguous, ask whether the user means the largest production shock, most people affected, most deaths, or most events. Explain that results are associations and do not prove a disaster caused a production change.';

CREATE OR REPLACE AGENT ANALYTICS.FOOD_SYSTEM_RESILIENCE_AGENT
  COMMENT = 'Answers questions about climate disasters, food-production shocks, recovery, and undernourishment'
  PROFILE = '{"display_name": "Food System Resilience Guide"}'
  FROM SPECIFICATION
  $$
  models:
    orchestration: auto

  instructions:
    response: "Answer in clear, concise language for a general audience. State whether a number refers to countries or country-event-year cases. Include important data limitations."
    orchestration: "Use the Resilience_Analyst tool for every question about countries, disasters, production, recovery, resilience, hunger, food vulnerability, or the priority rule. Do not answer those questions from general knowledge."
    sample_questions:
      - question: "Which countries meet all three priority rules?"
      - question: "Which drought cases had the largest production drops?"
      - question: "How do recovery outcomes differ by disaster type?"

  tools:
    - tool_spec:
        type: "cortex_analyst_text_to_sql"
        name: "Resilience_Analyst"
        description: "Queries the governed food-system resilience semantic view."

  tool_resources:
    Resilience_Analyst:
      semantic_view: "FAOSTAT_DB.ANALYTICS.SV_FOOD_SYSTEM_RESILIENCE"
      execution_environment:
        type: "warehouse"
        warehouse: "FAOSTAT_WH"
  $$;

-- Agent calls use the token owner's default role and warehouse.
ALTER USER JRAPSON SET DEFAULT_ROLE = DATATHON_APP_ROLE DEFAULT_WAREHOUSE = FAOSTAT_WH;

GRANT DATABASE ROLE SNOWFLAKE.CORTEX_AGENT_USER TO ROLE DATATHON_APP_ROLE;
GRANT USAGE ON WAREHOUSE FAOSTAT_WH TO ROLE DATATHON_APP_ROLE;
GRANT USAGE ON DATABASE FAOSTAT_DB TO ROLE DATATHON_APP_ROLE;
GRANT USAGE ON SCHEMA FAOSTAT_DB.ANALYTICS TO ROLE DATATHON_APP_ROLE;
GRANT REFERENCES, SELECT ON SEMANTIC VIEW ANALYTICS.SV_FOOD_SYSTEM_RESILIENCE TO ROLE DATATHON_APP_ROLE;
GRANT USAGE ON AGENT ANALYTICS.FOOD_SYSTEM_RESILIENCE_AGENT TO ROLE DATATHON_APP_ROLE;

SELECT *
FROM SEMANTIC_VIEW(
  ANALYTICS.SV_FOOD_SYSTEM_RESILIENCE
  DIMENSIONS resilience.recovery_status
  METRICS resilience.country_event_case_count
)
ORDER BY country_event_case_count DESC;

DESCRIBE AGENT ANALYTICS.FOOD_SYSTEM_RESILIENCE_AGENT;

