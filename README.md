# After the Shock

Food System Resilience Explorer for the 2026 Women in Data Datathon.

The Streamlit app explores how national food production changed during and
after droughts, wildfires, extreme temperatures, floods, and storms. It also
highlights whether countries already facing undernourishment had greater
difficulty recovering.

## Data

The app reads prepared analytical views in Snowflake. No raw FAOSTAT or EM-DAT
files are stored in this repository.

The **Ask the Data** tab uses a Snowflake semantic view and Cortex Agent. Run
`sql/18_create_resilience_semantic_agent.sql` in Snowsight as `ACCOUNTADMIN`
after the detrended-shock view exists.

## Run locally

1. Install the packages in `requirements.txt`.
2. Run `streamlit run streamlit_app.py`.
3. Enter your Snowflake credentials in the sidebar.

## Deploy to Streamlit Community Cloud

1. Select this GitHub repository and `streamlit_app.py` at
   https://share.streamlit.io.
2. Open **Advanced settings**.
3. Paste the values from `.streamlit/secrets.toml.example` into the Secrets
   field, replacing the placeholders with the Snowflake username and a
   role-restricted programmatic access token.
4. Deploy the app.

The same programmatic access token is used for the Snowflake connection and
the Cortex Agent request. You may store it as `password`; an optional `token`
entry is also supported if you want to keep those values separate.

When cloud secrets are configured, the login form is hidden and the app
connects automatically. Never commit a real password, private key, or access
token to this repository.
