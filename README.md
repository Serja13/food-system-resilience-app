# After the Shock: Humanitarian Aid Prioritization Tool

Food System Resilience Explorer for the 2026 Women in Data Datathon.

The Streamlit app uses historical production shocks, undernourishment,
recovery outcomes, and reported disaster impact to identify country-disaster
cases for closer humanitarian and agricultural review. Its transparent
priority tiers support human review and do not automatically allocate aid.

## Data

The app reads prepared analytical views in Snowflake. No raw FAOSTAT or EM-DAT
files are stored in this repository.

The **Guided explorer** calculates its answers directly from the prepared
Snowflake data and does not require Cortex Agents or a paid AI service.

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

When cloud secrets are configured, the login form is hidden and the app
connects automatically. Never commit a real password, private key, or access
token to this repository.
