# Himalayan Garnet Analysis Database

A Streamlit research dashboard for Himalayan garnet compositional data, provenance comparison, and community impact tracking.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

### Local development

1. Install the requirements

   ```bash
   pip install -r requirements.txt
   ```

2. Run the app

   ```bash
   streamlit run streamlit_app.py
   ```

### Deploying as a website

This repository is ready for deployment on Streamlit Community Cloud.

1. Push the repository to GitHub.
2. Sign in to [Streamlit Cloud](https://streamlit.io/cloud).
3. Create a new app and point it to this repository and `streamlit_app.py`.

Once deployed, the app will be live at a public Streamlit URL.

## Data submissions

The **Submit Data** tab validates contributed CSV files and writes accepted ones to a separate, private GitHub staging repository (`submissions/<id>/data.csv` + `metadata.json`) and opens an issue there, which emails repository watchers. Submissions never modify the public dataset.

Setup:

1. Create a private repository for staging (for example `gdp-dashboard-submissions`) and watch it so issue notifications reach you.
2. Create a fine-grained token limited to that repository with **Contents: read/write** and **Issues: read/write**.
3. In Streamlit Cloud, open **Settings > Secrets** and add:

```toml
STAGING_GITHUB_TOKEN = "github_pat_..."
STAGING_GITHUB_REPO = "your-user/gdp-dashboard-submissions"
```

Until these secrets exist, the tab still validates files but the submit button stays disabled.
