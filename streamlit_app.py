import streamlit as st
import submission
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.metrics import pairwise_distances

UNIT_COLORS = {
    'GHC': '#D55E00',
    'LHS': '#0072B2',
    'THS': '#009E73',
    'MCT': '#CC79A7',
    'HHL': '#E69F00',
    'AC': '#56B4E9',
    'GB': '#F0E442',
    'GD': '#332288',
    'IC': '#88CCEE',
    'ITSZ': '#44AA99',
    'NHG': '#AA4499',
    'Other': '#999999',
}

# -----------------------------------------------------------------------------
# Helper functions for garnet comparison

def calculate_aitchison_distances(database_df, input_composition):
    """Calculate Aitchison distances between input composition and database.

    This function uses a fast path when a precomputed CLR of the database
    is available via `get_db_clr()`.
    """
    endmember_cols = ['Almandine_%', 'Pyrope_%', 'Grossular_%', 'Spessartine_%']

    # Ensure endmembers exist
    db = database_df.copy()
    db_compositions = db[endmember_cols].fillna(0.0).replace(0, 0.001)

    # Convert input to series and handle zeros
    input_array = np.array([
        input_composition['Almandine_%'],
        input_composition['Pyrope_%'],
        input_composition['Grossular_%'],
        input_composition['Spessartine_%']
    ], dtype=float)
    input_array = np.where(input_array == 0, 0.001, input_array)

    # Attempt to use cached DB CLR for speed
    try:
        db_clr, db_index = get_db_clr()
        # compute input clr
        input_clr = clr_transform_array(input_array.reshape(1, -1))[0]
        # Euclidean distance in CLR space
        distances = np.linalg.norm(db_clr - input_clr, axis=1)
        results = db.loc[db_index].copy()
    except Exception:
        # Fallback: compute CLR on the fly
        def clr_series(arr):
            arr = np.where(np.array(arr, dtype=float) == 0, 0.001, arr.astype(float))
            return np.log(arr / np.exp(np.mean(np.log(arr))))

        db_clr_df = db_compositions.apply(lambda row: clr_series(row.values), axis=1)
        input_clr = clr_series(input_array)
        distances = np.sqrt(((np.vstack(db_clr_df.values) - input_clr) ** 2).sum(axis=1))
        results = db.copy()

    results['aitchison_distance'] = distances
    results['similarity_score'] = 1.0 / (1.0 + distances)
    return results


def clr_transform_array(arr_2d):
    """Compute CLR transform for a 2D numpy array of compositions.

    Replaces zeros with small value and returns transformed array.
    """
    arr = np.array(arr_2d, dtype=float)
    arr[arr == 0] = 0.001
    log_arr = np.log(arr)
    gm = np.exp(np.mean(log_arr, axis=1, keepdims=True))
    clr = np.log(arr / gm)
    return clr


@st.cache_data
def get_db_clr():
    """Return cached CLR-transformed database and its row positions.

    This avoids recomputing the CLR for every comparison.
    """
    endmember_cols = ['Almandine_%', 'Pyrope_%', 'Grossular_%', 'Spessartine_%']
    df = garnet_df.copy()
    comps = df[endmember_cols].fillna(0.0).replace(0, 0.001).to_numpy(dtype=float)
    db_clr = clr_transform_array(comps)
    return db_clr, np.arange(len(df), dtype=int)


def batch_aitchison_distances_for_upload(db_df, upload_array, matches_per_sample=5, chunk_size=256):
    """Compute Aitchison distances for multiple uploaded samples and return matches.

    upload_array: numpy array shape (n_uploads, 4)
    Returns a dataframe concatenating top matches per uploaded sample.
    """
    endmember_cols = ['Almandine_%', 'Pyrope_%', 'Grossular_%', 'Spessartine_%']
    db_clr, db_index = get_db_clr()

    # Ensure upload_array shape
    X_clr = clr_transform_array(upload_array)

    all_rows = []
    # Compute pairwise distances in chunks for memory efficiency
    for start in range(0, X_clr.shape[0], chunk_size):
        end = min(start + chunk_size, X_clr.shape[0])
        chunk = X_clr[start:end]
        dists = pairwise_distances(chunk, db_clr, metric='euclidean', n_jobs=-1)
        for i in range(dists.shape[0]):
            row_idx = start + i
            dist_row = dists[i]
            top_idx = np.argsort(dist_row)[:matches_per_sample]
            for rank, tidx in enumerate(top_idx, start=1):
                db_row = db_df.iloc[tidx].copy()
                db_row['aitchison_distance'] = dist_row[tidx]
                db_row['similarity_score'] = 1.0 / (1.0 + dist_row[tidx])
                db_row['match_rank'] = rank
                all_rows.append((row_idx, db_row))

    if not all_rows:
        return pd.DataFrame()

    # Build final dataframe
    records = []
    for row_idx, db_row in all_rows:
        rec = db_row.to_dict()
        rec['uploaded_row'] = int(row_idx)
        records.append(rec)

    results_df = pd.DataFrame.from_records(records)
    return results_df


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points in kilometers."""
    earth_radius_km = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return earth_radius_km * c


def add_geographic_distances(df, user_lat, user_lon):
    df = df.copy()
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        return df

    latitudes = pd.to_numeric(df['Latitude'], errors='coerce')
    longitudes = pd.to_numeric(df['Longitude'], errors='coerce')
    df['distance_km'] = haversine_distance_km(user_lat, user_lon, latitudes, longitudes)
    return df


def filter_database_by_radius(df, user_lat, user_lon, radius_km):
    df = add_geographic_distances(df, user_lat, user_lon)
    return df[df['distance_km'] <= radius_km].copy()


def assign_similarity_and_dominance(results_df):
    """Assign similarity classes and dominance categories based on Aitchison distance."""
    results = results_df.copy()
    results['similarity_class'] = 'Low-Similarity'
    results.loc[results['aitchison_distance'] <= 0.40, 'similarity_class'] = 'High-Similarity'
    results.loc[(results['aitchison_distance'] > 0.40) & (results['aitchison_distance'] <= 0.79), 'similarity_class'] = 'Moderate-Similarity'

    # Default dominance category for non-high-similarity matches
    results['dominance_category'] = 'Not High-Similarity'

    high_results = results[results['similarity_class'] == 'High-Similarity']
    if len(high_results) == 1:
        results.loc[high_results.index, 'dominance_category'] = 'Unique'
    elif len(high_results) > 1:
        prefix = high_results['Garnet_ID'].astype(str).str.split('-').str[0]
        prefix_counts = prefix.value_counts()
        total_high = len(high_results)
        prefix_pct = prefix_counts / total_high

        category_map = {}
        for prefix_value, pct in prefix_pct.items():
            if pct >= 0.70:
                category_map[prefix_value] = 'Strongly dominant'
            elif pct >= 0.55:
                category_map[prefix_value] = 'Moderately dominant'
            elif pct >= 0.45:
                category_map[prefix_value] = 'Weakly dominant'
            else:
                category_map[prefix_value] = 'Mixed'

        for prefix_value, category in category_map.items():
            prefix_rows = results['Garnet_ID'].astype(str).str.split('-').str[0] == prefix_value
            high_rows = results['similarity_class'] == 'High-Similarity'
            results.loc[prefix_rows & high_rows, 'dominance_category'] = category

    results['similarity_category'] = results['similarity_class']
    return results


def find_tolerance_matches(database_df, input_composition, tolerance, max_matches=None):
    """Find matches within tolerance for all end-members"""
    endmember_cols = ['Almandine_%', 'Pyrope_%', 'Grossular_%', 'Spessartine_%']

    matches = []
    for idx, row in database_df.iterrows():
        within_tolerance = True
        for col in endmember_cols:
            diff = abs(row[col] - input_composition[col])
            if diff > tolerance:
                within_tolerance = False
                break

        if within_tolerance:
            match_row = row.copy()
            match_row['endmember_differences'] = {
                col: abs(row[col] - input_composition[col]) for col in endmember_cols
            }
            matches.append(match_row)

        if max_matches is not None and len(matches) >= max_matches:
            break

    if matches:
        results = pd.DataFrame(matches)
        # Calculate average difference as a simple similarity score
        results['avg_difference'] = results['endmember_differences'].apply(
            lambda x: np.mean(list(x.values()))
        )
        results['similarity_score'] = 1.0 / (1.0 + results['avg_difference'])
        results = results.sort_values('avg_difference')
    else:
        results = pd.DataFrame()

    return results

def display_comparison_results(results_df, input_composition, method):
    """Display comparison results"""
    endmember_cols = ['Almandine_%', 'Pyrope_%', 'Grossular_%', 'Spessartine_%']

    # Build display columns with available fields.
    display_cols = []
    for col in ['sample_id', 'Garnet_ID', 'Author_Sample_ID', 'Country', 'Location', 'Unit', 'Lithology_Category', 'Latitude', 'Longitude', 'distance_km']:
        if col in results_df.columns:
            display_cols.append(col)
    display_cols += endmember_cols
    if method == "Aitchison distance":
        for col in ['aitchison_distance', 'similarity_category', 'dominance_category', 'DOI']:
            if col in results_df.columns:
                display_cols.append(col)
        st.dataframe(results_df[display_cols].head(5), width='stretch')
    else:
        for col in ['avg_difference', 'DOI']:
            if col in results_df.columns:
                display_cols.append(col)
        st.dataframe(results_df[display_cols].head(5), width='stretch')

    # Summary charts
    col1, col2 = st.columns(2)

    with col1:
        # Unit distribution
        unit_counts = results_df['Unit'].fillna('Other').value_counts().head(10)
        if len(unit_counts) > 0:
            unit_summary = pd.DataFrame({'Unit': unit_counts.index, 'Count': unit_counts.values})
            fig = px.bar(
                unit_summary,
                x='Unit',
                y='Count',
                color='Unit',
                title="Matched Units",
                labels={'Unit': 'Unit', 'Count': 'Count'},
                color_discrete_map=UNIT_COLORS
            )
            st.plotly_chart(fig, width='stretch')

    with col2:
        # Lithology distribution
        lith_counts = results_df['Lithology_Category'].value_counts().head(10)
        if len(lith_counts) > 0:
            fig = px.bar(
                x=lith_counts.index,
                y=lith_counts.values,
                title="Matched Lithologies",
                labels={'x': 'Lithology', 'y': 'Count'},
                color_discrete_map=UNIT_COLORS
            )
            st.plotly_chart(fig, width='stretch')

    # Map if coordinates available
    if 'Latitude' in results_df.columns and 'Longitude' in results_df.columns:
        map_data = results_df.dropna(subset=['Latitude', 'Longitude']).head(50).copy()
        if len(map_data) > 0:
            map_data['Unit'] = map_data['Unit'].fillna('Other')
            st.subheader("Locations of Top Matching Garnets (first 50 shown)")
            fig = px.scatter_map(
                map_data,
                lat='Latitude',
                lon='Longitude',
                hover_name='Location',
                hover_data=['Country', 'Unit', 'Lithology_Category'],
                color='Unit',
                title="Locations of Top Matching Garnets",
                zoom=5,
                height=400,
                color_discrete_map=UNIT_COLORS
            )
            fig.update_layout(map={"style": "open-street-map"})
            st.plotly_chart(fig, width='stretch')

    # Download button
    st.download_button(
        label="📥 Download All Match Results (CSV)",
        data=results_df.to_csv(index=False),
        file_name="garnet_matches.csv",
        mime="text/csv",
        type="primary"
    )


def display_automatic_interpretation(results_df, total_matches=None):
    if total_matches is None:
        total_matches = len(results_df)

    st.subheader("Automatic Interpretation")
    st.markdown(f"**Your search returned {total_matches} garnets.**")

    if 'Unit' in results_df.columns and results_df['Unit'].notna().any():
        unit_counts = results_df['Unit'].fillna('Other').value_counts()
        top_unit = unit_counts.index[0]
        unit_pct = unit_counts.iloc[0] / unit_counts.sum() * 100
        st.markdown(f"**Most common matched unit:** {top_unit} — {unit_pct:.0f}%")

    if 'Lithology_Category' in results_df.columns and results_df['Lithology_Category'].notna().any():
        lith_counts = results_df['Lithology_Category'].fillna('Unknown').value_counts()
        top_lith = lith_counts.index[0]
        lith_pct = lith_counts.iloc[0] / lith_counts.sum() * 100
        st.markdown(f"**Most common lithology:** {top_lith} — {lith_pct:.0f}%")

    if 'aitchison_distance' in results_df.columns and len(results_df) > 0:
        closest = results_df.nsmallest(1, 'aitchison_distance').iloc[0]
        closest_id = closest.get('Garnet_ID', closest.get('sample_id', 'Unknown'))
        closest_dist = closest['aitchison_distance']
        st.markdown(f"**Closest match:** {closest_id}, Aitchison distance = {closest_dist:.2f}")

    if 'Country' in results_df.columns and results_df['Country'].notna().any():
        country_counts = results_df['Country'].fillna('Unknown').value_counts()
        top_countries = country_counts.head(2).index.tolist()
        if len(top_countries) == 1:
            geo_desc = f"Most matches occur in {top_countries[0]}."
        else:
            geo_desc = f"Most matches occur in {top_countries[0]} and {top_countries[1]}."
        st.markdown(f"**Geographic concentration:** {geo_desc}")

    if 'Latitude' in results_df.columns and 'Longitude' in results_df.columns:
        map_data = results_df.dropna(subset=['Latitude', 'Longitude']).copy()
        if len(map_data) > 0:
            map_data['Unit'] = map_data['Unit'].fillna('Other')
            st.subheader("Geographic Distribution of All Matches")
            fig = px.scatter_map(
                map_data,
                lat='Latitude',
                lon='Longitude',
                hover_name='Garnet_ID' if 'Garnet_ID' in map_data.columns else 'sample_id',
                hover_data=['Country', 'Location', 'Unit', 'Lithology_Category', 'distance_km'],
                color='Unit' if 'Unit' in map_data.columns else None,
                title='Geographic Distribution of All Matching Garnets',
                zoom=4,
                height=500,
                color_discrete_map=UNIT_COLORS
            )
            fig.update_layout(map={"style": "open-street-map"})
            st.plotly_chart(fig, width='stretch')

    st.info("Interpretation note: These are compositional similarities, not unique source assignments.")

def display_upload_results(results_df, upload_df):
    """Display upload comparison results"""
    # Summary statistics
    st.subheader("Summary Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Matches", len(results_df))

    with col2:
        unique_uploaded = results_df['uploaded_sample_id'].nunique()
        st.metric("Uploaded Samples", unique_uploaded)

    with col3:
        avg_distance = results_df['aitchison_distance'].mean()
        min_distance = results_df['aitchison_distance'].min()
        st.metric("Avg Aitchison Distance", f"{avg_distance:.3f}")
        st.metric("Best Aitchison Distance", f"{min_distance:.3f}")
        if 'match_distance_km' in results_df.columns and results_df['match_distance_km'].notna().any():
            closest_physical = results_df['match_distance_km'].min()
            st.metric("Closest Match Distance (km)", f"{closest_physical:.1f}")

    # Summary of distances per uploaded sample
    st.subheader("Closest Distance per Uploaded Sample")
    sample_distance = (
        results_df.groupby('uploaded_sample_id', dropna=False)['aitchison_distance']
        .min()
        .reset_index()
        .rename(columns={'aitchison_distance': 'Best_Aitchison_Distance'})
        .sort_values('Best_Aitchison_Distance')
    )
    if 'match_distance_km' in results_df.columns:
        sample_distance['Best_Physical_Distance_km'] = (
            results_df.groupby('uploaded_sample_id', dropna=False)['match_distance_km']
            .min()
            .reset_index(drop=True)
        )
    st.dataframe(sample_distance.head(20), width='stretch')

    # Results table (first 100 rows)
    st.subheader("Match Results")
    display_cols = [
        'uploaded_sample_id', 'uploaded_analysis_id', 'match_rank',
        'Garnet_ID', 'Author_Sample_ID', 'Country', 'Location', 'Unit',
        'Lithology_Category', 'DOI', 'aitchison_distance'
    ]
    if 'match_distance_km' in results_df.columns:
        display_cols.append('match_distance_km')

    st.dataframe(results_df[display_cols].head(100), width='stretch')

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        # Dominant matched units
        unit_counts = results_df['Unit'].fillna('Other').value_counts().head(10)
        unit_summary = pd.DataFrame({'Unit': unit_counts.index, 'Count': unit_counts.values})
        fig = px.bar(
            unit_summary,
            x='Unit',
            y='Count',
            color='Unit',
            title="Dominant Matched Units",
            labels={'Unit': 'Unit', 'Count': 'Match Count'},
            color_discrete_map=UNIT_COLORS
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        # Dominant matched lithologies
        lith_counts = results_df['Lithology_Category'].value_counts().head(10)
        fig = px.bar(
            x=lith_counts.index,
            y=lith_counts.values,
            title="Dominant Matched Lithologies",
            labels={'x': 'Lithology', 'y': 'Match Count'},
            color_discrete_map=UNIT_COLORS
        )
        st.plotly_chart(fig, width='stretch')

    # Aitchison distance histogram
    st.subheader("Aitchison Distance Distribution")
    fig = px.histogram(
        results_df,
        x='aitchison_distance',
        color=results_df['Unit'].fillna('Other'),
        title="Distribution of Aitchison Distances",
        labels={'aitchison_distance': 'Aitchison Distance'},
        nbins=50,
        barmode='overlay',
        opacity=0.65,
        color_discrete_map=UNIT_COLORS
    )
    st.plotly_chart(fig, width='stretch')

    # Map of matched locations
    if 'Latitude' in results_df.columns and 'Longitude' in results_df.columns:
        map_data = results_df.dropna(subset=['Latitude', 'Longitude']).head(200).copy()
        if len(map_data) > 0:
            map_data['Unit'] = map_data['Unit'].fillna('Other')
            st.subheader("Map of Matched Locations")
            fig = px.scatter_map(
                map_data,
                lat='Latitude',
                lon='Longitude',
                hover_name='Location',
                hover_data=['Country', 'Unit', 'uploaded_sample_id'],
                color='Unit',
                title="Locations of Matching Garnets",
                zoom=5,
                height=500,
                color_discrete_map=UNIT_COLORS
            )
            fig.update_layout(map={"style": "open-street-map"})
            st.plotly_chart(fig, width='stretch')

    # Download results
    st.download_button(
        label="📥 Download All Match Results (CSV)",
        data=results_df.to_csv(index=False),
        file_name="batch_garnet_matches.csv",
        mime="text/csv",
        type="primary"
    )

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Himalayan Garnet Analysis Database',
    page_icon=str(Path(__file__).parent / 'assets' / 'garnet-mark.svg'),
    layout='wide'
)

st.logo(str(Path(__file__).parent / 'assets' / 'garnet-mark.svg'), size='small')

st.markdown(
    """
    <style>
        .stApp {
            background: #FAF8F4;
        }
        .stButton>button, .stDownloadButton>button {
            background-color: #8C1C2B;
            color: white;
            border-radius: 0.5rem;
            border: none;
            padding: 0.65rem 1rem;
            font-weight: 600;
        }
        .stButton>button:hover, .stDownloadButton>button:hover {
            background-color: #6F1521;
            border-color: #6F1521;
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div>div {
            border-radius: 0.4rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            border-bottom: none;
            flex-wrap: wrap;
        }
        .stTabs [data-baseweb="tab"] {
            height: auto;
            padding: 0.55rem 1.1rem;
            border: 2px solid #8C1C2B;
            border-radius: 0.6rem;
            background: transparent;
        }
        .stTabs [data-baseweb="tab"] p {
            font-size: 1.05rem;
            font-weight: 600;
            color: #8C1C2B;
        }
        .stTabs [aria-selected="true"] {
            background: #8C1C2B;
        }
        .stTabs [aria-selected="true"] p {
            color: #FFFFFF;
        }
        .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Himalayan Garnet Analysis Database")
st.caption(
    "A curated compilation of Himalayan garnet major-element analyses, "
    "with tools for filtering, comparison and provenance matching."
)
st.markdown(
    '<hr style="border:0;border-top:2px solid #8C1C2B;margin:0 0 1rem">',
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
@st.cache_data
def get_garnet_data():
    """Grab garnet analysis data from a CSV file.

    This uses caching to avoid having to read the file every time.
    """

    DATA_FILENAME = Path(__file__).parent/'data/garnet_data.csv'
    df = pd.read_csv(DATA_FILENAME)

    # Clean up column names (remove trailing commas and spaces)
    df.columns = df.columns.str.strip()

    # Extract Unit from Garnet_ID prefix
    df['Unit'] = df['Garnet_ID'].str.split('-').str[0]

    return df


USER_REGISTRY_FILE = Path(__file__).parent / 'data' / 'user_registry.csv'
USER_REGISTRY_COLUMNS = [
    'timestamp',
    'name',
    'institution',
    'country',
    'contact',
    'use_case',
    'project',
    'citation',
    'permission_to_list'
]


def get_user_registry_entries():
    if not USER_REGISTRY_FILE.exists():
        return pd.DataFrame(columns=USER_REGISTRY_COLUMNS)

    try:
        df = pd.read_csv(USER_REGISTRY_FILE)
    except Exception:
        df = pd.DataFrame(columns=USER_REGISTRY_COLUMNS)

    # Ensure the expected columns are present
    for col in USER_REGISTRY_COLUMNS:
        if col not in df.columns:
            df[col] = ''
    return df[USER_REGISTRY_COLUMNS]


def append_user_registry_entry(entry):
    df = get_user_registry_entries()
    new_row = pd.DataFrame([entry], columns=USER_REGISTRY_COLUMNS)
    updated = pd.concat([df, new_row], ignore_index=True)
    updated.to_csv(USER_REGISTRY_FILE, index=False)


def validate_uploaded_csv_file(uploaded_file):
    if not uploaded_file.name.lower().endswith('.csv'):
        return False, 'Only files with a .csv extension are accepted.'

    uploaded_file.seek(0)
    sample = uploaded_file.read(8192)
    uploaded_file.seek(0)

    if b'\x00' in sample:
        return False, 'Binary data detected. Please upload a text-based CSV file.'

    try:
        sample.decode('utf-8')
    except UnicodeDecodeError:
        try:
            sample.decode('latin1')
        except UnicodeDecodeError:
            return False, 'Unable to read the file as text. Use a valid CSV format.'

    lowered = sample.lower()
    suspicious_tokens = [b'<?php', b'<script', b'eval(', b'system(', b'os.system', b'subprocess', b'#!/', b'powershell', b'cmd.exe']
    if any(token in lowered for token in suspicious_tokens):
        return False, 'Potential executable or script content detected. Please upload a clean CSV file.'

    return True, None


def create_dataset_summary(df, top_n_categories=50):
    summary_rows = []

    # Dataset overview metrics
    summary_rows.append({
        'Section': 'Dataset Overview',
        'Metric': 'Total garnet analyses',
        'Category': '',
        'Value': len(df),
        'Count': '',
        'Percent': ''
    })

    if 'Garnet_ID' in df.columns:
        summary_rows.append({
            'Section': 'Dataset Overview',
            'Metric': 'Unique Garnet_ID values',
            'Category': '',
            'Value': df['Garnet_ID'].nunique(dropna=True),
            'Count': '',
            'Percent': ''
        })

    if 'Author_Sample_ID' in df.columns:
        summary_rows.append({
            'Section': 'Dataset Overview',
            'Metric': 'Unique Author_Sample_ID values',
            'Category': '',
            'Value': df['Author_Sample_ID'].nunique(dropna=True),
            'Count': '',
            'Percent': ''
        })

    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        coords_count = df.dropna(subset=['Latitude', 'Longitude']).shape[0]
        summary_rows.append({
            'Section': 'Dataset Overview',
            'Metric': 'Records with coordinates',
            'Category': '',
            'Value': coords_count,
            'Count': '',
            'Percent': ''
        })

    if 'Peak_Pressure_kbar' in df.columns and 'Peak_Temperature_C' in df.columns:
        pt_count = df.dropna(subset=['Peak_Pressure_kbar', 'Peak_Temperature_C']).shape[0]
        summary_rows.append({
            'Section': 'Dataset Overview',
            'Metric': 'Records with P–T data',
            'Category': '',
            'Value': pt_count,
            'Count': '',
            'Percent': ''
        })

    if 'DOI' in df.columns:
        doi_count = df['DOI'].notna().sum()
        summary_rows.append({
            'Section': 'Dataset Overview',
            'Metric': 'Records with DOI',
            'Category': '',
            'Value': doi_count,
            'Count': '',
            'Percent': ''
        })

    # Numeric composition and P–T summaries
    numeric_columns = [
        'Almandine_%', 'Pyrope_%', 'Grossular_%', 'Spessartine_%',
        'Peak_Pressure_kbar', 'Peak_Temperature_C', 'Elevation_m',
        'SiO2_wt%', 'Al2O3_wt%', 'FeO_wt%', 'MnO_wt%', 'MgO_wt%', 'CaO_wt%'
    ]

    for column in numeric_columns:
        if column in df.columns:
            series = pd.to_numeric(df[column], errors='coerce')
            summary_rows.append({
                'Section': 'Numeric Summary',
                'Metric': 'count',
                'Category': column,
                'Value': int(series.count()),
                'Count': '',
                'Percent': ''
            })
            summary_rows.append({
                'Section': 'Numeric Summary',
                'Metric': 'mean',
                'Category': column,
                'Value': series.mean(),
                'Count': '',
                'Percent': ''
            })
            summary_rows.append({
                'Section': 'Numeric Summary',
                'Metric': 'standard deviation',
                'Category': column,
                'Value': series.std(),
                'Count': '',
                'Percent': ''
            })
            summary_rows.append({
                'Section': 'Numeric Summary',
                'Metric': 'minimum',
                'Category': column,
                'Value': series.min(),
                'Count': '',
                'Percent': ''
            })
            summary_rows.append({
                'Section': 'Numeric Summary',
                'Metric': '25th percentile',
                'Category': column,
                'Value': series.quantile(0.25),
                'Count': '',
                'Percent': ''
            })
            summary_rows.append({
                'Section': 'Numeric Summary',
                'Metric': 'median',
                'Category': column,
                'Value': series.median(),
                'Count': '',
                'Percent': ''
            })
            summary_rows.append({
                'Section': 'Numeric Summary',
                'Metric': '75th percentile',
                'Category': column,
                'Value': series.quantile(0.75),
                'Count': '',
                'Percent': ''
            })
            summary_rows.append({
                'Section': 'Numeric Summary',
                'Metric': 'maximum',
                'Category': column,
                'Value': series.max(),
                'Count': '',
                'Percent': ''
            })

    # Categorical count summaries
    categorical_columns = [
        'Country', 'Location', 'Lithology_Category',
        'Author_Lithology', 'Cluster', 'DOI'
    ]

    for column in categorical_columns:
        if column in df.columns:
            category_counts = df[column].dropna().astype(str).value_counts()
            total_categories = category_counts.sum()
            if column == 'DOI':
                category_items = category_counts.items()
            else:
                category_items = category_counts.head(top_n_categories).items()
            for category, count in category_items:
                percent = (count / total_categories * 100) if total_categories else 0.0
                summary_rows.append({
                    'Section': 'Categorical Summary',
                    'Metric': column,
                    'Category': category,
                    'Value': '',
                    'Count': int(count),
                    'Percent': round(percent, 2)
                })

    summary_df = pd.DataFrame(summary_rows, columns=['Section', 'Metric', 'Category', 'Value', 'Count', 'Percent'])
    return summary_df


garnet_df = get_garnet_data()
# Precompute dominant garnet for database (for filtering/annotation)
endmember_cols = ['Almandine_%', 'Pyrope_%', 'Grossular_%', 'Spessartine_%']
if all(col in garnet_df.columns for col in endmember_cols):
    endmember_data = garnet_df[endmember_cols]
    garnet_df['dominant_garnet'] = endmember_data.fillna(-np.inf).idxmax(axis=1).where(
        endmember_data.notna().any(axis=1)
    )
else:
    garnet_df['dominant_garnet'] = None

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.

# Unit mapping for descriptive names
UNIT_MAPPING = {
    'AC': 'AC - Asian Craton',
    'GB': 'GB - Gangdese Batholith',
    'GD': 'GD - Gneiss Domes',
    'GHC': 'GHC - Greater Himalayan Crystallines',
    'HHL': 'HHL - High Himalayan Leucogranites',
    'IC': 'IC - Indian Shield/Craton',
    'ITSZ': 'ITSZ - Indus Tsangpot Suture Zone',
    'LHS': 'LHS - Lesser Himalayan Sequence',
    'MCT': 'MCT - Main Central Thrust',
    'NHG': 'NHG - North Himalayan Granites',
    'THS': 'THS - Tethyan Himalaya Sequence'
}

unit_values = sorted(garnet_df['Unit'].dropna().unique())
if garnet_df['Unit'].isna().any():
    unit_values.append('Other')
unit_options = [UNIT_MAPPING.get(unit, unit) for unit in unit_values]
country_options = sorted(garnet_df['Country'].dropna().unique())
lithology_options = sorted(garnet_df['Lithology_Category'].dropna().unique())

pressure_values = pd.to_numeric(garnet_df['Peak_Pressure_kbar'], errors='coerce').dropna()
temperature_values = pd.to_numeric(garnet_df['Peak_Temperature_C'], errors='coerce').dropna()
pressure_bounds = (float(pressure_values.min()), float(pressure_values.max())) if not pressure_values.empty else (0.0, 100.0)
temperature_bounds = (float(temperature_values.min()), float(temperature_values.max())) if not temperature_values.empty else (0.0, 1000.0)


def _clear_all_filters():
    st.session_state['filter_countries'] = []
    st.session_state['filter_units'] = []
    st.session_state['filter_lithologies'] = []
    st.session_state['filter_pressure'] = pressure_bounds
    st.session_state['filter_temperature'] = temperature_bounds
    st.session_state['map_selection_version'] = st.session_state.get('map_selection_version', 0) + 1


if 'map_selection_version' not in st.session_state:
    st.session_state['map_selection_version'] = 0
if 'filter_pressure' not in st.session_state:
    st.session_state['filter_pressure'] = pressure_bounds
if 'filter_temperature' not in st.session_state:
    st.session_state['filter_temperature'] = temperature_bounds

with st.sidebar:
    st.subheader('Filter analyses')
    selected_countries = st.multiselect('Country', country_options, key='filter_countries')
    selected_unit_descriptions = st.multiselect('Tectonic unit', unit_options, key='filter_units')
    selected_lithologies = st.multiselect('Lithology', lithology_options, key='filter_lithologies')

    if pressure_bounds[0] < pressure_bounds[1]:
        pressure_range_global = st.slider(
            'Pressure range (kbar)', *pressure_bounds, key='filter_pressure'
        )
    else:
        pressure_range_global = pressure_bounds

    if temperature_bounds[0] < temperature_bounds[1]:
        temperature_range_global = st.slider(
            'Temperature range (°C)', *temperature_bounds, key='filter_temperature'
        )
    else:
        temperature_range_global = temperature_bounds

    st.button('Clear all filters', on_click=_clear_all_filters, use_container_width=True)

filtered_df = garnet_df.copy()
if selected_countries:
    filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]
if selected_lithologies:
    filtered_df = filtered_df[filtered_df['Lithology_Category'].isin(selected_lithologies)]
if selected_unit_descriptions:
    selected_units = [description.split(' - ')[0] for description in selected_unit_descriptions]
    unit_mask = filtered_df['Unit'].isin([unit for unit in selected_units if unit != 'Other'])
    if 'Other' in selected_units:
        unit_mask |= filtered_df['Unit'].isna() | filtered_df['Unit'].eq('Other')
    filtered_df = filtered_df[unit_mask]

if pressure_range_global != pressure_bounds:
    pressure = pd.to_numeric(filtered_df['Peak_Pressure_kbar'], errors='coerce')
    filtered_df = filtered_df[pressure.between(*pressure_range_global)]
if temperature_range_global != temperature_bounds:
    temperature = pd.to_numeric(filtered_df['Peak_Temperature_C'], errors='coerce')
    filtered_df = filtered_df[temperature.between(*temperature_range_global)]

overview_tab, explore_tab, chemistry_tab, pt_tab, match_tab, submit_tab, himalaya_tab, docs_tab = st.tabs(
    ['Overview', 'Explore', 'Chemistry', 'P-T', 'Compare Your Garnets', 'Submit Data', 'Himalayan detrital garnets', 'Data & Docs']
)

with overview_tab:
    st.header('Dataset Overview', divider='gray')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Total Analyses', len(garnet_df))
    with col2:
        st.metric('Countries', garnet_df['Country'].nunique())
    with col3:
        unique_locations = garnet_df[['Latitude', 'Longitude']].drop_duplicates().shape[0]
        st.metric('Geographic Sites', unique_locations)
    with col4:
        st.metric('Lithology Types', garnet_df['Lithology_Category'].nunique())

    st.subheader('Garnet locations')
    st.markdown('[Open the full Himalayan Garnet ArcGIS map](https://tinyurl.com/HimalayanGarnetMap)')
    map_data = filtered_df.dropna(subset=['Latitude', 'Longitude']).copy()
    if not map_data.empty:
        map_data['_analysis_index'] = map_data.index
        map_data['Unit'] = map_data['Unit'].fillna('Other')
        color_options = [
            column for column in ['Unit', 'Country', 'Lithology_Category']
            if column in map_data.columns and map_data[column].nunique() > 1
        ]
        color_by = st.selectbox('Color locations by', color_options, key='overview_map_color') if color_options else None
        map_styles = {
            'Open Street Map': 'open-street-map',
            'CartoDB Positron': 'carto-positron',
            'CartoDB Dark Matter': 'carto-darkmatter'
        }
        map_style = st.selectbox('Map style', list(map_styles), key='overview_map_style')
        fig = px.scatter_map(
            map_data,
            lat='Latitude',
            lon='Longitude',
            custom_data=['_analysis_index'],
            hover_name='Location',
            hover_data=['Country', 'Lithology_Category', 'Garnet_ID', 'Elevation_m'],
            color=color_by,
            color_discrete_map=UNIT_COLORS,
            title=f'{len(map_data):,} analyses with coordinates',
            zoom=4,
            height=600
        )
        fig.update_layout(
            map={'style': map_styles[map_style], 'center': {'lat': 28.5, 'lon': 84.0}, 'zoom': 4},
            dragmode='select',
            margin={'r': 0, 't': 45, 'l': 0, 'b': 0}
        )
        map_filter_signature = hash((
            tuple(selected_countries), tuple(selected_unit_descriptions), tuple(selected_lithologies),
            pressure_range_global, temperature_range_global, st.session_state['map_selection_version']
        ))
        map_event = st.plotly_chart(
            fig,
            width='stretch',
            on_select='rerun',
            selection_mode='box',
            key=f'garnet_map_{map_filter_signature}'
        )
        selected_points = map_event.selection.get('points', [])
        selected_indices = [
            int(point['customdata'][0])
            for point in selected_points
            if point.get('customdata') and int(point['customdata'][0]) in map_data.index
        ]
        if selected_indices:
            filtered_df = filtered_df.loc[selected_indices]
            st.info(f'{len(filtered_df):,} analyses selected by the map. Draw another box to update the selection.')
        st.caption('Drag a box on the map to filter analyses by location. Clear all filters in the sidebar to reset.')
    else:
        st.info('No analyses with coordinates match the current filters.')

with explore_tab:
    st.header('Explore analyses', divider='gray')
    st.subheader(f'Showing {len(filtered_df):,} analyses')
    st.dataframe(filtered_df.head(100), width='stretch')
    st.download_button(
        label='Download filtered analyses (CSV)',
        data=filtered_df.to_csv(index=False),
        file_name='filtered_garnet_data.csv',
        mime='text/csv',
        help='Download all analyses matching the current sidebar filters and map selection.',
        type='primary'
    )

with chemistry_tab:
    # Chemical composition plots
    st.header('Chemical Composition Analysis', divider='gray')

    chem_filtered_df = filtered_df.copy()

    st.download_button(
        label='📥 Download Filtered Chemical Composition Data (CSV)',
        data=chem_filtered_df.to_csv(index=False).encode('utf-8'),
        file_name='filtered_chemical_composition_data.csv',
        mime='text/csv',
        key='chem_download',
        type="primary"
    )

    # Select elements to plot
    elements = ['SiO2_wt%', 'Al2O3_wt%', 'FeO_wt%', 'MnO_wt%', 'MgO_wt%', 'CaO_wt%']
    selected_elements = st.multiselect(
        'Select elements to analyze',
        elements,
        default=['SiO2_wt%', 'Al2O3_wt%', 'FeO_wt%', 'MgO_wt%', 'CaO_wt%']
    )

    if selected_elements:
        # Histograms
        st.subheader("Element Distributions")
        st.write(f"**Filtered analyses:** {len(chem_filtered_df)}")
        for element in selected_elements:
            if element in chem_filtered_df.columns:
                fig = px.histogram(
                    chem_filtered_df,
                    x=element,
                    color=chem_filtered_df['Unit'].fillna('Other'),
                    title=f"Distribution of {element}",
                    nbins=50,
                    barmode='overlay',
                    opacity=0.65,
                    color_discrete_map=UNIT_COLORS
                )
                st.plotly_chart(fig, width='stretch')

    # Garnet end-member compositions
    st.header('Garnet End-Member Compositions', divider='gray')

    em_filtered_df = filtered_df.copy()

    st.write(f"**Filtered analyses:** {len(em_filtered_df)}")

    st.download_button(
        label='📥 Download Filtered End-Member Data (CSV)',
        data=em_filtered_df.to_csv(index=False).encode('utf-8'),
        file_name='filtered_end_member_data.csv',
        mime='text/csv',
        key='em_download',
        type='primary'
    )

    end_members = ['Almandine_%', 'Spessartine_%', 'Pyrope_%', 'Grossular_%']

    # Check if end-member columns exist
    available_end_members = [em for em in end_members if em in em_filtered_df.columns]

    if available_end_members:
        st.subheader("End-Member Proportions")

        # Bar chart of average compositions
        avg_compositions = em_filtered_df[available_end_members].mean()

        fig = px.bar(
            x=avg_compositions.index,
            y=avg_compositions.values,
            title="Average Garnet End-Member Compositions",
            labels={'x': 'End Member', 'y': 'Percentage (%)'},
            color_discrete_map=UNIT_COLORS
        )
        st.plotly_chart(fig, width='stretch')

        # Ternary diagram if we have the main components
        required_components = ['Almandine_%', 'Pyrope_%', 'Grossular_%', 'Spessartine_%']
        if all(comp in em_filtered_df.columns for comp in required_components):
            st.subheader("Garnet Ternary Diagrams")

            # Ternary plot options
            ternary_options = {
                "Almandine-Pyrope-Grossular": {
                    "a": "Almandine_%",
                    "b": "Pyrope_%",
                    "c": "Grossular_%",
                    "title": "Almandine-Pyrope-Grossular"
                },
                "Pyrope-(Almandine+Spessartine)-Grossular": {
                    "components": {
                        "Pyrope": "Pyrope_%",
                        "Almandine+Spessartine": ["Almandine_%", "Spessartine_%"],
                        "Grossular": "Grossular_%"
                    },
                    "title": "Pyrope-(Almandine+Spessartine)-Grossular"
                },
                "Grossular-Pyrope-Spessartine": {
                    "a": "Grossular_%",
                    "b": "Pyrope_%",
                    "c": "Spessartine_%",
                    "title": "Grossular-Pyrope-Spessartine"
                },
                "Grossular-Spessartine-Almandine": {
                    "a": "Grossular_%",
                    "b": "Spessartine_%",
                    "c": "Almandine_%",
                    "title": "Grossular-Spessartine-Almandine"
                },
                "Spessartine-Grossular-(Almandine+Pyrope)": {
                    "components": {
                        "Spessartine": "Spessartine_%",
                        "Grossular": "Grossular_%",
                        "Almandine+Pyrope": ["Almandine_%", "Pyrope_%"]
                    },
                    "title": "Spessartine-Grossular-(Almandine+Pyrope)"
                }
            }

            selected_ternary = st.selectbox(
                "Select Ternary Diagram Type:",
                options=list(ternary_options.keys()),
                index=0
            )

            ternary_config = ternary_options[selected_ternary]
            ternary_data = em_filtered_df[required_components].copy()

            # For ternary plot, we need to normalize to 100%
            if "components" in ternary_config:
                # Handle combined components
                plot_data = pd.DataFrame()
                for name, cols in ternary_config["components"].items():
                    if isinstance(cols, list):
                        plot_data[name] = ternary_data[[c for c in cols if c in ternary_data.columns]].sum(axis=1)
                    else:
                        plot_data[name] = ternary_data[cols]
            
                plot_data = plot_data.div(plot_data.sum(axis=1), axis=0) * 100
                a, b, c = list(plot_data.columns)
                plot_data['Unit'] = em_filtered_df['Unit'].fillna('Other').to_numpy()

                fig = px.scatter_ternary(
                    plot_data,
                    a=a,
                    b=b,
                    c=c,
                    color='Unit',
                    color_discrete_map=UNIT_COLORS,
                    title=f"Garnet Compositions in {ternary_config['title']} Space"
                )
            else:
                # Standard ternary plot
                plot_data = ternary_data[[ternary_config['a'], ternary_config['b'], ternary_config['c']]].copy()
                plot_data = plot_data.div(plot_data.sum(axis=1), axis=0) * 100
                plot_data['Unit'] = em_filtered_df['Unit'].fillna('Other').to_numpy()

                fig = px.scatter_ternary(
                    plot_data,
                    a=ternary_config['a'],
                    b=ternary_config['b'],
                    c=ternary_config['c'],
                    color='Unit',
                    color_discrete_map=UNIT_COLORS,
                    title=f"Garnet Compositions in {ternary_config['title']} Space"
                )
        
            st.plotly_chart(fig, width='stretch')

with pt_tab:
    # P-T conditions
    st.header('Pressure-Temperature Conditions', divider='gray')

    pt_filtered_df = filtered_df.copy()

    st.write(f"**Filtered analyses:** {len(pt_filtered_df)}")

    st.download_button(
        label='📥 Download Filtered P-T Data (CSV)',
        data=pt_filtered_df.to_csv(index=False).encode('utf-8'),
        file_name='filtered_pressure_temperature_data.csv',
        mime='text/csv',
        key='pt_download',
        type='primary'
    )

    pt_cols = ['Peak_Pressure_kbar', 'Peak_Temperature_C']
    available_pt = [col for col in pt_cols if col in pt_filtered_df.columns]

    if available_pt:
        st.subheader("P-T Distribution")

        # Remove NaN values for plotting
        pt_data = pt_filtered_df[available_pt + ['Unit']].dropna(subset=available_pt).copy()
        pt_data['Unit'] = pt_data['Unit'].fillna('Other')

        if len(pt_data) > 0:
            fig = px.scatter(
                pt_data,
                x='Peak_Temperature_C',
                y='Peak_Pressure_kbar',
                color='Unit',
                title="Peak P-T Conditions",
                color_discrete_map=UNIT_COLORS,
                labels={
                    'Peak_Temperature_C': 'Temperature (°C)',
                    'Peak_Pressure_kbar': 'Pressure (kbar)'
                }
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No P-T data available for selected filters")

with match_tab:
    # Compare Your Garnets section
    st.header('Compare Your Garnets', divider='gray')

    st.subheader('Garnet Matching Tool')

    st.markdown("""
    Use this tool to compare new garnet end-member compositions with garnets in the Himalayan Garnet Provenance Database.
    Enter a single analysis for a quick comparison, or upload a CSV file to compare an entire dataset.
    """)

    with st.expander("Getting Accurate Matches", expanded=False):
        st.markdown("""
        - **Use precise end-member values**
          Use the most precise garnet end-member data available. Small differences in Alm, Prp, Grs, and Sps can affect similarity rankings.
        - **Compare like with like**
          Consider lithology, tectonic unit, metamorphic grade, and geographic setting. A mathematically close match may not always be geologically meaningful.
        - **Use multiple analyses when possible**
          Compare several grains, or core-rim analyses from the same sample, rather than relying on one-point analysis.
        """)

    with st.expander("Similarity Methods", expanded=False):
        st.markdown("""
        **Aitchison Distance:** Aitchison distance is designed for compositional data, where values are proportions that sum to a constant, such as garnet end-member percentages.
        - **Best for:** normalized end-member compositions
        - **Interpretation:** lower Aitchison distance means greater compositional similarity
        - **Advantage:** handles proportional data better than ordinary straight-line distance methods

        This is likely your best default method for garnet end-member comparisons.

        **End-Member Tolerance:** End-member tolerance identifies database samples whose Alm, Prp, Grs, and Sps values fall within a user-defined range of the submitted garnet composition. For example, a ±5% tolerance means a match must be within 5 percentage points for each selected end-member.
        - **Best for:** simple, transparent matching
        - **Interpretation:** matched samples fall within the selected compositional tolerance
        - **Advantage:** easy for users to understand and geologically intuitive

        **Dominant Unit Filter:** The comparison tool can also apply a dominant unit filter. After the closest compositional matches are identified, the tool determines which geological unit contributes the greatest proportion of matches. That unit is treated as the dominant matching unit, and results can be filtered to emphasize samples from that unit.

        This helps answer a geological question: “Which Himalayan unit does this garnet composition most closely resemble?” For example, if most of the closest matches come from the Greater Himalayan Sequence, the tool can apply a dominant-unit filter to focus the comparison on that unit.

        This is useful because it combines:
        - chemical similarity
        - geological context
        - regional provenance logic

        However, the dominant unit should not be treated as a definitive source assignment. It is a guide to the most common geological context among the best-matching samples.
        """)

    with st.expander("Interpreting Results", expanded=False):
        st.markdown("""
        - **Lower distance values indicate closer matches**
          For the Aitchison distance, lower values mean greater similarity.
        - **Tolerance matches are threshold-based**
          For end-member tolerance, samples either fall within the selected tolerance or they do not. A smaller tolerance gives stricter matches.
        - **Check geological plausibility**
          A strong chemical match should still make sense in terms of lithology, metamorphic conditions, tectonic unit, and geography.
        - **Multiple matching units may be meaningful**
          If good matches occur in several units, the garnet composition may not be unique to one tectonic setting. This could reflect shared protoliths, similar metamorphic conditions, or mixed provenance.
        - **No close matches can be informative**
          A lack of matches may indicate a rare composition, incomplete database coverage, unusual metamorphic/magmatic history, or analytical/normalization issues.
        - **Use the result as a comparison, not a final interpretation**
          The tool identifies compositional similarity. Geological interpretation should also consider field context, petrography, P–T conditions, mineral assemblage, and age data.
        """)

    st.info("""
    **Important Note**: Compositional matches are not unique source identifications. They should be interpreted as provenance hypotheses
    and evaluated alongside stratigraphic age, drainage context, geographic proximity, lithology, and regional structural geology.
    """)

    # Mode selection
    mode = st.radio("Select Comparison Mode:", ["Quick Compare", "Upload Dataset"], horizontal=True)

    if mode == "Quick Compare":
        st.subheader("Quick Compare")
        st.markdown("Enter a single garnet end-member composition for comparison.")

        # Input fields
        col1, col2 = st.columns(2)

        with col1:
            almandine = st.number_input("Almandine (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
            pyrope = st.number_input("Pyrope (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)

        with col2:
            grossular = st.number_input("Grossular (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
            spessartine = st.number_input("Spessartine (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

        # Check sum
        total = almandine + pyrope + grossular + spessartine
        if abs(total - 100.0) > 1.0:
            st.warning(f"End-member percentages sum to {total:.1f}%. They should sum to approximately 100%.")

        # Optional inputs
        sample_name = st.text_input("Sample Name (optional)")
        locality = st.text_input("Locality/Section (optional)")
        notes = st.text_area("Notes (optional)", height=100)

        # Similarity method
        similarity_method = st.selectbox(
            "Similarity Method",
            ["Aitchison distance", "End-member tolerance"],
            index=0
        )

        if similarity_method == "End-member tolerance":
            tolerance = st.slider("Tolerance (±%)", min_value=2, max_value=15, value=5, step=1)

        st.markdown("---")
        st.subheader("Match Filters")

        col1, col2 = st.columns(2)
        with col1:
            use_similarity_filter = st.checkbox("Apply Aitchison similarity class filter", value=True)
            similarity_label_options = [
                "High-Similarity (≤ 0.40)",
                "Moderate-Similarity (0.41 - 0.79)",
                "Low-Similarity (> 0.80)"
            ]
            similarity_label_map = {
                "High-Similarity (≤ 0.40)": "High-Similarity",
                "Moderate-Similarity (0.41 - 0.79)": "Moderate-Similarity",
                "Low-Similarity (> 0.80)": "Low-Similarity"
            }
            selected_similarity_labels = st.multiselect(
                "Similarity categories to include:",
                options=similarity_label_options,
                default=similarity_label_options
            )

        with col2:
            use_dominance_filter = st.checkbox("Apply dominance category filter", value=False)
            dominance_label_options = [
                "Strongly dominant",
                "Moderately dominant",
                "Weakly dominant",
                "Mixed",
                "Unique"
            ]
            selected_dominance_labels = st.multiselect(
                "Dominance categories to include:",
                options=dominance_label_options,
                default=dominance_label_options,
                help="Strongly dominant: One unit accounts for ≥70% of the High-Similarity matches.\nModerately dominant: One unit accounts for 55–70% of the High-Similarity matches.\nWeakly dominant: One unit accounts for 45–55% of the High-Similarity matches.\nMixed: No single unit accounts for ≥45% of the High-Similarity matches.\nUnique: Only one High-Similarity match was identified."
            )

        st.caption(
            "Strongly dominant: One unit accounts for ≥70% of High-Similarity matches. "
            "Moderately dominant: 55–70%. Weakly dominant: 45–55%. "
            "Mixed: no single unit ≥45%. Unique: only one High-Similarity match."
        )

        geo_radius = st.selectbox(
            "Geographic Match Radius",
            [
                "No distance filter",
                "50 km",
                "100 km",
                "200 km",
                "300 km",
                "400 km",
                "500 km"
            ],
            index=0
        )

        coord_col1, coord_col2 = st.columns(2)
        with coord_col1:
            latitude_text = st.text_input("Latitude (decimal degrees, optional)", "")
        with coord_col2:
            longitude_text = st.text_input("Longitude (decimal degrees, optional)", "")

        latitude = None
        longitude = None
        if latitude_text.strip() and longitude_text.strip():
            try:
                latitude = float(latitude_text)
                longitude = float(longitude_text)
            except ValueError:
                st.warning("Latitude and Longitude must be numeric values.")

        if geo_radius != "No distance filter" and (latitude is None or longitude is None):
            st.info("Enter both latitude and longitude to apply the geographic radius filter.")

        # Number of matches
        num_matches = st.selectbox("Number of matches to show", [10, 25, 50, 100], index=0)

        if st.button("Find Similar Garnets", type="primary"):
            # Prepare input composition
            input_composition = {
                'Almandine_%': almandine,
                'Pyrope_%': pyrope,
                'Grossular_%': grossular,
                'Spessartine_%': spessartine
            }

            matching_df = garnet_df.copy()
            if latitude is not None and longitude is not None:
                matching_df = add_geographic_distances(matching_df, latitude, longitude)

            if geo_radius != "No distance filter" and latitude is not None and longitude is not None:
                selected_radius_km = int(geo_radius.split()[0])
                matching_df = filter_database_by_radius(matching_df, latitude, longitude, selected_radius_km)
                if matching_df.empty:
                    st.warning("No garnet analyses were found within the selected geographic radius.")

            if similarity_method == "Aitchison distance":
                # Calculate Aitchison distances
                results = calculate_aitchison_distances(matching_df, input_composition)
                results = assign_similarity_and_dominance(results)
                results = results.sort_values(['similarity_score', 'aitchison_distance'], ascending=[False, True])

                if use_similarity_filter and selected_similarity_labels:
                    selected_similarity_values = [similarity_label_map[label] for label in selected_similarity_labels]
                    results = results[results['similarity_class'].isin(selected_similarity_values)]

                if use_dominance_filter and selected_dominance_labels:
                    results = results[results['dominance_category'].isin(selected_dominance_labels)]

                total_matches = len(results)
                top_results = results.head(num_matches)
            else:
                # Find matches within tolerance
                results = find_tolerance_matches(matching_df, input_composition, tolerance, None)
                if 'similarity_score' in results.columns:
                    results = results.sort_values('similarity_score', ascending=False)
                total_matches = len(results)
                top_results = results.head(num_matches)

            if total_matches > 0:
                st.success(f"Found {total_matches} similar garnets")
                if total_matches > len(top_results):
                    st.info(f"Showing top {len(top_results)} of {total_matches} matches.")

                # Show top 5 results in detail
                st.subheader("Top Matches")
                display_comparison_results(results, input_composition, similarity_method)
                display_automatic_interpretation(results, total_matches)
            else:
                st.warning("No matches found with the current criteria.")

    elif mode == "Upload Dataset":
        st.subheader("Upload Dataset")
        st.markdown("Upload a CSV file with garnet end-member compositions for batch comparison.")

        # CSV Format Guide
        with st.expander("📋 CSV Format Guide - Click to expand", expanded=False):
            st.markdown("""
            ### Required Columns
            Your CSV file **must** include these columns:
            - `Sample_ID`: Unique identifier for each sample
            - `Analysis_ID`: Unique identifier for each analysis (can be same as Sample_ID if single analysis per sample)
            - `Almandine_%`: Almandine end-member percentage (0-100)
            - `Pyrope_%`: Pyrope end-member percentage (0-100)
            - `Grossular_%`: Grossular end-member percentage (0-100)
            - `Spessartine_%`: Spessartine end-member percentage (0-100)

            ### Optional Columns (Recommended)
            These columns will enhance your comparison results:
            - `Section`: Field location or section name
            - `Formation`: Geological formation name
            - `MDA_Ma`: Maximum depositional age in millions of years (for provenance timing)
            - `Latitude`: Latitude in decimal degrees (for geographic plotting)
            - `Longitude`: Longitude in decimal degrees (for geographic plotting)
            - `Notes`: Additional notes about the sample

            ### Important Notes
            - **End-member sums**: All four end-member percentages should sum to approximately 100%
            - **Data quality**: Higher quality analyses will give more reliable matches
            - **Geographic context**: Including coordinates helps visualize potential source areas
            - **Age constraints**: MDA ages help evaluate whether matches are geologically plausible

            ### Example CSV Format
            ```csv
            Sample_ID,Analysis_ID,Almandine_%,Pyrope_%,Grossular_%,Spessartine_%,Section,Formation,MDA_Ma,Latitude,Longitude,Notes
            SAMPLE_001,ANALYSIS_001,45.2,32.1,18.7,4.0,River_Section_A,Unknown,25.5,28.5,85.3,Detrital garnet from modern river
            SAMPLE_002,ANALYSIS_002,67.8,15.2,12.0,5.0,Mountain_Stream,Schist,18.2,29.1,86.7,Single grain analysis
            ```
            """)

            template_csv = """Sample_ID,Analysis_ID,Almandine_%,Pyrope_%,Grossular_%,Spessartine_%,Section,Formation,MDA_Ma,Latitude,Longitude,Notes
    SAMPLE_001,ANALYSIS_001,45.2,32.1,18.7,4.0,River_Section_A,Unknown,25.5,28.5,85.3,Detrital garnet from modern river
    SAMPLE_002,ANALYSIS_002,67.8,15.2,12.0,5.0,Mountain_Stream,Schist,18.2,29.1,86.7,Single grain analysis"""

            st.download_button(
                label="📥 Download CSV Template (CSV)",
                data=template_csv,
                file_name="garnet_comparison_template.csv",
                mime="text/csv",
                help="Download a template CSV file with the correct format",
                type="primary"
            )

        # File uploader
        st.markdown(
            """
            **Upload policy:** Only CSV files are accepted. Uploaded datasets are processed temporarily in memory and are not permanently stored or used to overwrite any master dataset, reference database, or metadata table.
            """
        )
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")

        if uploaded_file is not None:
            try:
                valid_upload, upload_error = validate_uploaded_csv_file(uploaded_file)
                if not valid_upload:
                    st.error(f"❌ {upload_error}")
                    st.stop()

                # Try different parsing approaches for robustness
                upload_df = None
                parse_error = None

                # First try standard pandas read_csv
                try:
                    upload_df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    # Try different encodings
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        upload_df = pd.read_csv(uploaded_file, encoding='latin1')
                    except Exception as e:
                        parse_error = f"Encoding error: {str(e)}"
                except pd.errors.ParserError as e:
                    parse_error = f"CSV parsing error: {str(e)}"
                except Exception as e:
                    parse_error = f"File reading error: {str(e)}"

                if upload_df is None:
                    st.error(f"❌ Could not read the CSV file: {parse_error}")
                    st.info("💡 **Troubleshooting tips:**")
                    st.info("- Ensure the file is a valid CSV format")
                    st.info("- Try saving as 'CSV UTF-8' from Excel")
                    st.info("- Check for special characters in column names")
                    st.info("- Make sure the file isn't corrupted")
                    st.stop()

                # Show data preview
                st.subheader("📊 Data Preview")
                st.info(f"Found {len(upload_df)} rows and {len(upload_df.columns)} columns")

                # Preview first few rows
                st.dataframe(upload_df.head(5), width='stretch')

                # Show detected column names
                st.subheader("📋 Detected Columns")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**All columns found:**")
                    for i, col in enumerate(upload_df.columns, 1):
                        st.write(f"{i}. `{col}`")

                # Normalize column names for matching (case-insensitive, strip spaces)
                normalized_cols = {col.strip().lower(): col for col in upload_df.columns}

                # Required columns with flexible matching
                required_mappings = {
                    'sample_id': ['sample_id', 'sample', 'id', 'sampleid', 'sample_id'],
                    'analysis_id': ['analysis_id', 'analysis', 'analysisid', 'spot_id', 'point_id'],
                    'almandine_%': ['almandine_%', 'almandine', 'alm_%', 'alm', 'almandine_pct'],
                    'pyrope_%': ['pyrope_%', 'pyrope', 'pyr_%', 'pyr', 'pyrope_pct'],
                    'grossular_%': ['grossular_%', 'grossular', 'grs_%', 'grs', 'grossular_pct'],
                    'spessartine_%': ['spessartine_%', 'spessartine', 'sps_%', 'sps', 'spessartine_pct']
                }

                # Find matches for required columns
                matched_columns = {}
                missing_required = []

                for req_col, possible_names in required_mappings.items():
                    found = False
                    for possible in possible_names:
                        if possible in normalized_cols:
                            matched_columns[req_col] = normalized_cols[possible]
                            found = True
                            break
                    if not found:
                        missing_required.append(req_col)

                with col2:
                    st.write("**Column matching status:**")
                    for req_col in required_mappings.keys():
                        if req_col in matched_columns:
                            status = f"✅ Found: `{matched_columns[req_col]}`"
                        else:
                            status = "❌ Missing"
                        st.write(f"**{req_col.replace('_', ' ').title()}:** {status}")

                # Check if we have all required columns
                if missing_required:
                    st.error(f"❌ Missing required columns: {', '.join(missing_required)}")

                    st.info("💡 **Column name suggestions:**")
                    suggestions = {
                        'sample_id': 'Sample_ID, Sample, ID, SampleID',
                        'analysis_id': 'Analysis_ID, Analysis, Spot_ID, Point_ID',
                        'almandine_%': 'Almandine_%, Almandine, Alm_%, Alm',
                        'pyrope_%': 'Pyrope_%, Pyrope, Pyr_%, Pyr',
                        'grossular_%': 'Grossular_%, Grossular, Grs_%, Grs',
                        'spessartine_%': 'Spessartine_%, Spessartine, Sps_%, Sps'
                    }

                    for missing in missing_required:
                        st.info(f"- **{missing.replace('_', ' ').title()}:** {suggestions[missing]}")

                    st.warning("Please rename your columns to match the expected format and try again.")
                    st.stop()

                # Success - we have all required columns
                st.success("✅ All required columns found!")

                # Optional columns
                optional_mappings = {
                    'section': ['section', 'locality', 'location', 'site'],
                    'formation': ['formation', 'unit', 'lithology'],
                    'mda_ma': ['mda_ma', 'mda', 'age_ma', 'age', 'max_age'],
                    'latitude': ['latitude', 'lat', 'y'],
                    'longitude': ['longitude', 'lon', 'lng', 'x'],
                    'notes': ['notes', 'comments', 'description']
                }

                optional_found = {}
                for opt_col, possible_names in optional_mappings.items():
                    for possible in possible_names:
                        if possible in normalized_cols:
                            optional_found[opt_col] = normalized_cols[possible]
                            break

                if optional_found:
                    st.info(f"📍 Found optional columns: {', '.join(optional_found.keys())} - These will enhance your results!")

                # Create standardized dataframe
                std_df = pd.DataFrame()
                for req_col, orig_col in matched_columns.items():
                    std_df[req_col] = upload_df[orig_col]

                for opt_col, orig_col in optional_found.items():
                    std_df[opt_col] = upload_df[orig_col]

                # Data validation
                st.subheader("🔍 Data Validation")

                # Check for numeric data in endmember columns
                endmember_cols = ['almandine_%', 'pyrope_%', 'grossular_%', 'spessartine_%']
                validation_issues = []

                for col in endmember_cols:
                    try:
                        std_df[col] = pd.to_numeric(std_df[col], errors='coerce')
                        if std_df[col].isna().any():
                            validation_issues.append(f"Non-numeric values in {col}")
                    except:
                        validation_issues.append(f"Could not convert {col} to numbers")

                # Check compositions sum to ~100%
                if not validation_issues:
                    sums = std_df[endmember_cols].sum(axis=1)
                    bad_sums = std_df[abs(sums - 100) > 1]

                    if len(bad_sums) > 0:
                        st.warning(f"⚠️ {len(bad_sums)} samples have end-member percentages that don't sum to ~100%")
                        st.info("This may affect matching accuracy. Consider normalizing your data.")
                        with st.expander("View problematic samples"):
                            preview_cols = ['sample_id', 'analysis_id'] + endmember_cols
                            st.dataframe(bad_sums[preview_cols].head(10))

                if validation_issues:
                    st.error("❌ Data validation issues found:")
                    for issue in validation_issues:
                        st.error(f"- {issue}")
                    st.stop()

                # Show summary
                st.subheader("📈 Dataset Summary")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Analyses", len(std_df))

                with col2:
                    unique_samples = std_df['sample_id'].nunique()
                    st.metric("Unique Samples", unique_samples)

                with col3:
                    if 'latitude' in std_df.columns and 'longitude' in std_df.columns:
                        coords = std_df.dropna(subset=['latitude', 'longitude'])
                        st.metric("With Coordinates", len(coords))
                    else:
                        st.metric("With Coordinates", 0)

                # Proceed to comparison
                st.subheader("🔬 Comparison Settings")

                # Similarity method
                upload_similarity_method = st.selectbox(
                    "Similarity Method",
                    ["Aitchison distance"],
                    index=0,
                    key="upload_similarity"
                )

                # Matches per sample
                matches_per_sample = st.selectbox("Matches per uploaded garnet", [1, 3, 5, 10], index=2)

                # Dominant garnet options
                st.write("\n")
                show_dominant = st.checkbox("Show dominant garnet for uploaded samples", value=True)
                apply_dominant_filter = st.checkbox("Apply dominant-garnet filter when matching (restrict DB to same dominant)", value=False)

                if st.button("Compare Uploaded Dataset", type="primary"):
                    with st.spinner("Comparing dataset... This may take a moment."):
                        # Compute dominant garnet for uploaded data (based on provided end-members)
                        em_cols_lower = ['almandine_%', 'pyrope_%', 'grossular_%', 'spessartine_%']
                        std_df['dominant_garnet'] = std_df[em_cols_lower].idxmax(axis=1).str.replace('_%', '_%').str.title()

                        # Prepare upload array (order: Almandine, Pyrope, Grossular, Spessartine)
                        upload_array = std_df[em_cols_lower].to_numpy(dtype=float)

                        # If applying dominant filter, group uploads by their dominant and match against filtered DB subsets
                        all_results_list = []
                        if apply_dominant_filter:
                            # For each unique dominant value in uploads
                            for dom_value, group in std_df.groupby(std_df['dominant_garnet']):
                                # Find DB indices matching that dominant value
                                db_mask = garnet_df['dominant_garnet'] == dom_value
                                if db_mask.sum() == 0:
                                    # No matches in DB for this dominant; skip
                                    continue
                                db_subset = garnet_df[db_mask]
                                subset_indices = db_subset.index.to_numpy()
                                # Compute CLR of DB subset (use cached full DB CLR and slice)
                                db_clr_full, db_index_full = get_db_clr()
                                subset_positions = np.where(np.isin(db_index_full, subset_indices))[0]
                                db_clr_subset = db_clr_full[subset_positions]

                                # Prepare upload rows for this group
                                group_rows = group.index.to_numpy()
                                group_array = std_df.loc[group_rows, em_cols_lower].to_numpy(dtype=float)
                                group_clr = clr_transform_array(group_array)

                                # Compute distances and select top matches
                                dists = pairwise_distances(group_clr, db_clr_subset, metric='euclidean', n_jobs=-1)
                                for i_local, row_idx in enumerate(group_rows):
                                    dist_row = dists[i_local]
                                    top_idx = np.argsort(dist_row)[:matches_per_sample]
                                    for rank, tidx in enumerate(top_idx, start=1):
                                        db_pos = subset_positions[tidx]
                                        db_row = garnet_df.iloc[db_index_full[db_pos]].copy()
                                        rec = db_row.to_dict()
                                        rec['aitchison_distance'] = float(dist_row[tidx])
                                        rec['similarity_score'] = float(1.0 / (1.0 + dist_row[tidx]))
                                        rec['uploaded_row'] = int(row_idx)
                                        rec['match_rank'] = rank
                                        all_results_list.append(rec)
                        else:
                            # Fast batch compute across entire DB
                            results_batch = batch_aitchison_distances_for_upload(garnet_df, upload_array, matches_per_sample)
                            if not results_batch.empty:
                                for _, r in results_batch.iterrows():
                                    rec = r.to_dict()
                                    # uploaded_row refers to the zero-based position in the upload
                                    rec['uploaded_row'] = int(rec.get('uploaded_row', -1))
                                    all_results_list.append(rec)

                        if not all_results_list:
                            results_df = pd.DataFrame()
                        else:
                            results_df = pd.DataFrame.from_records(all_results_list)

                        # Add uploaded identifiers back into results
                        if not results_df.empty:
                            results_df['uploaded_sample_id'] = results_df['uploaded_row'].apply(lambda i: std_df.iloc[int(i)]['sample_id'] if (i is not None and i >= 0) else '')
                            results_df['uploaded_analysis_id'] = results_df['uploaded_row'].apply(lambda i: std_df.iloc[int(i)]['analysis_id'] if (i is not None and i >= 0) else '')
                            if 'latitude' in std_df.columns and 'longitude' in std_df.columns:
                                results_df['uploaded_latitude'] = results_df['uploaded_row'].apply(
                                    lambda i: float(std_df.iloc[int(i)]['latitude']) if (i is not None and i >= 0 and pd.notna(std_df.iloc[int(i)]['latitude'])) else np.nan
                                )
                                results_df['uploaded_longitude'] = results_df['uploaded_row'].apply(
                                    lambda i: float(std_df.iloc[int(i)]['longitude']) if (i is not None and i >= 0 and pd.notna(std_df.iloc[int(i)]['longitude'])) else np.nan
                                )
                            else:
                                results_df['uploaded_latitude'] = np.nan
                                results_df['uploaded_longitude'] = np.nan

                            if 'Latitude' in results_df.columns and 'Longitude' in results_df.columns:
                                results_df['Latitude'] = pd.to_numeric(results_df['Latitude'], errors='coerce')
                                results_df['Longitude'] = pd.to_numeric(results_df['Longitude'], errors='coerce')
                                valid_geo = results_df[['uploaded_latitude', 'uploaded_longitude', 'Latitude', 'Longitude']].notna().all(axis=1)
                                results_df.loc[valid_geo, 'match_distance_km'] = haversine_distance_km(
                                    results_df.loc[valid_geo, 'uploaded_latitude'],
                                    results_df.loc[valid_geo, 'uploaded_longitude'],
                                    results_df.loc[valid_geo, 'Latitude'],
                                    results_df.loc[valid_geo, 'Longitude']
                                )
                            else:
                                results_df['match_distance_km'] = np.nan

                        if results_df.empty:
                            st.warning("No matches found for the uploaded dataset (after applying filters).")
                        else:
                            st.success(f"Comparison complete! Found {len(results_df)} matches.")
                            display_upload_results(results_df, std_df)

            except Exception as e:
                st.error(f"❌ Unexpected error processing file: {str(e)}")
                st.info("Please check your CSV file format and try again. If the problem persists, contact support.")

with submit_tab:
    st.subheader('Submit new garnet data')
    st.markdown(
        'Contribute new analyses to keep the database current. Your file is checked immediately, '
        'then sent to a **staging area for curator review**. It is never added to the public dataset automatically.'
    )
    st.download_button(
        'Download bedrock CSV template', submission.template_csv(), 'garnet_submission_template.csv', 'text/csv',
        key='submission_template_download'
    )
    st.caption(
        'Template columns follow the Data Dictionary. Normalized oxides, apfu, end-members and cluster '
        'are calculated by the curators. Pressure in kbar, temperature in °C, oxides in wt%, '
        'coordinates in decimal degrees.'
    )

    st.markdown('**1. Contributor and source**')
    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        sub_name = st.text_input('Name', key='sub_name')
        sub_institution = st.text_input('Institution', key='sub_institution')
        sub_email = st.text_input('Contact email', key='sub_email')
    with meta_col2:
        sub_doi = st.text_input('DOI of the publication (published data)', key='sub_doi')
        sub_repo = st.text_input('Repository link (unpublished data)', key='sub_repo',
                                 help='For example a Zenodo, Figshare or GitHub link.')
        sub_notes = st.text_area('Notes for the curators (optional)', key='sub_notes', height=68)

    st.markdown('**2. Upload and check your file**')
    sub_file = st.file_uploader('CSV file', type=['csv'], key='submission_upload')

    if sub_file is not None:
        sub_bytes = sub_file.getvalue()
        sub_df, read_error = submission.read_upload(sub_bytes)
        if read_error:
            st.error(read_error)
        else:
            report, _ = submission.validate_dataframe(sub_df, garnet_df, lithology_options)
            st.write(f'{len(sub_df):,} rows read.')
            if report.errors:
                st.error(f'{len(report.errors)} problem(s) must be fixed before you can submit:')
                st.dataframe(pd.DataFrame(report.errors), hide_index=True, width='stretch')
            if report.warnings:
                st.warning(f'{len(report.warnings)} item(s) to review:')
                st.dataframe(pd.DataFrame(report.warnings), hide_index=True, width='stretch')
            if report.ok:
                st.success('All required checks passed.')

            st.markdown('**3. Submit for review**')
            meta_errors = submission.validate_metadata(sub_name, sub_email, sub_institution, sub_doi, sub_repo)
            for message in meta_errors:
                st.error(message)
            warnings_ok = st.checkbox(
                'I have reviewed the warnings above.', key='sub_ack', disabled=not report.warnings
            ) if report.warnings else True
            consent = st.checkbox(
                'I have the right to share these data and agree to their release under the database license.',
                key='sub_consent'
            )
            configured = submission.staging_configured(st.secrets)
            if not configured:
                st.info('Submissions are not yet enabled on this site (staging is not configured).')
            can_submit = report.ok and not meta_errors and warnings_ok and consent and configured
            if st.session_state.get('sub_count', 0) >= 3:
                can_submit = False
                st.info('Submission limit reached for this session.')
            if st.button('Submit for review', disabled=not can_submit, key='sub_submit'):
                try:
                    sub_id, _issue_url = submission.stage_submission(
                        sub_bytes,
                        {
                            'name': sub_name.strip(), 'institution': sub_institution.strip(),
                            'email': sub_email.strip(), 'doi': submission.normalize_doi(sub_doi) or '',
                            'repository_link': sub_repo.strip(), 'notes': sub_notes.strip(),
                            'rows': len(sub_df), 'warnings': len(report.warnings),
                        },
                        st.secrets['STAGING_GITHUB_TOKEN'], st.secrets['STAGING_GITHUB_REPO'],
                    )
                    st.session_state['sub_count'] = st.session_state.get('sub_count', 0) + 1
                    st.success(f'Thank you! Submission {sub_id} was received and is awaiting curator review.')
                except Exception:
                    st.error('The submission could not be saved. Please try again later or contact the curators.')

@st.cache_data
def get_siwalik_data():
    return pd.read_csv(Path(__file__).parent/'data/Siwalik.csv', encoding='utf-8-sig').rename(
        columns=lambda c: c.strip().lstrip('\ufeff')).rename(columns={'SGarnet_ID': 'Garnet_ID'})


with himalaya_tab:
    st.header('Himalayan detrital garnets', divider='gray')
    st.write(
        'Detrital garnet analyses from Siwalik (Himalayan foreland basin) sandstones, with major-element '
        'chemistry, end-member proportions, sample locations and depositional ages.'
    )
    siw = get_siwalik_data()
    # Rows without a formation would otherwise be dropped by the filters
    siw['Formation_Name'] = siw['Formation_Name'].fillna('Unspecified')
    ends = ['Almandine_%S', 'Spessartine_%S', 'Pyrope_%S', 'Grossular_%S']
    for c in ends + ['Latitude', 'Longitude', 'MDA_Ma']:
        siw[c] = pd.to_numeric(siw[c], errors='coerce')

    o1, o2, o3, o4 = st.columns(4)
    o1.metric('Total Analyses', len(siw))
    o2.metric('Countries', siw['Country'].nunique())
    o3.metric('Geographic Sites', siw[['Latitude', 'Longitude']].drop_duplicates().shape[0])
    o4.metric('Lithology Types', siw['Lithology'].nunique())

    f1, f2, f3 = st.columns(3)
    countries = sorted(siw['Country'].dropna().unique())
    sel_c = f1.multiselect('Country', countries, default=countries, key='siw_country')
    base = siw[siw['Country'].isin(sel_c)]
    groups = sorted(base['Stratigraphic_Group'].dropna().unique())
    sel_g = f2.multiselect('Stratigraphic group', groups, default=groups, key='siw_group')
    base = base[base['Stratigraphic_Group'].isin(sel_g)]
    forms = sorted(base['Formation_Name'].dropna().unique())
    sel_f = f3.multiselect('Formation', forms, default=forms, key='siw_form')
    sdf = base[base['Formation_Name'].isin(sel_f)]

    ages = siw['MDA_Ma'].dropna()
    if not ages.empty:
        a1, a2 = st.columns([3, 1])
        age_min, age_max = float(ages.min()), float(ages.max())
        age_range = a1.slider(
            'Maximum depositional age (MDA, Ma)', age_min, age_max, (age_min, age_max), key='siw_mda',
        )
        keep_unknown = a2.checkbox('Include samples without MDA', value=True, key='siw_mda_unknown')
        in_range = sdf['MDA_Ma'].between(*age_range)
        sdf = sdf[in_range | (keep_unknown & sdf['MDA_Ma'].isna())]
    st.caption(f'{len(sdf):,} of {len(siw):,} analyses')

    if sdf.empty:
        st.info('No analyses match the selected filters.')
    else:
        st.subheader('Map')
        color_by = st.selectbox('Colour map points by', ['Formation_Name', 'Country', 'Cluster'] + ends + ['MDA_Ma'], key='siw_color')
        mdf = sdf.dropna(subset=['Latitude', 'Longitude'])
        if color_by in ('Cluster',):
            mdf = mdf.assign(Cluster=mdf['Cluster'].astype(str))
        fig_map = px.scatter_map(
            mdf, lat='Latitude', lon='Longitude', color=color_by,
            hover_name='Garnet_ID', custom_data=['Garnet_ID'],
            hover_data=['Author_Sample_ID', 'Formation_Name', 'MDA_Ma'],
            height=500,
        )
        pad = 0.5
        fig_map.update_layout(
            map_style='open-street-map', margin=dict(l=0, r=0, t=0, b=0), dragmode='select',
            map=dict(bounds=dict(
                west=float(mdf['Longitude'].min()) - pad, east=float(mdf['Longitude'].max()) + pad,
                south=float(mdf['Latitude'].min()) - pad, north=float(mdf['Latitude'].max()) + pad,
            )),
        )
        map_event = st.plotly_chart(
            fig_map, width='stretch', on_select='rerun',
            selection_mode=('box', 'lasso', 'points'), key='siw_map',
        )
        st.caption('Use the box or lasso tool in the map toolbar to select samples.')
        picked_ids = {pt['customdata'][0] for pt in (map_event.selection.get('points', []) if map_event else []) if pt.get('customdata')}
        if picked_ids:
            sdf = sdf[sdf['Garnet_ID'].isin(picked_ids)]
            st.success(f'{len(sdf):,} garnets selected on the map; charts below show only these.')

        chem_tab, table_tab, dl_tab = st.tabs(['Garnet chemistry', 'Data table', 'Download'])
        with chem_tab:
            t1, t2 = st.columns(2)
            with t1:
                fig_t = px.scatter_ternary(
                    sdf.dropna(subset=ends), a='Almandine_%S', b='Spessartine_%S', c='Pyrope_%S',
                    color='Formation_Name', hover_name='Garnet_ID', height=450,
                )
                st.plotly_chart(fig_t, width='stretch')
            with t2:
                num = [c for c in siw.columns if c.endswith('_apfu') or c.endswith('_%S') or c.endswith('_Swt%')]
                x = st.selectbox('X axis', num, index=num.index('Almandine_%S') if 'Almandine_%S' in num else 0, key='siw_x')
                y = st.selectbox('Y axis', num, index=num.index('Grossular_%S') if 'Grossular_%S' in num else 1, key='siw_y')
                fig_s = px.scatter(sdf, x=x, y=y, color='Formation_Name', hover_name='Garnet_ID', height=330)
                st.plotly_chart(fig_s, width='stretch')

        with table_tab:
            st.dataframe(sdf, width='stretch')

        with dl_tab:
            st.write('Download the full Himalayan detrital garnet dataset, or only the analyses currently selected by the filters and map.')
            d1, d2 = st.columns(2)
            d1.download_button(
                f'Download full dataset ({len(siw):,} analyses)',
                data=(Path(__file__).parent/'data/Siwalik.csv').read_bytes(),
                file_name='Siwalik.csv', mime='text/csv', key='siwalik_download',
            )
            d2.download_button(
                f'Download selection ({len(sdf):,} analyses)',
                data=sdf.to_csv(index=False).encode('utf-8'),
                file_name='Siwalik_filtered.csv', mime='text/csv', key='siwalik_filtered_download',
            )


with docs_tab:
    # Download data
    st.header('Download Data', divider='gray')

    st.markdown("""
    Download different versions of the Himalayan Garnet Analysis Database. Choose from the options below based on your needs.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Full Dataset")
        st.download_button(
            label="📥 Download Complete Dataset (CSV)",
            data=garnet_df.to_csv(index=False),
            file_name="himalayan_garnet_database_complete.csv",
            mime="text/csv",
            help="Download the entire dataset with all analyses",
            type="primary"
        )

        st.download_button(
            label="📥 Download Dataset Summary (CSV)",
            data=create_dataset_summary(garnet_df).to_csv(index=False),
            file_name="himalayan_garnet_database_summary.csv",
            mime="text/csv",
            help="This summary includes dataset-level counts, numeric composition and P–T statistics, and categorical breakdowns for country, lithology, location, cluster, and DOI.",
            type="primary"
        )

    with col2:
        st.subheader("🔍 Filtered Data")
        st.markdown("**Note:** This downloads data matching the shared sidebar filters and map selection.")
        st.download_button(
            label="📥 Download Filtered Dataset (CSV)",
            data=filtered_df.to_csv(index=False),
            file_name="himalayan_garnet_database_filtered.csv",
            mime="text/csv",
            help="Download the dataset filtered by your current selections",
            type="primary"
        )

        # Show current filter summary
        if len(filtered_df) < len(garnet_df):
            st.info(f"📋 Current filters applied: {len(filtered_df)} of {len(garnet_df)} analyses selected")
        else:
            st.info("📋 No filters applied - full dataset available")

    with st.expander('Citation'):
        st.markdown('Catlos, Elizabeth, 2026, [Replication Data for: Himalayan Garnet Analysis Database, Version 1.0](https://doi.org/10.18738/T8/GZLMUW), Texas Data Repository, V5, UNF:6:2lCNPA4JWHY5wpBttC3yMA== [fileUNF].')

    st.subheader("📋 Data Dictionary")
    st.markdown("""
    **Column Descriptions:**

    **Major Heading** | **Database Fields** | **Meaning / Purpose**
    --- | --- | ---
    Sample and garnet identification | Garnet_ID; Author_Sample_ID | Unique identifiers linking individual garnet analyses to original samples and publications.
    Lithologic classification | Lithology_Category; Author_Lithology | Description of the host-rock lithology.
    Geographic context | Country; Location; Latitude; Longitude; Elevation (meters) | Spatial metadata situating garnet sources within the Himalayan orogen.
    Metamorphic conditions | Peak_Pressure_kbar; Peak_Temperature_C | Constraints on peak metamorphic conditions associated with garnet growth.
    Garnet oxides (wt%) | SiO2_wt%; Al2O3_wt%; FeO_wt%; MnO_wt%; MgO_wt%; CaO_wt% | Major-element garnet compositions.
    Reference | DOI | Persistent identifier linking each dataset to its original publication.
    Normalized major element chemistry | SiO2_wt%_N; Al2O3_wt%_N; FeO_wt%_N; MnO_wt%_N; MgO_wt%_N; CaO_wt%_N | Major-element compositions recalculated to a common basis to enable comparison across datasets with differing analytical totals.
    Structural formula | Si_apfu; Al_apfu; Fe_apfu; Mn_apfu; Mg_apfu; Ca_apfu | Atoms per formula unit (apfu) calculated from normalized compositions.
    Garnet end-members | Almandine_%; Spessartine_%; Pyrope_%; Grossular_% | Relative proportions of major garnet end-members used for provenance discrimination.
    Statistics | Cluster | Cluster assignment derived from multivariate compositional analysis, used to identify compositional populations.

    **Lithology_Category descriptions:**

    **Lithology_Category** | **Description** | **Examples**
    --- | --- | ---
    Pelitic schist (Ky/Sil/St variants) | Medium- to high-grade metapelites dominated by mica ± garnet ± aluminosilicates (kyanite, sillimanite, staurolite) | Grt-schist, Grt-Bt schist, Grt-Ky schist, Grt-St-Bt schist
    Pelitic gneiss / paragneiss | Banded to foliated high-grade metasedimentary rocks; commonly interlayered with schist and gneiss | Grt-paragneiss, Grt-Sil paragneiss, Grt-schist/gneiss
    Migmatite (pelitic to gneissic) | Partially melted garnet-bearing rocks showing mixed melt and residuum textures | Grt-migmatite, Grt-migmatite gneiss, Grt-Bt pelitic migmatite gneiss
    High-pressure mafic rocks (eclogite / blueschist) | High-pressure to ultra-high-pressure metamorphosed basalts and gabbros | Grt-eclogite, Grt-UHP eclogite, Grt-blueschist
    Mafic metamorphic rocks (amphibolite / metabasite) | Metamorphosed basaltic rocks at greenschist to granulite facies | Grt-amphibolite, Grt-metabasite, Grt-Hbl gneiss
    Skarn (calc-silicate metasomatic) | Contact or hydrothermal metasomatic rocks rich in garnet ± pyroxene ± epidote ± ore minerals | Grt-skarn, Grt-Wol skarn, Grt-Mal skarn
    Granite / leucogranite (2-mica/peraluminous) | Felsic intrusive or anatectic rocks, commonly peraluminous and muscovite- or tourmaline-bearing. | Grt-leucogranite, Grt-2Mca granite, Grt-Tur-granite
    Quartzite / metasandstone / metapsammite | Siliceous metasedimentary rocks; commonly garnet-bearing quartzites | Grt-quartzite, Grt-Ms metapsammite
    Vein / pegmatite / aplite | Late-stage felsic veins or pegmatitic intrusions | Grt-pegmatite, Grt-aplite, Grt-Qz-Mo-Ccp vein
    Pelitic granulite | High-temperature metapelites containing garnet ± kyanite ± sillimanite | Grt-pelitic granulite, Grt-Ky granulite
    Calc-silicate / marble / carbonate | Carbonate- and silicate-bearing metamorphic rocks of calcareous protoliths | Grt-calc-silicate, Grt-marble, Grt-calc-silicate granulite
    Graphitic / tourmaline-bearing metasediments | Metasedimentary rocks enriched in graphite or tourmaline | Grt-graphitic schist, Grt-tourmalinite
    Specialized or unique lithologies | Uncommon or mixed lithologies not included above | Grt-gondite, Grt-charnockite, Grt-schist/gneiss/quartzite
    Orthogneiss / granitoid gneiss | Felsic to intermediate gneisses of igneous or mixed origin, typically banded and garnet-bearing | Grt-gneiss, Grt-granitic gneiss, Grt-augen gneiss
    """)

    # Community use and impact section
    with st.expander('Community Use & Impact', expanded=False):
        st.markdown(
            """
            If this tool supported your work, please cite it and let us know using the form below.
            """
        )

        with st.form('user_registry_form'):
            col1, col2 = st.columns(2)
            with col1:
                user_name = st.text_input('Name')
                institution = st.text_input('Institution')
                country = st.text_input('Country')
            with col2:
                contact_info = st.text_input('Email or Website / ORCID')
                project_name = st.text_input('Project / Publication / Course Name')
                citation_link = st.text_input('Optional citation / DOI / project page')

            use_case = st.text_area('How did you use this site?', height=140)
            permission_to_list = st.checkbox('You may list my name/project on this site.', value=False)

            submitted = st.form_submit_button('Tell us how you used this site')

            if submitted:
                entry = {
                    'timestamp': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                    'name': user_name,
                    'institution': institution,
                    'country': country,
                    'contact': contact_info,
                    'use_case': use_case,
                    'project': project_name,
                    'citation': citation_link,
                    'permission_to_list': 'Yes' if permission_to_list else 'No'
                }
                append_user_registry_entry(entry)
                st.success('Thank you! Your entry has been recorded.')
                if permission_to_list:
                    st.info('With permission, your entry may be shown below; otherwise it is stored privately.')
                else:
                    st.info('Your entry is saved for impact tracking only.')

        # Approved community entries
        approved_entries = get_user_registry_entries()
        approved_entries = approved_entries[approved_entries['permission_to_list'] == 'Yes']
        if not approved_entries.empty:
            st.subheader('Community impact examples')
            display_entries = approved_entries[['name', 'institution', 'country', 'project', 'citation', 'use_case']].copy()
            display_entries = display_entries.rename(columns={
                'name': 'Name',
                'institution': 'Institution',
                'country': 'Country',
                'project': 'Project / Course',
                'citation': 'Citation / Link',
                'use_case': 'Use case'
            })
            st.dataframe(display_entries.head(20), width='stretch')
