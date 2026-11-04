"""Validation and staging helpers for contributed garnet analyses.

Nothing here touches the public dataset. Accepted files are written to a
separate GitHub staging repository for manual review.
"""

import base64
import difflib
import io
import json
import re
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_ROWS = 5000

OXIDES = ['SiO2_wt%', 'Al2O3_wt%', 'FeO_wt%', 'MnO_wt%', 'MgO_wt%', 'CaO_wt%']
TEMPLATE_COLUMNS = [
    'Author_Sample_ID', 'Lithology_Category', 'Author_Lithology', 'Country', 'Location',
    'Latitude', 'Longitude', 'Elevation_m', 'Peak_Pressure_kbar', 'Peak_Temperature_C',
] + OXIDES + ['DOI']
REQUIRED_COLUMNS = [
    'Author_Sample_ID', 'Lithology_Category', 'Country', 'Location', 'Latitude', 'Longitude',
] + OXIDES
# Common wrong-unit or wrong-name headers mapped to the expected column.
UNIT_HINTS = {
    'peak_pressure_gpa': 'Peak_Pressure_kbar', 'peak_pressure_mpa': 'Peak_Pressure_kbar',
    'peak_pressure_kb': 'Peak_Pressure_kbar', 'pressure': 'Peak_Pressure_kbar',
    'peak_temperature_k': 'Peak_Temperature_C', 'peak_temperature_f': 'Peak_Temperature_C',
    'temperature': 'Peak_Temperature_C', 'temp_c': 'Peak_Temperature_C',
    'lat': 'Latitude', 'lon': 'Longitude', 'long': 'Longitude', 'elevation': 'Elevation_m',
    'elevation_ft': 'Elevation_m', 'elevation_km': 'Elevation_m',
    **{ox.split('_')[0].lower(): ox for ox in OXIDES},
    **{ox.split('_')[0].lower() + '_mol%': ox for ox in OXIDES},
}

# Molar masses (g/mol) used for the 12-oxygen formula check.
MOLAR_MASS = {'SiO2_wt%': 60.08, 'Al2O3_wt%': 101.96, 'FeO_wt%': 71.84,
              'MnO_wt%': 70.94, 'MgO_wt%': 40.30, 'CaO_wt%': 56.08}
CATIONS_PER_OXIDE = {'SiO2_wt%': 1, 'Al2O3_wt%': 2, 'FeO_wt%': 1,
                     'MnO_wt%': 1, 'MgO_wt%': 1, 'CaO_wt%': 1}
OXYGENS_PER_OXIDE = {'SiO2_wt%': 2, 'Al2O3_wt%': 3, 'FeO_wt%': 1,
                     'MnO_wt%': 1, 'MgO_wt%': 1, 'CaO_wt%': 1}

TOTAL_RANGE = (97.0, 102.0)
SI_TOLERANCE = 0.1
CATION_SUM_RANGE = (7.9, 8.1)
LAT_RANGE = (22.0, 40.0)
LON_RANGE = (70.0, 100.0)

DOI_RE = re.compile(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?(10\.\d{4,9}/\S+)$', re.IGNORECASE)
EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
URL_RE = re.compile(r'^https?://[^\s/]+\.[^\s/]+\S*$', re.IGNORECASE)


def template_csv(example_row=True):
    row = {
        'Author_Sample_ID': 'EXAMPLE-1', 'Lithology_Category': 'Pelitic schist (Ky/Sil/St variants)',
        'Author_Lithology': 'Gt-Bt schist', 'Country': 'Nepal', 'Location': 'Example location',
        'Latitude': 27.8, 'Longitude': 86.7, 'Elevation_m': 3000, 'Peak_Pressure_kbar': 8,
        'Peak_Temperature_C': 650, 'SiO2_wt%': 37.2, 'Al2O3_wt%': 20.9, 'FeO_wt%': 33.0,
        'MnO_wt%': 3.5, 'MgO_wt%': 2.6, 'CaO_wt%': 2.5, 'DOI': '10.xxxx/replace-me',
    }
    df = pd.DataFrame([row] if example_row else [], columns=TEMPLATE_COLUMNS)
    return df.to_csv(index=False)


def normalize_doi(value):
    match = DOI_RE.match((value or '').strip())
    return match.group(1) if match else None


def validate_metadata(name, email, institution, doi, repo_link):
    """Return a list of error strings for the contributor/provenance form."""
    errors = []
    if not (name or '').strip():
        errors.append('Contributor name is required.')
    if not (institution or '').strip():
        errors.append('Institution is required.')
    if not EMAIL_RE.match((email or '').strip()):
        errors.append('A valid contact email is required.')
    doi, repo_link = (doi or '').strip(), (repo_link or '').strip()
    if doi and not normalize_doi(doi):
        errors.append('The DOI is not valid. Use a form like 10.1234/abcd or https://doi.org/10.1234/abcd.')
    if repo_link and not URL_RE.match(repo_link):
        errors.append('The repository link must be a full http(s) URL.')
    if not doi and not repo_link:
        errors.append('Provide a DOI for published data, or a repository link for unpublished data.')
    return errors


class Report:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, check, message):
        self.errors.append({'Check': check, 'Problem': message})

    def warn(self, check, message):
        self.warnings.append({'Check': check, 'Problem': message})

    @property
    def ok(self):
        return not self.errors


def _rows(index, limit=8):
    """Format a row index as 1-based spreadsheet rows (header is row 1)."""
    rows = [str(i + 2) for i in index[:limit]]
    more = f' and {len(index) - limit} more' if len(index) > limit else ''
    return 'row ' + ', '.join(rows) + more if len(index) == 1 else 'rows ' + ', '.join(rows) + more


def read_upload(raw_bytes):
    """Parse uploaded bytes into a DataFrame. Returns (df, error_message)."""
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        return None, f'File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.'
    if b'\x00' in raw_bytes[:8192]:
        return None, 'Binary data detected. Please upload a text CSV file.'
    for encoding in ('utf-8-sig', 'latin1'):
        try:
            text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except Exception as exc:
        return None, f'Could not parse the file as CSV: {exc}'
    df.columns = df.columns.str.strip()
    if df.empty:
        return None, 'The file has no data rows.'
    if len(df) > MAX_ROWS:
        return None, f'The file has {len(df):,} rows; the limit is {MAX_ROWS:,} per submission.'
    return df, None


def _numeric(df, column):
    """Blank -> NaN, text -> NaN (flagged separately by the caller)."""
    cleaned = df[column].astype(str).str.strip().replace('', np.nan)
    return pd.to_numeric(cleaned, errors='coerce'), cleaned


def validate_dataframe(df, existing_df, lithology_categories):
    """Run all automatic checks. Returns (Report, cleaned numeric DataFrame)."""
    report = Report()

    # Required columns and units
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    lower_to_expected = {c.lower(): c for c in TEMPLATE_COLUMNS}
    for column in [c for c in TEMPLATE_COLUMNS if c not in df.columns]:
        hint = next((h for h in df.columns
                     if UNIT_HINTS.get(h.lower().replace(' ', '_')) == column
                     or lower_to_expected.get(h.lower()) == column), None)
        if hint:
            report.error('Required columns & units',
                         f'Column "{hint}" should be named "{column}" with values in the template units.')
        elif column in REQUIRED_COLUMNS:
            report.error('Required columns & units', f'Missing required column "{column}".')
    unknown = [c for c in df.columns if c not in TEMPLATE_COLUMNS]
    if unknown:
        report.warn('Required columns & units',
                    'Ignored columns not in the template: ' + ', '.join(f'"{c}"' for c in unknown[:10]) +
                    '. Normalized values, apfu, end-members and clusters are calculated by the curators.')
    if missing:
        return report, None

    data = df.copy()
    for column in ['Latitude', 'Longitude', 'Elevation_m', 'Peak_Pressure_kbar', 'Peak_Temperature_C'] + OXIDES:
        if column not in data.columns:
            data[column] = np.nan
            continue
        numeric, cleaned = _numeric(df, column)
        bad = cleaned.notna() & numeric.isna()
        if bad.any():
            report.error('Required columns & units',
                         f'"{column}" has non-numeric values in {_rows(data.index[bad])} '
                         '(use plain numbers; no "<", "n.d." or units in cells).')
        data[column] = numeric
    for column in ['Author_Sample_ID', 'Lithology_Category', 'Country', 'Location']:
        blank = data[column].astype(str).str.strip() == ''
        if blank.any():
            report.error('Required columns & units', f'"{column}" is blank in {_rows(data.index[blank])}.')

    # Blanks vs zeros
    ox = data[OXIDES]
    blank_any = ox.isna().any(axis=1)
    if blank_any.any():
        report.error('Blanks vs zeros',
                     f'Blank oxide values in {_rows(data.index[blank_any])}. Enter 0 for below detection; '
                     'leave blank only for rows you remove.')
    for column in ['SiO2_wt%', 'FeO_wt%']:
        zero = data[column] == 0
        if zero.any():
            report.error('Blanks vs zeros', f'{column} is 0 in {_rows(data.index[zero])}; garnet cannot have none.')
    for column in ['Al2O3_wt%', 'MnO_wt%', 'MgO_wt%', 'CaO_wt%']:
        zero = data[column] == 0
        if zero.any():
            report.warn('Blanks vs zeros',
                        f'{column} is 0 in {_rows(data.index[zero])}. Confirm these are below detection, not unmeasured.')
    negative = (ox < 0).any(axis=1)
    if negative.any():
        report.error('Blanks vs zeros', f'Negative oxide values in {_rows(data.index[negative])}.')
    for column in ['Peak_Pressure_kbar', 'Peak_Temperature_C']:
        zero = data[column] == 0
        if zero.any():
            report.warn('Blanks vs zeros', f'{column} is 0 in {_rows(data.index[zero])}; leave blank if unknown.')

    # Totals
    complete = ox.notna().all(axis=1)
    total = ox.sum(axis=1, min_count=6)
    off = complete & ((total < TOTAL_RANGE[0]) | (total > TOTAL_RANGE[1]))
    if off.any():
        report.error('Oxide totals',
                     f'Totals outside {TOTAL_RANGE[0]:g}-{TOTAL_RANGE[1]:g} wt% in {_rows(data.index[off])} '
                     f'(e.g. {total[off].iloc[0]:.1f}). Check for mol% or fractions instead of wt%, or missing oxides.')

    # Stoichiometry on 12 oxygens
    moles = ox.div(pd.Series(MOLAR_MASS))
    oxygen = sum(moles[c] * OXYGENS_PER_OXIDE[c] for c in OXIDES)
    cations = moles.mul(pd.Series(CATIONS_PER_OXIDE)).mul(12).div(oxygen.replace(0, np.nan), axis=0)
    stoich_rows = complete & oxygen.gt(0)
    si_off = stoich_rows & ((cations['SiO2_wt%'] - 3).abs() > SI_TOLERANCE)
    sum_off = stoich_rows & ((cations.sum(axis=1) < CATION_SUM_RANGE[0]) | (cations.sum(axis=1) > CATION_SUM_RANGE[1]))
    if si_off.any():
        report.error('Stoichiometry (12 O)',
                     f'Si is not about 3 apfu in {_rows(data.index[si_off])} '
                     f'(e.g. {cations.loc[si_off, "SiO2_wt%"].iloc[0]:.2f}). These may not be garnet analyses.')
    if sum_off.any():
        report.error('Stoichiometry (12 O)',
                     f'Total cations are not about 8 apfu in {_rows(data.index[sum_off])} '
                     f'(e.g. {cations.sum(axis=1)[sum_off].iloc[0]:.2f}). These may not be garnet analyses.')

    # Coordinates
    lat, lon = data['Latitude'], data['Longitude']
    blank_xy = lat.isna() | lon.isna()
    if blank_xy.any():
        report.error('Coordinates', f'Latitude/Longitude missing or invalid in {_rows(data.index[blank_xy])}.')
    in_lat = lat.between(*LAT_RANGE)
    in_lon = lon.between(*LON_RANGE)
    swapped = ~blank_xy & ~(in_lat & in_lon) & lon.between(*LAT_RANGE) & lat.between(*LON_RANGE)
    if swapped.any():
        report.error('Coordinates', f'Latitude and longitude look swapped in {_rows(data.index[swapped])}.')
    outside = ~blank_xy & ~swapped & ~(in_lat & in_lon)
    if outside.any():
        report.error('Coordinates',
                     f'Outside the Himalayan region (lat {LAT_RANGE[0]:g}-{LAT_RANGE[1]:g}, '
                     f'lon {LON_RANGE[0]:g}-{LON_RANGE[1]:g}) in {_rows(data.index[outside])}. '
                     'Use decimal degrees (west/south negative).')

    # P and T units
    pressure, temperature = data['Peak_Pressure_kbar'], data['Peak_Temperature_C']
    if (pressure > 100).any():
        report.error('P in kbar, T in °C',
                     f'Pressure above 100 kbar in {_rows(data.index[pressure > 100])}; values look like MPa or bar.')
    elif (pressure > 45).any():
        report.warn('P in kbar, T in °C',
                    f'Pressure above 45 kbar in {_rows(data.index[pressure > 45])}; confirm the unit is kbar.')
    if pressure.notna().any() and pressure.max() <= 4 and pressure.median() < 2.5:
        report.warn('P in kbar, T in °C', 'All pressures are below 4; they may be in GPa (multiply by 10 for kbar).')
    if (temperature > 1500).any():
        report.error('P in kbar, T in °C',
                     f'Temperature above 1500 in {_rows(data.index[temperature > 1500])}; values look like Kelvin or °F.')
    elif ((temperature > 1100) | (temperature < 200)).any():
        bad = (temperature > 1100) | (temperature < 200)
        report.warn('P in kbar, T in °C', f'Temperature outside 200-1100 °C in {_rows(data.index[bad])}; confirm °C.')

    # Lithology mapping
    allowed = {c.lower(): c for c in lithology_categories}
    lith = data['Lithology_Category'].astype(str).str.strip()
    unmapped = ~lith.str.lower().isin(allowed) & (lith != '')
    if unmapped.any():
        examples = []
        for value in lith[unmapped].unique()[:3]:
            close = difflib.get_close_matches(value.lower(), list(allowed), n=1, cutoff=0.5)
            examples.append(f'"{value}"' + (f' (did you mean "{allowed[close[0]]}"?)' if close else ''))
        report.error('Lithology categories',
                     f'{unmapped.sum()} row(s) are not one of the {len(allowed)} categories, e.g. ' +
                     '; '.join(examples) + '. Put your own term in Author_Lithology.')
    data['Lithology_Category'] = lith.str.lower().map(allowed).fillna(lith)

    # Duplicates
    dup_in_file = data.duplicated(subset=OXIDES, keep='first') & complete
    if dup_in_file.any():
        report.warn('Duplicates', f'Identical oxide values repeated within the file in {_rows(data.index[dup_in_file])}.')
    if existing_df is not None and complete.any():
        existing_keys = set(map(tuple, existing_df[OXIDES].round(2).dropna().itertuples(index=False, name=None)))
        mine = [tuple(round(v, 2) for v in row) for row in data[OXIDES].itertuples(index=False, name=None)]
        dup_existing = pd.Series([complete.iloc[i] and k in existing_keys for i, k in enumerate(mine)],
                                 index=data.index)
        if dup_existing.any():
            report.error('Duplicates',
                         f'{dup_existing.sum()} analysis(es) already exist in the database (identical oxides) in '
                         f'{_rows(data.index[dup_existing])}. Remove them to submit only new data.')

    if 'DOI' not in data.columns or data['DOI'].astype(str).str.strip().eq('').any():
        report.warn('Reference', 'Some rows have no DOI; the submission-level DOI/link will be applied to them.')

    return report, data


def staging_configured(secrets):
    try:
        return bool(secrets['STAGING_GITHUB_TOKEN'] and secrets['STAGING_GITHUB_REPO'])
    except Exception:
        return False


def _gh(method, url, token, **kwargs):
    response = requests.request(
        method, url, timeout=20,
        headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json',
                 'X-GitHub-Api-Version': '2022-11-28'}, **kwargs)
    response.raise_for_status()
    return response.json()


def stage_submission(csv_bytes, metadata, token, repo):
    """Commit the CSV and metadata to submissions/<id>/ in the staging repo and open an
    issue, which emails the repository watchers. Returns (submission_id, issue_url)."""
    stamp = datetime.now(timezone.utc)
    submission_id = f"{stamp.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    base = f'https://api.github.com/repos/{repo}'
    metadata = {**metadata, 'submission_id': submission_id, 'submitted_utc': stamp.isoformat(timespec='seconds')}
    for name, payload in (('data.csv', csv_bytes),
                          ('metadata.json', json.dumps(metadata, indent=2).encode())):
        _gh('PUT', f'{base}/contents/submissions/{submission_id}/{name}', token, json={
            'message': f'Stage submission {submission_id}: {name}',
            'content': base64.b64encode(payload).decode(),
        })
    body = (f"New data submission `{submission_id}`\n\n"
            f"- Contributor: {metadata['name']} ({metadata['institution']})\n"
            f"- Contact: {metadata['email']}\n"
            f"- DOI: {metadata.get('doi') or 'n/a'}\n"
            f"- Repository link: {metadata.get('repository_link') or 'n/a'}\n"
            f"- Rows: {metadata['rows']}\n"
            f"- Validation warnings acknowledged: {metadata['warnings']}\n\n"
            f"Files: `submissions/{submission_id}/`")
    issue = _gh('POST', f'{base}/issues', token, json={
        'title': f'Data submission {submission_id} ({metadata["rows"]} analyses)', 'body': body})
    return submission_id, issue['html_url']
