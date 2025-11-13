"""
Improved Data Cleaning Script with Intelligent Name Matching
This script implements clever strategies to match film names across datasets
"""

import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from unidecode import unidecode

print("=" * 80)
print("INTELLIGENT DATA CLEANING & MATCHING")
print("=" * 80)

# ============================================================================
# STEP 1: Load all datasets
# ============================================================================
print("\n[1] Loading datasets...")
wiki_df = pd.read_csv('clean_wiki_data.csv')
imdb_df = pd.read_csv('imdb_results.csv')
letterboxd_df = pd.read_csv('letterboxd_results.csv')
rt_df = pd.read_csv('rottentomatoes_data_incomplete.csv')

print(f"  - Wikipedia: {len(wiki_df)} films (source of truth)")
print(f"  - IMDB: {len(imdb_df)} rows")
print(f"  - Letterboxd: {len(letterboxd_df)} films")
print(f"  - Rotten Tomatoes: {len(rt_df)} rows")

# ============================================================================
# STEP 2: Clean Rotten Tomatoes Data
# ============================================================================
print("\n[2] Cleaning Rotten Tomatoes data...")

# Remove invalid rows
rt_df_clean = rt_df[
    (~rt_df['tomatometer_score'].astype(str).str.contains('Tomatometer', na=False)) &
    (~rt_df['popcornmeter_score'].astype(str).str.contains('Popcornmeter', na=False))
].copy()

# Convert scores to numeric
rt_df_clean['tomatometer_score'] = rt_df_clean['tomatometer_score'].astype(str).str.rstrip('%')
rt_df_clean['popcornmeter_score'] = rt_df_clean['popcornmeter_score'].astype(str).str.rstrip('%')
rt_df_clean['tomatometer_score'] = pd.to_numeric(rt_df_clean['tomatometer_score'], errors='coerce')
rt_df_clean['popcornmeter_score'] = pd.to_numeric(rt_df_clean['popcornmeter_score'], errors='coerce')

print(f"  - Removed {len(rt_df) - len(rt_df_clean)} invalid rows")
print(f"  - Valid RT records: {len(rt_df_clean)}")

# ============================================================================
# STEP 3: Smart Matching for Rotten Tomatoes
# ============================================================================
print("\n[3] Implementing smart matching for Rotten Tomatoes...")

def clean_rt_name(rt_name):
    """Convert RT name format to Wikipedia clean_name format"""
    if pd.isna(rt_name):
        return None
    # Replace underscores with hyphens
    cleaned = rt_name.replace('_', '-')
    return cleaned.lower()

rt_df_clean['film_key_cleaned'] = rt_df_clean['film_name'].apply(clean_rt_name)

# Strategy 1: Direct match
print("  - Strategy 1: Direct match after cleaning underscores...")
wiki_clean_names = set(wiki_df['clean_name'])
direct_matches = rt_df_clean[rt_df_clean['film_key_cleaned'].isin(wiki_clean_names)]
print(f"    ✓ Found {len(direct_matches)} direct matches")

# Strategy 2: Match by removing year suffix from RT, then trying all possible years from wiki
print("  - Strategy 2: Match by film name without year...")

def extract_name_without_year(name):
    """Remove year suffix from name (e.g., 'bad-girl-1931' -> 'bad-girl')"""
    if pd.isna(name):
        return None
    # Remove year pattern at the end (e.g., -1927, -2022)
    name_no_year = re.sub(r'-\d{4}$', '', name)
    return name_no_year

rt_df_clean['name_without_year'] = rt_df_clean['film_key_cleaned'].apply(extract_name_without_year)

# Create a mapping from wiki films: name_without_year -> full clean_name
wiki_df['name_without_year'] = wiki_df['clean_name'].apply(extract_name_without_year)

# For RT films without direct match, try matching by name_without_year
rt_unmatched = rt_df_clean[~rt_df_clean['film_key_cleaned'].isin(wiki_clean_names)].copy()

# Match by name_without_year - if there's only one film with that base name, use it
# If there are multiple (different years), we'll need additional logic
matched_by_name = []
for idx, row in rt_unmatched.iterrows():
    name_no_year = row['name_without_year']
    # Find all wiki films with this base name
    wiki_matches = wiki_df[wiki_df['name_without_year'] == name_no_year]
    if len(wiki_matches) == 1:
        # Exactly one match - use it!
        matched_by_name.append({
            'rt_name': row['film_name'],
            'rt_cleaned': row['film_key_cleaned'],
            'matched_wiki_name': wiki_matches.iloc[0]['clean_name'],
            'Film': wiki_matches.iloc[0]['Film']
        })
    elif len(wiki_matches) > 1:
        # Multiple films with same base name (e.g., different years)
        # Use the most common year (earliest one)
        matched_by_name.append({
            'rt_name': row['film_name'],
            'rt_cleaned': row['film_key_cleaned'],
            'matched_wiki_name': wiki_matches.iloc[0]['clean_name'],  # Take first (earliest)
            'Film': wiki_matches.iloc[0]['Film']
        })

matched_by_name_df = pd.DataFrame(matched_by_name)
print(f"    ✓ Found {len(matched_by_name_df)} matches by name without year")

# Create the final mapping for RT
rt_to_wiki_mapping = {}

# Add direct matches
for idx, row in direct_matches.iterrows():
    rt_to_wiki_mapping[row['film_name']] = row['film_key_cleaned']

# Add name-based matches
for idx, row in matched_by_name_df.iterrows():
    if row['rt_name'] not in rt_to_wiki_mapping:
        rt_to_wiki_mapping[row['rt_name']] = row['matched_wiki_name']

# Apply the mapping
rt_df_clean['film_key'] = rt_df_clean['film_name'].map(rt_to_wiki_mapping)
rt_df_clean['film_key'] = rt_df_clean['film_key'].fillna(rt_df_clean['film_key_cleaned'])

rt_matched = rt_df_clean[rt_df_clean['film_key'].isin(wiki_clean_names)]
print(f"  - Total RT matches: {len(rt_matched)} / {len(rt_df_clean)} ({len(rt_matched)/len(rt_df_clean)*100:.1f}%)")

# ============================================================================
# STEP 4: Smart Matching for IMDB
# ============================================================================
print("\n[4] Implementing smart matching for IMDB...")

def extract_film_key_from_url(url):
    """Extract clean film name from Letterboxd URL"""
    if pd.isna(url) or url == '':
        return None
    # URL format: https://letterboxd.com/film/film-name-year or https://letterboxd.com/film/film-name
    match = re.search(r'letterboxd\.com/film/([^/?]+)', str(url))
    if match:
        return match.group(1)
    return None

# Strategy 1: Extract from url_tried first (has year), then url_used
print("  - Strategy 1: Extract film key from Letterboxd URLs...")
imdb_df['film_key_from_url_tried'] = imdb_df['url_tried'].apply(extract_film_key_from_url)
imdb_df['film_key_from_url_used'] = imdb_df['url_used'].apply(extract_film_key_from_url)

# Prefer url_tried (has year), fallback to url_used
imdb_df['film_key_from_url'] = imdb_df['film_key_from_url_tried'].fillna(imdb_df['film_key_from_url_used'])

url_matches = imdb_df[imdb_df['film_key_from_url'].isin(wiki_clean_names)]
url_tried_matches = imdb_df[imdb_df['film_key_from_url_tried'].isin(wiki_clean_names)]
url_used_matches = imdb_df[(imdb_df['film_key_from_url_tried'].isna() | ~imdb_df['film_key_from_url_tried'].isin(wiki_clean_names)) &
                           imdb_df['film_key_from_url_used'].isin(wiki_clean_names)]
print(f"    ✓ From url_tried: {len(url_tried_matches)} matches")
print(f"    ✓ From url_used: {len(url_used_matches)} matches")
print(f"    ✓ Total URL matches: {len(url_matches)}")

# Strategy 2: Create clean name from Film + year
print("  - Strategy 2: Create clean name from Film name + year...")

def create_clean_name(film_name, year):
    """Create clean_name format from film name and year"""
    if pd.isna(film_name) or film_name == '':
        return None
    # Convert accented characters to ASCII equivalents
    name = unidecode(str(film_name))
    # Convert to lowercase
    name = name.lower()
    # Remove special characters and replace spaces with hyphens
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', '-', name)
    name = re.sub(r'-+', '-', name)  # Replace multiple hyphens with single
    name = name.strip('-')

    # Add year if available
    if pd.notna(year) and year != '':
        try:
            year_int = int(float(year))
            return f"{name}-{year_int}"
        except:
            pass
    return name

imdb_df['film_key_from_name'] = imdb_df.apply(
    lambda row: create_clean_name(row['Film'], row['year']), axis=1
)

# Try matching with year
name_year_matches = imdb_df[
    (imdb_df['film_key_from_url'].isna()) &
    (imdb_df['film_key_from_name'].isin(wiki_clean_names))
]
print(f"    ✓ Found {len(name_year_matches)} additional matches from name+year")

# Strategy 3: Match by name without year (MORE AGGRESSIVE)
print("  - Strategy 3: Match by name without year...")

# For ALL IMDB films (not just unmatched), create name_without_year from Film column directly
def create_name_without_year(film_name):
    """Create a clean name without year from the film title"""
    if pd.isna(film_name) or film_name == '':
        return None
    # Convert accented characters to ASCII equivalents
    name = unidecode(str(film_name))
    name = name.lower()
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', '-', name)
    name = re.sub(r'-+', '-', name)
    name = name.strip('-')
    return name

imdb_df['name_only'] = imdb_df['Film'].apply(create_name_without_year)

# Build a mapping from name_only -> best wiki match
name_to_wiki = {}
for idx, imdb_row in imdb_df.iterrows():
    name_only = imdb_row['name_only']
    if pd.isna(name_only):
        continue

    # Find all wiki films matching this base name
    wiki_matches = wiki_df[wiki_df['name_without_year'] == name_only]

    if len(wiki_matches) > 0:
        # If we have the year, find closest match
        if pd.notna(imdb_row['year']) and imdb_row['year'] != '':
            try:
                imdb_year = int(float(imdb_row['year']))
                # Extract years from wiki and find closest
                wiki_years = wiki_matches['Year of Film Release'].str.extract(r'(\d{4})')[0].astype(float)
                year_diffs = (wiki_years - imdb_year).abs()
                best_idx = year_diffs.idxmin()
                name_to_wiki[imdb_row['Film']] = wiki_matches.loc[best_idx, 'clean_name']
            except:
                # No valid year comparison, use first match
                name_to_wiki[imdb_row['Film']] = wiki_matches.iloc[0]['clean_name']
        else:
            # No year available, use first match
            name_to_wiki[imdb_row['Film']] = wiki_matches.iloc[0]['clean_name']

print(f"    ✓ Found {len(name_to_wiki)} potential matches by film name")

# Create final film_key for IMDB with priority order
imdb_df['film_key'] = None

# Priority 1: URL extraction (ONLY if it matches Wikipedia)
mask_url = (imdb_df['film_key_from_url'].notna()) & (imdb_df['film_key_from_url'].isin(wiki_clean_names))
imdb_df.loc[mask_url, 'film_key'] = imdb_df.loc[mask_url, 'film_key_from_url']

# Priority 2: Name+year exact match
mask_name_year = (imdb_df['film_key'].isna()) & (imdb_df['film_key_from_name'].isin(wiki_clean_names))
imdb_df.loc[mask_name_year, 'film_key'] = imdb_df.loc[mask_name_year, 'film_key_from_name']

# Priority 3: Name-based matching (year-agnostic)
for film_name, wiki_name in name_to_wiki.items():
    mask = (imdb_df['Film'] == film_name) & (imdb_df['film_key'].isna())
    imdb_df.loc[mask, 'film_key'] = wiki_name

# Count matches
imdb_matched = imdb_df[imdb_df['film_key'].isin(wiki_clean_names)]
url_matches_count = imdb_df[imdb_df['film_key_from_url'].notna() & imdb_df['film_key_from_url'].isin(wiki_clean_names)].shape[0]
name_year_matches_count = imdb_matched[imdb_matched['film_key'] == imdb_matched['film_key_from_name']].shape[0] - url_matches_count
name_only_matches_count = len(imdb_matched) - url_matches_count - name_year_matches_count

print(f"    ✓ URL matches: {url_matches_count}")
print(f"    ✓ Name+Year exact matches: {name_year_matches_count}")
print(f"    ✓ Name-only matches: {name_only_matches_count}")
print(f"  - Total IMDB matches: {len(imdb_matched)} / {len(imdb_df)} ({len(imdb_matched)/len(imdb_df)*100:.1f}%)")

# ============================================================================
# STEP 5: Create Match Report
# ============================================================================
print("\n[5] Creating match report...")

wiki_films = set(wiki_df['clean_name'])
imdb_matched_films = set(imdb_matched['film_key'].dropna())
rt_matched_films = set(rt_matched['film_key'].dropna())
letterboxd_films = set(letterboxd_df['film_name'].dropna())

match_report = f"""
{'=' * 80}
MATCHING RESULTS REPORT
{'=' * 80}

Source of Truth: Wikipedia ({len(wiki_films)} films)

IMDB Matching:
  - Matched: {len(imdb_matched_films)} films ({len(imdb_matched_films)/len(wiki_films)*100:.1f}%)
  - Unmatched Wikipedia films: {len(wiki_films - imdb_matched_films)}
  - IMDB total records: {len(imdb_df)}
  - Method breakdown:
    * From Letterboxd URL: {url_matches_count}
    * From name+year exact: {name_year_matches_count}
    * From name-only (year-agnostic): {name_only_matches_count}

Rotten Tomatoes Matching:
  - Matched: {len(rt_matched_films)} films ({len(rt_matched_films)/len(wiki_films)*100:.1f}%)
  - Unmatched Wikipedia films: {len(wiki_films - rt_matched_films)}
  - Method breakdown:
    * Direct match: {len(direct_matches)}
    * Name without year: {len(matched_by_name_df)}

Letterboxd Matching:
  - Matched: {len(letterboxd_films & wiki_films)} films ({len(letterboxd_films & wiki_films)/len(wiki_films)*100:.1f}%)
  - Unmatched Wikipedia films: {len(wiki_films - letterboxd_films)}

Films missing from all external sources:
"""

missing_from_all = wiki_films - imdb_matched_films - rt_matched_films - letterboxd_films
if len(missing_from_all) > 0:
    match_report += f"  - Count: {len(missing_from_all)}\n"
    for film in sorted(missing_from_all)[:20]:
        match_report += f"    * {film}\n"
    if len(missing_from_all) > 20:
        match_report += f"    ... and {len(missing_from_all) - 20} more\n"
else:
    match_report += "  - None! All films found in at least one source.\n"

match_report += "\n" + "=" * 80

print(match_report)

# Save the report
with open('insights/matching_report.txt', 'w') as f:
    f.write(match_report)

print("\n  - Match report saved to insights/matching_report.txt")

# ============================================================================
# STEP 6: Save cleaned datasets with film_key
# ============================================================================
print("\n[6] Saving cleaned datasets...")

# Save cleaned datasets
imdb_matched.to_csv('insights/imdb_matched.csv', index=False)
rt_matched.to_csv('insights/rt_matched.csv', index=False)

print(f"  - Saved insights/imdb_matched.csv ({len(imdb_matched)} records)")
print(f"  - Saved insights/rt_matched.csv ({len(rt_matched)} records)")

# ============================================================================
# STEP 7: Create master merged dataset
# ============================================================================
print("\n[7] Creating master merged dataset...")

# Start with Wikipedia as base
merged = wiki_df.copy()
merged['film_key'] = merged['clean_name']

# Merge Letterboxd
merged = merged.merge(
    letterboxd_df.rename(columns={'film_name': 'film_key'}),
    on='film_key',
    how='left',
    suffixes=('', '_letterboxd')
)

# Merge IMDB
imdb_cols = ['film_key', 'year', 'rating', 'metascore', 'user_reviews', 'critic_reviews',
             'keywords', 'budget', 'revenue', 'cast_number']
imdb_subset = imdb_matched[imdb_cols].copy()
imdb_subset.columns = ['film_key', 'imdb_year', 'imdb_rating', 'imdb_metascore',
                       'imdb_user_reviews', 'imdb_critic_reviews', 'imdb_keywords',
                       'imdb_budget', 'imdb_revenue', 'imdb_cast_number']

merged = merged.merge(imdb_subset, on='film_key', how='left')

# Merge Rotten Tomatoes
rt_subset = rt_matched[['film_key', 'tomatometer_score', 'popcornmeter_score']].copy()
merged = merged.merge(rt_subset, on='film_key', how='left')

print(f"  - Master dataset: {len(merged)} films × {len(merged.columns)} columns")

# Calculate completeness
completeness_stats = {
    'Wikipedia (base)': 100.0,
    'Letterboxd': (merged['rating'].notna().sum() / len(merged) * 100),
    'IMDB': (merged['imdb_rating'].notna().sum() / len(merged) * 100),
    'Rotten Tomatoes': (merged['tomatometer_score'].notna().sum() / len(merged) * 100),
}

print("\n  Data Completeness:")
for source, pct in completeness_stats.items():
    print(f"    - {source:.<30} {pct:>6.1f}%")

# Save master dataset
merged.to_csv('insights/master_dataset.csv', index=False)
print(f"\n  - Saved insights/master_dataset.csv")

print("\n" + "=" * 80)
print("DATA CLEANING COMPLETE!")
print("=" * 80)
print("\nImproved Results:")
print(f"  - IMDB coverage increased to {completeness_stats['IMDB']:.1f}%")
print(f"  - Rotten Tomatoes coverage increased to {completeness_stats['Rotten Tomatoes']:.1f}%")
print(f"  - Ready for enhanced EDA analysis!")
print("=" * 80)
