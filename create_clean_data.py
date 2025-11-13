"""
Clean Data Creation Script
Creates one master CSV with all merged data and separate CSVs for missing movies
"""

import pandas as pd
import numpy as np
import re
from unidecode import unidecode

print("=" * 80)
print("CREATING CLEAN MERGED DATA")
print("=" * 80)

# ============================================================================
# STEP 1: Load all datasets
# ============================================================================
print("\n[1] Loading datasets...")
wiki_df = pd.read_csv('clean_wiki_data.csv')
imdb_df = pd.read_csv('imdb_results.csv')
letterboxd_df = pd.read_csv('letterboxd_results.csv')
rt_df = pd.read_csv('rottentomatoes_data_incomplete.csv')

print(f"  ✓ Wikipedia: {len(wiki_df)} films (source of truth)")
print(f"  ✓ IMDB: {len(imdb_df)} records")
print(f"  ✓ Letterboxd: {len(letterboxd_df)} films")
print(f"  ✓ Rotten Tomatoes: {len(rt_df)} records")

# ============================================================================
# STEP 2: Clean Rotten Tomatoes
# ============================================================================
print("\n[2] Cleaning Rotten Tomatoes data...")
rt_df_clean = rt_df[
    (~rt_df['tomatometer_score'].astype(str).str.contains('Tomatometer', na=False)) &
    (~rt_df['popcornmeter_score'].astype(str).str.contains('Popcornmeter', na=False))
].copy()

rt_df_clean['tomatometer_score'] = pd.to_numeric(rt_df_clean['tomatometer_score'].astype(str).str.rstrip('%'), errors='coerce')
rt_df_clean['popcornmeter_score'] = pd.to_numeric(rt_df_clean['popcornmeter_score'].astype(str).str.rstrip('%'), errors='coerce')

def clean_rt_name(rt_name):
    if pd.isna(rt_name):
        return None
    return rt_name.replace('_', '-').lower()

def extract_name_without_year(name):
    if pd.isna(name):
        return None
    return re.sub(r'-\d{4}$', '', name)

rt_df_clean['film_key'] = rt_df_clean['film_name'].apply(clean_rt_name)
rt_df_clean['name_without_year'] = rt_df_clean['film_key'].apply(extract_name_without_year)

wiki_df['name_without_year'] = wiki_df['clean_name'].apply(extract_name_without_year)

# Match RT to Wiki
rt_to_wiki = {}
wiki_clean_names = set(wiki_df['clean_name'])

# Direct match
for idx, row in rt_df_clean.iterrows():
    if row['film_key'] in wiki_clean_names:
        rt_to_wiki[row['film_name']] = row['film_key']

# Name without year match
for idx, row in rt_df_clean.iterrows():
    if row['film_name'] not in rt_to_wiki:
        matches = wiki_df[wiki_df['name_without_year'] == row['name_without_year']]
        if len(matches) > 0:
            rt_to_wiki[row['film_name']] = matches.iloc[0]['clean_name']

rt_df_clean['film_key'] = rt_df_clean['film_name'].map(rt_to_wiki).fillna(rt_df_clean['film_key'])
rt_matched = rt_df_clean[rt_df_clean['film_key'].isin(wiki_clean_names)]
print(f"  ✓ Matched {len(rt_matched)} / {len(rt_df_clean)} records")

# ============================================================================
# STEP 3: Clean IMDB
# ============================================================================
print("\n[3] Cleaning IMDB data...")

def extract_film_key_from_url(url):
    if pd.isna(url) or url == '':
        return None
    match = re.search(r'letterboxd\.com/film/([^/?]+)', str(url))
    if match:
        return match.group(1)
    return None

def create_clean_name(film_name, year):
    if pd.isna(film_name) or film_name == '':
        return None
    name = unidecode(str(film_name))
    name = name.lower()
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', '-', name)
    name = re.sub(r'-+', '-', name)
    name = name.strip('-')
    if pd.notna(year) and year != '':
        try:
            year_int = int(float(year))
            return f"{name}-{year_int}"
        except:
            pass
    return name

def create_name_without_year(film_name):
    if pd.isna(film_name) or film_name == '':
        return None
    name = unidecode(str(film_name))
    name = name.lower()
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', '-', name)
    name = re.sub(r'-+', '-', name)
    name = name.strip('-')
    return name

# Extract from url_tried (contains year!)
imdb_df['film_key_from_url'] = imdb_df['url_tried'].apply(extract_film_key_from_url)
imdb_df['film_key_from_name'] = imdb_df.apply(lambda row: create_clean_name(row['Film'], row['year']), axis=1)
imdb_df['name_only'] = imdb_df['Film'].apply(create_name_without_year)

# Match IMDB to Wiki
name_to_wiki = {}
for idx, imdb_row in imdb_df.iterrows():
    name_only = imdb_row['name_only']
    if pd.isna(name_only):
        continue

    wiki_matches = wiki_df[wiki_df['name_without_year'] == name_only]
    if len(wiki_matches) > 0:
        if pd.notna(imdb_row['year']) and imdb_row['year'] != '':
            try:
                imdb_year = int(float(imdb_row['year']))
                wiki_years = wiki_matches['Year of Film Release'].str.extract(r'(\d{4})')[0].astype(float)
                year_diffs = (wiki_years - imdb_year).abs()
                best_idx = year_diffs.idxmin()
                name_to_wiki[imdb_row['Film']] = wiki_matches.loc[best_idx, 'clean_name']
            except:
                name_to_wiki[imdb_row['Film']] = wiki_matches.iloc[0]['clean_name']
        else:
            name_to_wiki[imdb_row['Film']] = wiki_matches.iloc[0]['clean_name']

# Assign film_key with priority
imdb_df['film_key'] = None
mask_url = (imdb_df['film_key_from_url'].notna()) & (imdb_df['film_key_from_url'].isin(wiki_clean_names))
imdb_df.loc[mask_url, 'film_key'] = imdb_df.loc[mask_url, 'film_key_from_url']

mask_name_year = (imdb_df['film_key'].isna()) & (imdb_df['film_key_from_name'].isin(wiki_clean_names))
imdb_df.loc[mask_name_year, 'film_key'] = imdb_df.loc[mask_name_year, 'film_key_from_name']

for film_name, wiki_name in name_to_wiki.items():
    mask = (imdb_df['Film'] == film_name) & (imdb_df['film_key'].isna())
    imdb_df.loc[mask, 'film_key'] = wiki_name

imdb_matched = imdb_df[imdb_df['film_key'].isin(wiki_clean_names)]
print(f"  ✓ Matched {len(imdb_matched)} / {len(imdb_df)} records")

# ============================================================================
# STEP 4: Create Master Merged CSV
# ============================================================================
print("\n[4] Creating master merged dataset...")

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

merged.to_csv('clean_merged_data.csv', index=False)
print(f"  ✓ Saved clean_merged_data.csv ({len(merged)} films × {len(merged.columns)} columns)")

# ============================================================================
# STEP 5: Create Missing Movies CSVs
# ============================================================================
print("\n[5] Creating missing movies CSVs...")

# Get matched film sets
wiki_films = set(wiki_df['clean_name'])
letterboxd_films = set(letterboxd_df['film_name'])
rt_films = set(rt_matched['film_key'])
imdb_films = set(imdb_matched['film_key'])

# Missing from Letterboxd
missing_letterboxd = wiki_films - letterboxd_films
missing_letterboxd_df = wiki_df[wiki_df['clean_name'].isin(missing_letterboxd)][['Film', 'Year of Film Release', 'clean_name', 'is_winner']]
missing_letterboxd_df.to_csv('missing_from_letterboxd.csv', index=False)
print(f"  ✓ missing_from_letterboxd.csv ({len(missing_letterboxd_df)} films)")

# Missing from Rotten Tomatoes
missing_rt = wiki_films - rt_films
missing_rt_df = wiki_df[wiki_df['clean_name'].isin(missing_rt)][['Film', 'Year of Film Release', 'clean_name', 'is_winner']]
missing_rt_df.to_csv('missing_from_rottentomatoes.csv', index=False)
print(f"  ✓ missing_from_rottentomatoes.csv ({len(missing_rt_df)} films)")

# Missing from IMDB
missing_imdb = wiki_films - imdb_films
missing_imdb_df = wiki_df[wiki_df['clean_name'].isin(missing_imdb)][['Film', 'Year of Film Release', 'clean_name', 'is_winner']]
missing_imdb_df.to_csv('missing_from_imdb.csv', index=False)
print(f"  ✓ missing_from_imdb.csv ({len(missing_imdb_df)} films)")

# ============================================================================
# STEP 6: Summary Report
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nMaster Dataset:")
print(f"  - clean_merged_data.csv: {len(merged)} films")
print(f"  - Data completeness:")
print(f"    * Letterboxd: {(merged['rating'].notna().sum() / len(merged) * 100):.1f}%")
print(f"    * IMDB: {(merged['imdb_rating'].notna().sum() / len(merged) * 100):.1f}%")
print(f"    * Rotten Tomatoes: {(merged['tomatometer_score'].notna().sum() / len(merged) * 100):.1f}%")

print(f"\nMissing Movies CSVs:")
print(f"  - missing_from_letterboxd.csv: {len(missing_letterboxd_df)} films")
print(f"  - missing_from_rottentomatoes.csv: {len(missing_rt_df)} films")
print(f"  - missing_from_imdb.csv: {len(missing_imdb_df)} films")

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
