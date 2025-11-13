"""
Exploratory Data Analysis for Oscar Best Picture Nominees
This script performs comprehensive EDA including:
- Loading and cleaning data from multiple sources
- Identifying missing movies across datasets
- Generating insights and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create insights folder
insights_dir = Path('insights')
insights_dir.mkdir(exist_ok=True)

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS: Oscar Best Picture Nominees")
print("=" * 80)

# ============================================================================
# STEP 1: Load all datasets
# ============================================================================
print("\n[1] Loading datasets...")

wiki_df = pd.read_csv('clean_wiki_data.csv')
imdb_df = pd.read_csv('imdb_results.csv')
letterboxd_df = pd.read_csv('letterboxd_results.csv')
rt_df = pd.read_csv('rottentomatoes_data_incomplete.csv')

print(f"  - Wikipedia data: {len(wiki_df)} films")
print(f"  - IMDB data: {len(imdb_df)} films")
print(f"  - Letterboxd data: {len(letterboxd_df)} films")
print(f"  - Rotten Tomatoes data: {len(rt_df)} films")

# ============================================================================
# STEP 2: Data exploration and cleaning
# ============================================================================
print("\n[2] Exploring data structure...")

# Create a report file
with open('insights/data_exploration_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DATA EXPLORATION REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write("1. WIKIPEDIA DATA\n")
    f.write("-" * 40 + "\n")
    f.write(f"Shape: {wiki_df.shape}\n")
    f.write(f"Columns: {list(wiki_df.columns)}\n")
    f.write(f"Missing values:\n{wiki_df.isnull().sum()}\n")
    f.write(f"\nSample data:\n{wiki_df.head()}\n\n")

    f.write("2. IMDB DATA\n")
    f.write("-" * 40 + "\n")
    f.write(f"Shape: {imdb_df.shape}\n")
    f.write(f"Columns: {list(imdb_df.columns)}\n")
    f.write(f"Missing values:\n{imdb_df.isnull().sum()}\n\n")

    f.write("3. LETTERBOXD DATA\n")
    f.write("-" * 40 + "\n")
    f.write(f"Shape: {letterboxd_df.shape}\n")
    f.write(f"Columns: {list(letterboxd_df.columns)}\n")
    f.write(f"Missing values:\n{letterboxd_df.isnull().sum()}\n\n")

    f.write("4. ROTTEN TOMATOES DATA\n")
    f.write("-" * 40 + "\n")
    f.write(f"Shape: {rt_df.shape}\n")
    f.write(f"Columns: {list(rt_df.columns)}\n")
    f.write(f"Missing values:\n{rt_df.isnull().sum()}\n\n")

print("  - Data exploration report saved to insights/data_exploration_report.txt")

# ============================================================================
# STEP 3: Standardize film names for joining
# ============================================================================
print("\n[3] Standardizing film names for merging...")

# Clean Rotten Tomatoes scores (remove "Tomatometer" and "Popcornmeter" rows)
rt_df_clean = rt_df[
    (~rt_df['tomatometer_score'].astype(str).str.contains('Tomatometer', na=False)) &
    (~rt_df['popcornmeter_score'].astype(str).str.contains('Popcornmeter', na=False))
].copy()

# Convert RT scores to numeric (remove % signs)
rt_df_clean['tomatometer_score'] = rt_df_clean['tomatometer_score'].astype(str).str.rstrip('%')
rt_df_clean['popcornmeter_score'] = rt_df_clean['popcornmeter_score'].astype(str).str.rstrip('%')
rt_df_clean['tomatometer_score'] = pd.to_numeric(rt_df_clean['tomatometer_score'], errors='coerce')
rt_df_clean['popcornmeter_score'] = pd.to_numeric(rt_df_clean['popcornmeter_score'], errors='coerce')

# Standardize names across datasets
wiki_df['film_key'] = wiki_df['clean_name']
letterboxd_df['film_key'] = letterboxd_df['film_name']
rt_df_clean['film_key'] = rt_df_clean['film_name']

# For IMDB, we need to create a standardized key
# First, let's check if there's a Film column
if 'Film' in imdb_df.columns:
    imdb_df['film_key'] = imdb_df['Film'].str.lower().str.replace(' ', '-')
    # Try to match the pattern in clean_name (film-year)
    # This is approximate and may need manual adjustment

print(f"  - Cleaned datasets ready for merging")
print(f"  - Rotten Tomatoes: {len(rt_df_clean)} valid records (removed {len(rt_df) - len(rt_df_clean)} invalid)")

# ============================================================================
# STEP 4: Identify missing movies across datasets
# ============================================================================
print("\n[4] Analyzing missing movies across datasets...")

# Get unique film keys from each dataset
wiki_films = set(wiki_df['film_key'].dropna())
imdb_films = set(imdb_df['film_key'].dropna()) if 'film_key' in imdb_df.columns else set()
letterboxd_films = set(letterboxd_df['film_key'].dropna())
rt_films = set(rt_df_clean['film_key'].dropna())

# Find missing films
missing_analysis = {
    'In Wikipedia but not in IMDB': wiki_films - imdb_films,
    'In Wikipedia but not in Letterboxd': wiki_films - letterboxd_films,
    'In Wikipedia but not in Rotten Tomatoes': wiki_films - rt_films,
    'In IMDB but not in Wikipedia': imdb_films - wiki_films,
    'In Letterboxd but not in Wikipedia': letterboxd_films - wiki_films,
    'In Rotten Tomatoes but not in Wikipedia': rt_films - wiki_films,
}

# Save missing films analysis
with open('insights/missing_films_analysis.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("MISSING FILMS ANALYSIS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total unique films in each dataset:\n")
    f.write(f"  - Wikipedia: {len(wiki_films)}\n")
    f.write(f"  - IMDB: {len(imdb_films)}\n")
    f.write(f"  - Letterboxd: {len(letterboxd_films)}\n")
    f.write(f"  - Rotten Tomatoes: {len(rt_films)}\n\n")

    for category, films in missing_analysis.items():
        f.write(f"\n{category}: {len(films)} films\n")
        f.write("-" * 40 + "\n")
        for film in sorted(films)[:50]:  # Show first 50
            f.write(f"  - {film}\n")
        if len(films) > 50:
            f.write(f"  ... and {len(films) - 50} more\n")

print(f"  - Missing films analysis saved to insights/missing_films_analysis.txt")

# Create visualization of data coverage
fig, ax = plt.subplots(figsize=(10, 6))
coverage_data = {
    'Wikipedia': len(wiki_films),
    'IMDB': len(imdb_films),
    'Letterboxd': len(letterboxd_films),
    'Rotten Tomatoes': len(rt_films)
}
bars = ax.bar(coverage_data.keys(), coverage_data.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_ylabel('Number of Films', fontsize=12)
ax.set_title('Film Coverage Across Data Sources', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('insights/data_coverage.png', dpi=300, bbox_inches='tight')
plt.close()
print("  - Saved visualization: insights/data_coverage.png")

# ============================================================================
# STEP 5: Merge datasets
# ============================================================================
print("\n[5] Merging datasets...")

# Start with Wikipedia as the base (it has the complete list of nominees)
merged_df = wiki_df.copy()

# Merge with Letterboxd
merged_df = merged_df.merge(
    letterboxd_df,
    on='film_key',
    how='left',
    suffixes=('', '_letterboxd')
)

# Merge with Rotten Tomatoes
merged_df = merged_df.merge(
    rt_df_clean[['film_key', 'tomatometer_score', 'popcornmeter_score']],
    on='film_key',
    how='left'
)

# Merge with IMDB (if film_key exists)
if 'film_key' in imdb_df.columns:
    # Select relevant columns from IMDB
    imdb_cols = ['film_key', 'rating', 'metascore', 'user_reviews', 'critic_reviews']
    imdb_relevant = imdb_df[imdb_cols].copy()
    imdb_relevant.columns = ['film_key', 'imdb_rating', 'imdb_metascore', 'imdb_user_reviews', 'imdb_critic_reviews']

    merged_df = merged_df.merge(
        imdb_relevant,
        on='film_key',
        how='left'
    )

print(f"  - Merged dataset contains {len(merged_df)} records with {len(merged_df.columns)} columns")

# Save merged dataset
merged_df.to_csv('insights/merged_oscar_data.csv', index=False)
print(f"  - Saved merged dataset to insights/merged_oscar_data.csv")

# ============================================================================
# STEP 6: Data quality analysis
# ============================================================================
print("\n[6] Analyzing data quality...")

# Calculate completeness for each data source
completeness = pd.DataFrame({
    'Wikipedia': [100.0],  # Base dataset
    'Letterboxd': [merged_df['rating'].notna().sum() / len(merged_df) * 100],
    'Rotten Tomatoes': [merged_df['tomatometer_score'].notna().sum() / len(merged_df) * 100],
    'IMDB': [merged_df['imdb_rating'].notna().sum() / len(merged_df) * 100] if 'imdb_rating' in merged_df.columns else [0]
})

# Visualization: Data Completeness
fig, ax = plt.subplots(figsize=(10, 6))
completeness_T = completeness.T
completeness_T.columns = ['Completeness']
bars = ax.barh(completeness_T.index, completeness_T['Completeness'],
               color=['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e'])
ax.set_xlabel('Completeness (%)', fontsize=12)
ax.set_title('Data Completeness by Source', fontsize=14, fontweight='bold')
ax.set_xlim(0, 105)
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
            f'{width:.1f}%',
            ha='left', va='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('insights/data_completeness.png', dpi=300, bbox_inches='tight')
plt.close()
print("  - Saved visualization: insights/data_completeness.png")

# ============================================================================
# STEP 7: Insights and Analysis
# ============================================================================
print("\n[7] Generating insights and visualizations...")

# Insight 1: Winners vs Nominees - Rating Comparison
print("  - Analyzing winners vs nominees ratings...")
if 'rating' in merged_df.columns:
    winners_ratings = merged_df[merged_df['is_winner'] == True]['rating'].dropna()
    nominees_ratings = merged_df[merged_df['is_winner'] == False]['rating'].dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot
    data_to_plot = [winners_ratings, nominees_ratings]
    box = ax1.boxplot(data_to_plot, labels=['Winners', 'Nominees'], patch_artist=True)
    for patch, color in zip(box['boxes'], ['#FFD700', '#C0C0C0']):
        patch.set_facecolor(color)
    ax1.set_ylabel('Letterboxd Rating', fontsize=12)
    ax1.set_title('Letterboxd Ratings: Winners vs Nominees', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Distribution plot
    ax2.hist(winners_ratings, alpha=0.7, label='Winners', bins=20, color='#FFD700', edgecolor='black')
    ax2.hist(nominees_ratings, alpha=0.7, label='Nominees', bins=20, color='#C0C0C0', edgecolor='black')
    ax2.set_xlabel('Letterboxd Rating', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Ratings', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('insights/winners_vs_nominees_ratings.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: insights/winners_vs_nominees_ratings.png")

# Insight 2: Ratings correlation between platforms
print("  - Analyzing rating correlations across platforms...")
if all(col in merged_df.columns for col in ['rating', 'tomatometer_score', 'imdb_rating']):
    correlation_data = merged_df[['rating', 'tomatometer_score', 'imdb_rating']].dropna()

    if len(correlation_data) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Normalize ratings to 0-100 scale for comparison
        correlation_data_norm = correlation_data.copy()
        correlation_data_norm['rating'] = correlation_data_norm['rating'] * 20  # Letterboxd is 0-5
        # tomatometer_score already 0-100
        correlation_data_norm['imdb_rating'] = correlation_data_norm['imdb_rating'] * 10  # IMDB is 0-10

        correlation_data_norm.columns = ['Letterboxd (0-100)', 'Rotten Tomatoes', 'IMDB (0-100)']

        # Calculate correlation matrix
        corr = correlation_data_norm.corr()

        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=2, cbar_kws={"shrink": 0.8},
                   fmt='.3f', ax=ax)
        ax.set_title('Rating Correlations Across Platforms', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('insights/rating_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: insights/rating_correlations.png")

# Insight 3: Temporal trends - Winners over time
print("  - Analyzing temporal trends...")

# Extract year from 'Year of Film Release'
merged_df['year_numeric'] = merged_df['Year of Film Release'].str.extract(r'(\d{4})')[0].astype(float)

# Group by decade
merged_df['decade'] = (merged_df['year_numeric'] // 10 * 10).astype('Int64')

# Count winners by decade
winners_by_decade = merged_df[merged_df['is_winner'] == True].groupby('decade').size()
nominees_by_decade = merged_df.groupby('decade').size()

fig, ax = plt.subplots(figsize=(14, 7))
x = nominees_by_decade.index.astype(str) + 's'
width = 0.35
x_pos = np.arange(len(x))

bars1 = ax.bar(x_pos - width/2, nominees_by_decade.values, width,
               label='Total Nominees', color='#C0C0C0', edgecolor='black')
bars2 = ax.bar(x_pos + width/2, winners_by_decade.reindex(nominees_by_decade.index, fill_value=0).values,
               width, label='Winners', color='#FFD700', edgecolor='black')

ax.set_xlabel('Decade', fontsize=12)
ax.set_ylabel('Number of Films', fontsize=12)
ax.set_title('Oscar Best Picture Nominees and Winners by Decade', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(x, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('insights/nominees_by_decade.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ Saved: insights/nominees_by_decade.png")

# Insight 4: Popularity metrics (Letterboxd)
print("  - Analyzing popularity metrics...")
if 'number_of_watches' in merged_df.columns:
    # Convert number_of_watches (remove K suffix and convert)
    def convert_to_numeric(val):
        if pd.isna(val):
            return np.nan
        val = str(val)
        if 'M' in val:
            return float(val.replace('M', '')) * 1000000
        if 'K' in val:
            return float(val.replace('K', '')) * 1000
        try:
            return float(val)
        except ValueError:
            return np.nan

    merged_df['watches_numeric'] = merged_df['number_of_watches'].apply(convert_to_numeric)
    merged_df['likes_numeric'] = merged_df['number_of_likes'].apply(convert_to_numeric)

    # Top 20 most watched films
    top_watched = merged_df.nlargest(20, 'watches_numeric')[['Film', 'watches_numeric', 'is_winner']]

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ['#FFD700' if w else '#1f77b4' for w in top_watched['is_winner']]
    bars = ax.barh(range(len(top_watched)), top_watched['watches_numeric'], color=colors, edgecolor='black')
    ax.set_yticks(range(len(top_watched)))
    ax.set_yticklabels(top_watched['Film'], fontsize=10)
    ax.set_xlabel('Number of Watches', fontsize=12)
    ax.set_title('Top 20 Most Watched Oscar Nominees on Letterboxd', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#FFD700', edgecolor='black', label='Winner'),
                      Patch(facecolor='#1f77b4', edgecolor='black', label='Nominee')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig('insights/most_watched_films.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: insights/most_watched_films.png")

# Insight 5: Rotten Tomatoes - Critics vs Audience
print("  - Analyzing critics vs audience scores...")
if all(col in merged_df.columns for col in ['tomatometer_score', 'popcornmeter_score']):
    valid_rt = merged_df[['Film', 'tomatometer_score', 'popcornmeter_score', 'is_winner']].dropna()

    if len(valid_rt) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Scatter plot
        winners = valid_rt[valid_rt['is_winner'] == True]
        nominees = valid_rt[valid_rt['is_winner'] == False]

        ax1.scatter(nominees['tomatometer_score'], nominees['popcornmeter_score'],
                   alpha=0.6, s=100, color='#1f77b4', label='Nominees', edgecolors='black', linewidth=0.5)
        ax1.scatter(winners['tomatometer_score'], winners['popcornmeter_score'],
                   alpha=0.8, s=150, color='#FFD700', label='Winners', edgecolors='black', linewidth=1, marker='*')

        # Add diagonal line (critics = audience)
        ax1.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='Critics = Audience')

        ax1.set_xlabel('Tomatometer (Critics) Score', fontsize=12)
        ax1.set_ylabel('Popcornmeter (Audience) Score', fontsize=12)
        ax1.set_title('Critics vs Audience Scores', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 105)
        ax1.set_ylim(0, 105)

        # Difference analysis
        valid_rt['score_diff'] = valid_rt['tomatometer_score'] - valid_rt['popcornmeter_score']

        # Top 10 most controversial (biggest difference)
        controversial = valid_rt.nlargest(15, 'score_diff')[['Film', 'score_diff', 'tomatometer_score', 'popcornmeter_score']]

        colors = ['#d62728' if diff > 0 else '#2ca02c' for diff in controversial['score_diff']]
        bars = ax2.barh(range(len(controversial)), controversial['score_diff'], color=colors, edgecolor='black')
        ax2.set_yticks(range(len(controversial)))
        ax2.set_yticklabels(controversial['Film'], fontsize=9)
        ax2.set_xlabel('Score Difference (Critics - Audience)', fontsize=12)
        ax2.set_title('Most Controversial Films\n(Positive = Critics Loved More)', fontsize=12, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig('insights/critics_vs_audience.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: insights/critics_vs_audience.png")

# ============================================================================
# STEP 8: Summary Statistics
# ============================================================================
print("\n[8] Generating summary statistics...")

summary_stats = {
    'Total Films': len(merged_df),
    'Winners': len(merged_df[merged_df['is_winner'] == True]),
    'Nominees': len(merged_df[merged_df['is_winner'] == False]),
    'Year Range': f"{merged_df['year_numeric'].min():.0f} - {merged_df['year_numeric'].max():.0f}",
    'Avg Letterboxd Rating': f"{merged_df['rating'].mean():.2f}",
    'Avg Tomatometer': f"{merged_df['tomatometer_score'].mean():.1f}%",
    'Avg Popcornmeter': f"{merged_df['popcornmeter_score'].mean():.1f}%",
}

with open('insights/summary_statistics.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("SUMMARY STATISTICS\n")
    f.write("=" * 80 + "\n\n")
    for key, value in summary_stats.items():
        f.write(f"{key:.<40} {value}\n")

    f.write("\n\nKEY FINDINGS:\n")
    f.write("-" * 80 + "\n")

    # Finding 1: Average rating difference
    if 'rating' in merged_df.columns:
        winner_avg = merged_df[merged_df['is_winner'] == True]['rating'].mean()
        nominee_avg = merged_df[merged_df['is_winner'] == False]['rating'].mean()
        f.write(f"\n1. Winners have an average Letterboxd rating of {winner_avg:.2f}\n")
        f.write(f"   compared to nominees' {nominee_avg:.2f}\n")
        f.write(f"   Difference: {winner_avg - nominee_avg:.2f} points\n")

    # Finding 2: Data completeness
    f.write(f"\n2. Data completeness varies significantly:\n")
    f.write(f"   - Wikipedia (base): 100%\n")
    f.write(f"   - Letterboxd: {completeness['Letterboxd'].values[0]:.1f}%\n")
    f.write(f"   - Rotten Tomatoes: {completeness['Rotten Tomatoes'].values[0]:.1f}%\n")
    if 'IMDB' in completeness.columns:
        f.write(f"   - IMDB: {completeness['IMDB'].values[0]:.1f}%\n")

    # Finding 3: Critics vs Audience
    if all(col in merged_df.columns for col in ['tomatometer_score', 'popcornmeter_score']):
        avg_critic = merged_df['tomatometer_score'].mean()
        avg_audience = merged_df['popcornmeter_score'].mean()
        f.write(f"\n3. On Rotten Tomatoes:\n")
        f.write(f"   - Critics average: {avg_critic:.1f}%\n")
        f.write(f"   - Audience average: {avg_audience:.1f}%\n")
        f.write(f"   - Critics are {'more generous' if avg_critic > avg_audience else 'harsher'} by {abs(avg_critic - avg_audience):.1f} points\n")

print("  - Saved summary statistics to insights/summary_statistics.txt")

# ============================================================================
# FINAL REPORT
# ============================================================================
print("\n" + "=" * 80)
print("EDA COMPLETE!")
print("=" * 80)
print("\nGenerated files in 'insights/' folder:")
print("  1. merged_oscar_data.csv - Complete merged dataset")
print("  2. data_exploration_report.txt - Detailed data structure analysis")
print("  3. missing_films_analysis.txt - Missing films across datasets")
print("  4. summary_statistics.txt - Key statistics and findings")
print("\nGenerated visualizations:")
print("  1. data_coverage.png - Film coverage across sources")
print("  2. data_completeness.png - Data completeness by source")
print("  3. winners_vs_nominees_ratings.png - Rating comparison")
print("  4. rating_correlations.png - Cross-platform rating correlations")
print("  5. nominees_by_decade.png - Temporal trends")
print("  6. most_watched_films.png - Top watched films on Letterboxd")
print("  7. critics_vs_audience.png - Critics vs audience analysis")
print("\n" + "=" * 80)
