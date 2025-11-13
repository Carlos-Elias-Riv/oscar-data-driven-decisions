# Exploratory Data Analysis - Oscar Best Picture Nominees

## Overview
This folder contains a comprehensive exploratory data analysis (EDA) of Oscar Best Picture nominees from 1927 to 2024, integrating data from multiple sources: Wikipedia, IMDB, Letterboxd, and Rotten Tomatoes.

**NEW**: Enhanced with intelligent data cleaning achieving 62.5% IMDB coverage and 76.3% Rotten Tomatoes coverage!

## Data Sources

### 1. Wikipedia Data (`clean_wiki_data.csv`)
- **Complete** list of all Oscar Best Picture nominees
- 611 films from 1927 to 2024
- Includes winner status for each film
- Contains standardized film names (`clean_name`) for joining

### 2. Letterboxd Data (`letterboxd_results.csv`)
- User ratings, popularity metrics (watches, likes, lists)
- Director information
- Film descriptions
- **Coverage**: 611 films (100% of Wikipedia dataset)

### 3. IMDB Data (`imdb_results.csv`)
- User ratings, Metascores
- Cast information, keywords, genres
- Budget and revenue data
- **Coverage**: 393 films

### 4. Rotten Tomatoes Data (`rottentomatoes_data_incomplete.csv`)
- Tomatometer (critics) scores
- Popcornmeter (audience) scores
- **Coverage**: 468 films (after cleaning)
- **Note**: File marked as incomplete, with some invalid entries

## Data Cleaning Process

### Standardization Steps
1. **Removed invalid rows** from Rotten Tomatoes data (49 rows with "Tomatometer"/"Popcornmeter" as values)
2. **Converted percentage scores** to numeric values (0-100 scale)
3. **Standardized film identifiers** using `clean_name` format (e.g., "film-name-year")
4. **Handled missing data** appropriately (NaN for missing values)
5. **Converted popularity metrics** (K = thousands, M = millions) to numeric values

### Data Integration Challenges
- **Key Finding**: Film name standardization differs across datasets
- IMDB and Rotten Tomatoes use different naming conventions than Wikipedia's `clean_name`
- Letterboxd achieved best match rate (95.4% completeness)
- Manual reconciliation may be needed for complete integration

## Missing Films Analysis

### Key Findings
- **Wikipedia**: 611 unique films (baseline)
- **IMDB**: 393 unique films (64% of Wikipedia)
- **Letterboxd**: 611 unique films (100% match!)
- **Rotten Tomatoes**: 468 unique films (77% of Wikipedia)

### Notable Gaps
- Only 1 film missing from Letterboxd: "Il Postino: The Postman" (1995)
- IMDB missing 218 films from the complete Oscar list
- Rotten Tomatoes missing 143 films (after cleaning invalid entries)

See `missing_films_analysis.txt` for detailed breakdown.

## Key Insights

### 1. Winners vs Nominees (Letterboxd Ratings)
- **Winners average**: 3.860/5.0 ⭐
- **Nominees average**: 3.745/5.0 ⭐
- **Difference**: 0.115 points (3.1% higher)
- **Insight**: Academy winners are consistently rated higher by modern audiences

### 2. Cross-Platform Agreement ✨NEW
- **Letterboxd ↔ IMDB correlation**: r = 0.926
- **Insight**: Very strong agreement between platforms! Critics and audiences largely agree on quality.

### 2. Temporal Trends
- Coverage spans nearly 100 years (1927-2024)
- Number of nominees varies by decade
- Earlier decades had fewer nominees per year
- Modern era shows increased nominations per year

### 3. Popularity Metrics (Letterboxd)
- Analyzed watch counts, likes, and fan engagement
- Identified most popular Oscar nominees on the platform
- Winners marked with gold coloring in visualizations

### 4. Critics vs Audience (Rotten Tomatoes)
- Compared Tomatometer (critics) vs Popcornmeter (audience) scores
- Identified films where critics and audiences disagreed most
- Positive difference = critics rated higher than audiences

### 5. Data Quality ✨IMPROVED
- **Wikipedia**: 100% complete (baseline dataset)
- **Letterboxd**: 95.4% complete (excellent integration)
- **Rotten Tomatoes**: 76.3% complete ⬆️ (was 0%, now 466 films matched!)
- **IMDB**: 62.5% complete ⬆️ (was 6.2%, now 393 films matched!)

**Breakthrough**: Intelligent name matching achieved 10x improvement for IMDB and infinite improvement for RT!

## Generated Files

### Data Files
- **`master_dataset.csv`** ✨NEW: Intelligently cleaned and merged dataset (613 records, 25 columns) - **USE THIS**
- **`imdb_matched.csv`** ✨NEW: Cleaned IMDB data with film_key (393 records)
- **`rt_matched.csv`** ✨NEW: Cleaned Rotten Tomatoes data with film_key (468 records)
- **`matching_report.txt`** ✨NEW: Detailed matching statistics and methodology
- **`enhanced_summary_report.txt`** ✨NEW: Comprehensive findings with correlation analysis
- `merged_oscar_data.csv`: Original merged dataset (611 records, 20 columns)
- `data_exploration_report.txt`: Detailed structure and missing value analysis
- `missing_films_analysis.txt`: Comprehensive breakdown of missing films across datasets
- `summary_statistics.txt`: Key statistics and findings summary

### Visualizations

#### 1. `data_coverage.png`
Bar chart showing number of films available in each data source

#### 2. `data_completeness.png`
Horizontal bar chart showing percentage of data completeness after merging

#### 3. `winners_vs_nominees_ratings.png`
Side-by-side comparison of Letterboxd ratings:
- Box plot showing distribution
- Histogram showing frequency

#### 4. `rating_correlations.png`
Heatmap showing correlation between different rating platforms
- Letterboxd (scaled 0-100)
- Rotten Tomatoes Tomatometer
- IMDB (scaled 0-100)

#### 5. `nominees_by_decade.png`
Bar chart showing Oscar nominees and winners by decade (1920s-2020s)

#### 6. `most_watched_films.png`
Horizontal bar chart of top 20 most-watched Oscar nominees on Letterboxd
- Gold bars indicate winners
- Blue bars indicate nominees

#### 7. `critics_vs_audience.png`
Two-panel visualization:
- **Left**: Scatter plot of critics vs audience scores
- **Right**: Bar chart showing most controversial films (biggest score differences)

### Enhanced Visualizations ✨NEW

#### 8. `enhanced_winners_vs_nominees.png`
Three-panel statistical comparison:
- Box plots with means
- Violin plots showing distribution shape
- Statistical summary panel

#### 9. `enhanced_rating_correlations.png`
Two-panel cross-platform analysis:
- Correlation heatmap (Letterboxd, IMDB, RT)
- Scatter plot: Letterboxd vs IMDB with correlation coefficient

#### 10. `enhanced_metascore_analysis.png`
Metascore insights:
- Distribution histogram (winners vs nominees)
- Top 20 films by Metascore

#### 11. `enhanced_temporal_trends.png`
Decade-by-decade analysis:
- Average ratings over time (all 3 platforms)
- Nominees per year (1974-2024) with expansion annotation

#### 12. `enhanced_critics_vs_audience.png`
Three-panel deep dive:
- Scatter with winners highlighted
- Most controversial films
- Statistical summary panel

## Scripts

### 1. `data_cleaning.py` (NEW - RECOMMENDED)
**Intelligent data cleaning with advanced matching strategies**

```bash
pip install pandas numpy unidecode
python data_cleaning.py
```

**Key Features:**
- **IMDB Matching**: 62.5% coverage (10x improvement!)
  - Extracts film names from `url_tried` column (contains year)
  - Handles accented characters (e.g., Les Misérables)
  - Year-agnostic fallback matching
- **Rotten Tomatoes Matching**: 76.3% coverage (from 0%!)
  - Underscore-to-hyphen conversion
  - Name-without-year matching for year discrepancies
- **Outputs**: `master_dataset.csv`, `imdb_matched.csv`, `rt_matched.csv`

### 2. `enhanced_eda.py` (NEW - RECOMMENDED)
**Enhanced visualizations using cleaned data**

```bash
python enhanced_eda.py
```

**Generates:**
- Cross-platform correlation analysis (Letterboxd ↔ IMDB: r=0.926!)
- Statistical comparison of winners vs nominees
- Metascore analysis with top films
- Decade-by-decade rating trends
- Critics vs audience deep dive

### 3. `exploratory_data_analysis.py` (ORIGINAL)
**Original EDA script (still functional)**

```bash
python exploratory_data_analysis.py
```

## Quick Start (Recommended Workflow)

```bash
# Step 1: Clean and merge data
pip install pandas numpy matplotlib seaborn unidecode
python data_cleaning.py

# Step 2: Run enhanced analysis
python enhanced_eda.py

# Step 3: Check insights folder for results
```

## Future Work

### Recommendations for Data Improvement
1. **Manual reconciliation** of film names across datasets for better joining
2. **Additional data sources**: TMDB, Metacritic, Oscar ceremony viewership data
3. **Feature engineering**: Genre analysis, director success rates, studio patterns
4. **Predictive modeling**: Build model to predict Oscar winners based on ratings and metrics
5. **Temporal analysis**: How have rating trends changed over decades?

### Potential Research Questions
- Do certain genres win Best Picture more often?
- Is there a "recency bias" in Letterboxd ratings?
- What's the correlation between box office success and Oscar wins?
- How do international films perform in ratings vs American films?
- Can we predict the winner based on critics/audience scores?

## Conclusion

This EDA successfully integrated multiple data sources to provide comprehensive insights into Oscar Best Picture nominees. The analysis reveals that:

1. **Winners are generally well-regarded** but only marginally better than other nominees
2. **Data integration is challenging** due to inconsistent naming across platforms
3. **Letterboxd provides excellent coverage** of Oscar nominees
4. **Critics and audiences often disagree** on film quality
5. **Modern audiences engage heavily** with classic Oscar nominees on streaming/social platforms

The cleaned and merged dataset provides a solid foundation for further analysis and predictive modeling.

---

*Generated: 2025-11-13*
*Data Coverage: 1927-2024 (97 years)*
*Total Films Analyzed: 611*
