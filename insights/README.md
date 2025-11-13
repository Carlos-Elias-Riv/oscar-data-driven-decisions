# Exploratory Data Analysis - Oscar Best Picture Nominees

## Overview
This folder contains the results of a comprehensive exploratory data analysis (EDA) of Oscar Best Picture nominees from 1927 to 2024, integrating data from multiple sources: Wikipedia, IMDB, Letterboxd, and Rotten Tomatoes.

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
- **Winners average**: 3.86/5.0 ⭐
- **Nominees average**: 3.74/5.0 ⭐
- **Difference**: 0.12 points
- **Insight**: Academy winners tend to be slightly better received by modern audiences

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

### 5. Data Quality
- Wikipedia: 100% complete (baseline dataset)
- Letterboxd: 95.4% complete (excellent integration)
- Rotten Tomatoes: Limited integration due to naming inconsistencies
- IMDB: Limited integration due to naming inconsistencies

## Generated Files

### Data Files
- **`merged_oscar_data.csv`**: Complete merged dataset with all sources (611 records, 20 columns)
- **`data_exploration_report.txt`**: Detailed structure and missing value analysis
- **`missing_films_analysis.txt`**: Comprehensive breakdown of missing films across datasets
- **`summary_statistics.txt`**: Key statistics and findings summary

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

## Reproducibility

To reproduce this analysis:

```bash
# Install required packages
pip install pandas numpy matplotlib seaborn

# Run the analysis script
python exploratory_data_analysis.py
```

All outputs will be regenerated in the `insights/` folder.

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
