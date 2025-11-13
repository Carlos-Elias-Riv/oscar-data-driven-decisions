# Oscar Best Picture Data - Clean Merged Dataset

This repository contains clean, merged data for Oscar Best Picture nominees (1927-2024) from multiple sources: Wikipedia, IMDB, Letterboxd, and Rotten Tomatoes.

## Quick Start

```bash
# Install dependencies
pip install pandas numpy unidecode

# Run the cleaning script
python create_clean_data.py
```

## Output Files

### 1. `clean_merged_data.csv`
**Master dataset** with all sources intelligently merged
- **613 films** from 1927-2024
- **25 columns** including ratings, metadata, and identifiers
- **Data completeness**: Letterboxd 95.4%, IMDB 62.5%, Rotten Tomatoes 76.3%

### 2. `missing_from_letterboxd.csv`
Films in Wikipedia but NOT found in Letterboxd
- **1 film**: Il Postino: The Postman (1995)

### 3. `missing_from_rottentomatoes.csv`
Films in Wikipedia but NOT found in Rotten Tomatoes
- **145 films** missing RT data
- Includes film name, year, clean_name, and winner status

### 4. `missing_from_imdb.csv`
Films in Wikipedia but NOT found in IMDB dataset
- **218 films** missing IMDB data
- Includes film name, year, clean_name, and winner status

## Data Sources

### Wikipedia (`clean_wiki_data.csv`)
- **Source of truth** - Complete list of all Best Picture nominees
- 611 films from 1927 to 2024
- Includes winner status for each film

### Letterboxd (`letterboxd_results.csv`)
- User ratings (0-5 scale)
- Popularity metrics (watches, likes, lists)
- Director information
- **Match rate**: 99.8% (610/611 films)

### IMDB (`imdb_results.csv`)
- User ratings (0-10 scale)
- Metascores
- Cast information, keywords, genres
- Budget and revenue data
- **Match rate**: 64.3% (393/611 films)

### Rotten Tomatoes (`rottentomatoes_data_incomplete.csv`)
- Tomatometer (critics) scores
- Popcornmeter (audience) scores
- **Match rate**: 76.3% (466/611 films)

## Data Cleaning Features

### Intelligent Name Matching
The `create_clean_data.py` script implements clever strategies to match film names across datasets:

1. **IMDB Matching** (62.5% coverage)
   - Extracts film names from `url_tried` column (contains year!)
   - Handles accented characters (Les Misérables, etc.)
   - Year-agnostic fallback matching for year discrepancies

2. **Rotten Tomatoes Matching** (76.3% coverage)
   - Underscore-to-hyphen conversion
   - Name-without-year matching
   - Handles year variations between sources

3. **Accent Normalization**
   - Uses `unidecode` library
   - Converts accented characters to ASCII equivalents
   - Ensures matches for international films

### Key Techniques
- **URL extraction**: Uses Letterboxd URLs from IMDB data (breakthrough!)
- **Year-agnostic matching**: Handles year discrepancies between sources
- **Multiple fallback strategies**: Maximizes successful matches
- **Wikipedia as source of truth**: All matching validated against complete nominee list

## Column Descriptions

### From Wikipedia
- `Film` - Official film title
- `Year of Film Release` - Oscar ceremony year
- `Film Studio/Producer(s)` - Production information
- `is_winner` - Boolean (True if won Best Picture)
- `clean_name` - Standardized identifier (e.g., "film-name-year")

### From Letterboxd
- `rating` - Average user rating (0-5)
- `description` - Film synopsis
- `number_of_lists` - Times added to user lists
- `number_of_watches` - Total watch count
- `number_of_likes` - Total likes
- `number_of_fans` - Fan count
- `name_of_director` - Director name

### From IMDB
- `imdb_rating` - User rating (0-10)
- `imdb_metascore` - Critic metascore (0-100)
- `imdb_user_reviews` - Number of user reviews
- `imdb_critic_reviews` - Number of critic reviews
- `imdb_keywords` - Associated keywords/genres
- `imdb_cast_number` - Cast size
- `imdb_budget` - Production budget
- `imdb_revenue` - Box office revenue

### From Rotten Tomatoes
- `tomatometer_score` - Critics score (0-100)
- `popcornmeter_score` - Audience score (0-100)

## Statistics

- **Total Best Picture nominees**: 611 (source: Wikipedia)
- **Best Picture winners**: 97
- **Year range**: 1927-2024 (97 years)
- **Average Letterboxd rating**: 3.76/5.0
- **Winners vs Nominees**: Winners average 3.86, nominees 3.74 (+0.12)

## Usage Examples

```python
import pandas as pd

# Load the clean merged data
df = pd.read_csv('clean_merged_data.csv')

# Get all Best Picture winners
winners = df[df['is_winner'] == True]

# Find films with both IMDB and RT scores
complete_ratings = df.dropna(subset=['imdb_rating', 'tomatometer_score'])

# Get films from a specific decade
df['year'] = df['Year of Film Release'].str.extract(r'(\d{4})')
films_2010s = df[df['year'].between('2010', '2019')]

# Load missing films to see what needs to be added
missing_rt = pd.read_csv('missing_from_rottentomatoes.csv')
print(f"Need to add {len(missing_rt)} films to Rotten Tomatoes data")
```

## Re-running the Cleaning

To regenerate the clean datasets:

```bash
python create_clean_data.py
```

This will:
1. Load all source CSVs
2. Apply intelligent matching strategies
3. Create clean_merged_data.csv
4. Generate missing_from_*.csv files
5. Print summary statistics

## Notes

- **Letterboxd** has the best coverage (99.8%) - only missing "Il Postino: The Postman"
- **IMDB** dataset may not contain all nominees (only 393 of 611)
- **Rotten Tomatoes** data is marked as incomplete in the filename
- Some films have year discrepancies between sources (handled by year-agnostic matching)
- Accented characters are normalized for matching but preserved in original data

## Advanced Analysis

The repository also includes additional scripts in the `insights/` folder for exploratory data analysis and visualizations. See `insights/README.md` for details.

---

**Data Last Updated**: 2024
**Script**: `create_clean_data.py`
**Contact**: Check the repository for issues or contributions
