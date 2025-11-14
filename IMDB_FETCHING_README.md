# Fetching Missing IMDb Data

## Summary

This repository contains scripts to fetch missing IMDb data for Oscar-nominated movies. Due to IMDb's strict anti-scraping measures, the recommended approach is to use the **OMDb API** (Open Movie Database).

## Problem Encountered

When attempting to scrape IMDb directly, we encountered the following issues:
1. **HTTP 403 Forbidden errors** - IMDb blocks automated requests
2. **Puppeteer browser download blocked** - Network restrictions preventing Chromium download
3. **Cinemagoer library blocked** - IMDb's anti-bot protection blocks the library

## Solution: OMDb API

The OMDb API provides official access to IMDb data through a RESTful API.

### Steps to Use

#### 1. Get Your Free API Key

1. Visit: https://www.omdbapi.com/apikey.aspx
2. Select "FREE! (1,000 daily limit)" option
3. Enter your email address
4. Check your email and verify the activation link
5. You'll receive your API key

#### 2. Set Up the API Key

Option A - Environment Variable (Recommended):
```bash
export OMDB_API_KEY='your_api_key_here'
```

Option B - Enter when prompted by the script

#### 3. Run the Script

```bash
cd /home/user/oscar-data-driven-decisions
python3 fetch_imdb_omdb.py
```

The script will:
- Process all 218 missing movies
- Fetch IMDb data via OMDb API
- Save results to `imdb_results_updated.csv`
- Generate a summary in `fetch_summary.json`

## What Data is Fetched

The script retrieves:
- ✓ IMDb Rating
- ✓ Year
- ✓ Runtime
- ✓ Genre (keywords)
- ✓ Cast (top 3 actors)
- ✓ Metascore
- ✓ Number of votes (user reviews)
- ✓ Box office revenue
- ✓ Content rating (PG, R, etc.)
- ✓ IMDb URL

## Files in This Repository

- `fetch_imdb_omdb.py` - **Recommended script** using OMDb API
- `fetch_imdb_http.py` - HTTP scraping attempt (blocked by IMDb)
- `fetch_imdb_data.py` - Cinemagoer library attempt (blocked by IMDb)
- `mcp-server.js` - MCP Puppeteer server (requires browser download)
- `missing_from_imdb.csv` - List of 218 movies missing IMDb data

## Alternative Approaches

If you cannot use OMDb API, consider:

1. **Local Puppeteer MCP Server** - Set up on your local machine where you have better network access
2. **TMDb API** - The Movie Database API (also free, different data source)
3. **Manual Data Entry** - For smaller datasets
4. **Proxy Service** - Use a proxy to bypass restrictions (not recommended)

## Rate Limits

- OMDb Free Tier: 1,000 requests per day
- Our script: 218 movies to process
- Estimated completion time: ~2 minutes with 0.5s delay between requests

## Next Steps

After fetching the data:
1. Review `imdb_results_updated.csv` for accuracy
2. Merge with existing `imdb_results.csv` if needed
3. Update `clean_merged_data.csv` with the new data
4. Commit and push changes to your repository

## Support

- OMDb API Documentation: https://www.omdbapi.com/
- OMDb Support: support@omdbapi.com
