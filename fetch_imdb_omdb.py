#!/usr/bin/env python3
"""
Script to fetch missing IMDb data using OMDb API.
Get your free API key from: https://www.omdbapi.com/apikey.aspx
"""

import csv
import time
import re
import os
from typing import Dict, List, Optional
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_year(year_str: str) -> Optional[str]:
    """Extract the year from the Year of Film Release column."""
    if not year_str:
        return None
    # Extract first 4-digit year from strings like "1929/30 (3rd)"
    match = re.search(r'(\d{4})', year_str)
    return match.group(1) if match else None


def search_omdb_movie(title: str, year: Optional[str] = None, api_key: str = None) -> Optional[Dict]:
    """
    Search for a movie using OMDb API and return its data.

    Args:
        title: Movie title to search for
        year: Optional year to help narrow down the search
        api_key: OMDb API key (get free key from https://www.omdbapi.com/apikey.aspx)

    Returns:
        Dictionary containing IMDb data or None if not found
    """
    try:
        import requests

        if not api_key:
            logger.error("No API key provided. Get a free key from: https://www.omdbapi.com/apikey.aspx")
            return None

        logger.info(f"Searching for: {title} ({year})")

        # Build API URL
        params = {
            'apikey': api_key,
            't': title,  # Search by title
            'type': 'movie',
            'plot': 'short'
        }

        if year:
            params['y'] = year

        response = requests.get('http://www.omdbapi.com/', params=params, timeout=30)

        if response.status_code != 200:
            logger.warning(f"API request failed with status {response.status_code}")
            return {
                'year': year or '',
                'classification_rating': '',
                'rating': '',
                'keywords': '',
                'stars': '',
                'user_reviews': '',
                'critic_reviews': '',
                'metascore': '',
                'cast_number': '',
                'budget': '',
                'revenue': '',
                'opening_weekend': '',
                'longitud': '',
                'imdb_url': '',
                'Film': title,
                'url_tried': f"omdbapi.com?t={title}&y={year}",
                'url_used': '',
                'found': False,
                'error': f'HTTP {response.status_code}'
            }

        data_json = response.json()

        if data_json.get('Response') == 'False':
            logger.warning(f"Movie not found: {title}")
            return {
                'year': year or '',
                'classification_rating': '',
                'rating': '',
                'keywords': '',
                'stars': '',
                'user_reviews': '',
                'critic_reviews': '',
                'metascore': '',
                'cast_number': '',
                'budget': '',
                'revenue': '',
                'opening_weekend': '',
                'longitud': '',
                'imdb_url': '',
                'Film': title,
                'url_tried': f"omdbapi.com?t={title}&y={year}",
                'url_used': '',
                'found': False,
                'error': data_json.get('Error', 'Not found')
            }

        # Extract data from the API response
        imdb_id = data_json.get('imdbID', '')
        imdb_url = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else ''

        # Parse actors
        actors = data_json.get('Actors', '')
        actors_list = [a.strip() for a in actors.split(',')[:3]] if actors and actors != 'N/A' else []

        # Parse runtime (e.g., "142 min" -> "142")
        runtime = data_json.get('Runtime', '')
        runtime_match = re.search(r'(\d+)', runtime)
        runtime_minutes = runtime_match.group(1) if runtime_match else ''

        # Parse IMDb rating
        imdb_rating = data_json.get('imdbRating', '')
        if imdb_rating == 'N/A':
            imdb_rating = ''

        # Parse metascore
        metascore = data_json.get('Metascore', '')
        if metascore == 'N/A':
            metascore = ''

        # Parse IMDb votes (this is the user reviews count)
        imdb_votes = data_json.get('imdbVotes', '')
        if imdb_votes and imdb_votes != 'N/A':
            # Remove commas (e.g., "1,234,567" -> "1234567")
            imdb_votes = imdb_votes.replace(',', '')

        data = {
            'year': data_json.get('Year', year or ''),
            'classification_rating': data_json.get('Rated', ''),
            'rating': imdb_rating,
            'keywords': str([g.strip() for g in data_json.get('Genre', '').split(',')]) if data_json.get('Genre') and data_json.get('Genre') != 'N/A' else '',
            'stars': str(actors_list),
            'user_reviews': imdb_votes,
            'critic_reviews': '',  # Not available in OMDb
            'metascore': metascore,
            'cast_number': len(actors_list),
            'budget': '',  # Not available in basic OMDb response
            'revenue': data_json.get('BoxOffice', '').replace('$', '').replace(',', '') if data_json.get('BoxOffice') and data_json.get('BoxOffice') != 'N/A' else '',
            'opening_weekend': '',  # Not available in OMDb
            'longitud': runtime_minutes,
            'imdb_url': imdb_url,
            'Film': title,
            'url_tried': f"omdbapi.com?t={title}&y={year}",
            'url_used': imdb_url,
            'found': True,
            'error': ''
        }

        logger.info(f"Found: {title} - {imdb_url} - Rating: {imdb_rating}")
        return data

    except ImportError:
        logger.error("Requests library not installed. Install with: pip install requests")
        return None
    except Exception as e:
        logger.error(f"Error fetching data for {title}: {str(e)}")
        return {
            'year': year or '',
            'classification_rating': '',
            'rating': '',
            'keywords': '',
            'stars': '',
            'user_reviews': '',
            'critic_reviews': '',
            'metascore': '',
            'cast_number': '',
            'budget': '',
            'revenue': '',
            'opening_weekend': '',
            'longitud': '',
            'imdb_url': '',
            'Film': title,
            'url_tried': '',
            'url_used': '',
            'found': False,
            'error': str(e)
        }


def main():
    """Main function to process missing movies and fetch IMDb data."""

    # Get API key from environment variable or user input
    api_key = os.environ.get('OMDB_API_KEY')

    if not api_key:
        print("\n" + "="*70)
        print("OMDb API Key Required")
        print("="*70)
        print("\nTo use this script, you need a free OMDb API key.")
        print("\nSteps to get your API key:")
        print("1. Visit: https://www.omdbapi.com/apikey.aspx")
        print("2. Select 'FREE! (1,000 daily limit)' option")
        print("3. Enter your email and verify it")
        print("4. You'll receive your API key by email")
        print("\nOnce you have your key, you can:")
        print("- Set environment variable: export OMDB_API_KEY='your_key_here'")
        print("- Or enter it when prompted below")
        print("\n" + "="*70 + "\n")

        api_key = input("Enter your OMDb API key (or press Enter to exit): ").strip()

        if not api_key:
            print("No API key provided. Exiting.")
            return

    # Read missing movies
    missing_movies = []
    with open('missing_from_imdb.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        missing_movies = list(reader)

    logger.info(f"Found {len(missing_movies)} missing movies")

    # Read existing IMDb results to append to
    existing_data = []
    fieldnames = []
    try:
        with open('imdb_results.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            existing_data = list(reader)
    except FileNotFoundError:
        logger.warning("imdb_results.csv not found, will create new file")
        fieldnames = ['year', 'classification_rating', 'rating', 'keywords', 'stars',
                     'user_reviews', 'critic_reviews', 'metascore', 'cast_number',
                     'budget', 'revenue', 'opening_weekend', 'longitud', 'imdb_url',
                     'Film', 'url_tried', 'url_used', 'found', 'error']

    # Fetch data for missing movies
    new_data = []
    for idx, movie in enumerate(missing_movies, 1):
        title = movie['Film']
        year = extract_year(movie.get('Year of Film Release', ''))

        logger.info(f"Processing {idx}/{len(missing_movies)}: {title}")

        # Fetch IMDb data
        imdb_data = search_omdb_movie(title, year, api_key)

        if imdb_data:
            new_data.append(imdb_data)

        # Be polite - add a small delay between requests (free tier has 1000 daily limit)
        if idx < len(missing_movies):
            time.sleep(0.5)  # 0.5 second delay

    # Combine with existing data
    all_data = existing_data + new_data

    # Write updated results
    with open('imdb_results_updated.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

    logger.info(f"Successfully processed {len(new_data)} movies")
    logger.info(f"Results written to imdb_results_updated.csv")

    # Write a summary
    found_count = sum(1 for d in new_data if d.get('found'))
    not_found = [d['Film'] for d in new_data if not d.get('found')]

    logger.info(f"Successfully found: {found_count}/{len(new_data)}")
    if not_found:
        logger.warning(f"Not found ({len(not_found)}): {', '.join(not_found[:10])}{'...' if len(not_found) > 10 else ''}")

    # Save summary to file
    with open('fetch_summary.json', 'w') as f:
        json.dump({
            'total_processed': len(new_data),
            'found': found_count,
            'not_found': not_found
        }, f, indent=2)

    print(f"\n✓ Processed {len(new_data)} movies")
    print(f"✓ Successfully found: {found_count}/{len(new_data)}")
    print(f"✓ Results saved to: imdb_results_updated.csv")
    print(f"✓ Summary saved to: fetch_summary.json\n")


if __name__ == '__main__':
    main()
