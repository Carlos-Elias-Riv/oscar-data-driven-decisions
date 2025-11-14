#!/usr/bin/env python3
"""
Script to fetch missing IMDb data for Oscar-nominated movies.
Uses the Cinemagoer library (formerly IMDbPY) to access IMDb data.
"""

import csv
import time
import re
from typing import Dict, List, Optional
import logging

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


def search_imdb_movie(title: str, year: Optional[str] = None) -> Optional[Dict]:
    """
    Search for a movie on IMDb and return its data.

    Args:
        title: Movie title to search for
        year: Optional year to help narrow down the search

    Returns:
        Dictionary containing IMDb data or None if not found
    """
    try:
        from imdb import Cinemagoer

        ia = Cinemagoer()

        # Search for the movie
        logger.info(f"Searching for: {title} ({year})")
        results = ia.search_movie(title)

        if not results:
            logger.warning(f"No results found for: {title}")
            return None

        # Try to find the best match
        movie = None
        if year:
            # Try to match by year
            for result in results[:5]:  # Check first 5 results
                if 'year' in result.data and str(result.data['year']) == year:
                    movie = result
                    break

        # If no year match, take the first result
        if not movie:
            movie = results[0]

        # Get full movie details
        ia.update(movie)

        # Extract the data
        data = {
            'year': movie.data.get('year', ''),
            'classification_rating': movie.data.get('certificates', [''])[0] if movie.data.get('certificates') else '',
            'rating': movie.data.get('rating', ''),
            'keywords': str([kw for kw in movie.data.get('keywords', [])[:10]]),  # Limit to 10 keywords
            'stars': str([person['name'] for person in movie.data.get('cast', [])[:3]]) if movie.data.get('cast') else '',
            'user_reviews': movie.data.get('votes', ''),
            'critic_reviews': '',  # Not directly available
            'metascore': movie.data.get('metascore', ''),
            'cast_number': len(movie.data.get('cast', [])),
            'budget': '',  # Would need additional API call
            'revenue': '',  # Would need additional API call
            'opening_weekend': '',  # Would need additional API call
            'longitud': movie.data.get('runtimes', [''])[0] if movie.data.get('runtimes') else '',
            'imdb_url': f"https://www.imdb.com/title/tt{movie.movieID}/",
            'Film': title,
            'url_tried': f"https://www.imdb.com/find?q={title.replace(' ', '+')}",
            'url_used': f"https://www.imdb.com/title/tt{movie.movieID}/",
            'found': True,
            'error': ''
        }

        logger.info(f"Found: {title} - {data['imdb_url']}")
        return data

    except ImportError:
        logger.error("Cinemagoer library not installed. Install with: pip install cinemagoer")
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
        imdb_data = search_imdb_movie(title, year)

        if imdb_data:
            new_data.append(imdb_data)

        # Be polite to IMDb servers - add a delay
        if idx < len(missing_movies):
            time.sleep(1)  # 1 second delay between requests

    # Combine with existing data
    all_data = existing_data + new_data

    # Write updated results
    with open('imdb_results_updated.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

    logger.info(f"Successfully fetched data for {len(new_data)} movies")
    logger.info(f"Results written to imdb_results_updated.csv")

    # Also write a summary of what was found
    found_count = sum(1 for d in new_data if d.get('found'))
    not_found = [d['Film'] for d in new_data if not d.get('found')]

    logger.info(f"Successfully found: {found_count}/{len(new_data)}")
    if not_found:
        logger.warning(f"Not found: {', '.join(not_found)}")


if __name__ == '__main__':
    main()
