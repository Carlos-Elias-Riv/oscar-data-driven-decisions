#!/usr/bin/env python3
"""
Script to fetch missing IMDb data for Oscar-nominated movies using HTTP requests.
Uses beautifulsoup4 and requests with proper headers to avoid detection.
"""

import csv
import time
import re
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


def search_imdb_movie_http(title: str, year: Optional[str] = None) -> Optional[Dict]:
    """
    Search for a movie on IMDb using HTTP requests and return its data.

    Args:
        title: Movie title to search for
        year: Optional year to help narrow down the search

    Returns:
        Dictionary containing IMDb data or None if not found
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        # Build search URL
        search_query = f"{title} {year}" if year else title
        search_url = f"https://www.imdb.com/find/?q={search_query.replace(' ', '+')}&s=tt"

        # Set up headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        logger.info(f"Searching for: {title} ({year})")
        logger.info(f"URL: {search_url}")

        # Make the search request
        session = requests.Session()
        response = session.get(search_url, headers=headers, timeout=30)

        if response.status_code != 200:
            logger.warning(f"Search request failed with status {response.status_code}")
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
                'url_tried': search_url,
                'url_used': '',
                'found': False,
                'error': f'HTTP {response.status_code}'
            }

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the first movie result
        # Try different selectors for finding results
        first_result = None

        # Try new IMDb layout
        result_list = soup.find('ul', class_='ipc-metadata-list')
        if result_list:
            first_result = result_list.find('li', class_='ipc-metadata-list-summary-item')

        # Try older layout
        if not first_result:
            result_table = soup.find('table', class_='findList')
            if result_table:
                first_result = result_table.find('tr', class_='findResult')

        if not first_result:
            logger.warning(f"No results found for: {title}")
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
                'url_tried': search_url,
                'url_used': '',
                'found': False,
                'error': 'No results found'
            }

        # Extract the movie URL
        movie_link = first_result.find('a', href=re.compile(r'/title/tt\d+/'))
        if not movie_link:
            logger.warning(f"Could not find movie link for: {title}")
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
                'url_tried': search_url,
                'url_used': '',
                'found': False,
                'error': 'No movie link found'
            }

        movie_url = 'https://www.imdb.com' + movie_link['href']
        logger.info(f"Found movie: {movie_url}")

        # Fetch the movie page
        time.sleep(1)  # Be polite
        response = session.get(movie_url, headers=headers, timeout=30)

        if response.status_code != 200:
            logger.warning(f"Movie page request failed with status {response.status_code}")
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
                'imdb_url': movie_url,
                'Film': title,
                'url_tried': search_url,
                'url_used': movie_url,
                'found': False,
                'error': f'HTTP {response.status_code} on movie page'
            }

        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract data from the movie page
        data = {
            'Film': title,
            'url_tried': search_url,
            'url_used': movie_url,
            'imdb_url': movie_url,
            'found': True,
            'error': ''
        }

        # Extract rating
        rating_elem = soup.find('span', {'data-testid': 'hero-rating-bar__aggregate-rating__score'})
        if rating_elem:
            rating_text = rating_elem.get_text(strip=True)
            data['rating'] = rating_text.split('/')[0] if '/' in rating_text else rating_text
        else:
            data['rating'] = ''

        # Extract year
        year_elem = soup.find('a', href=re.compile(r'/title/tt\d+/releaseinfo'))
        if year_elem:
            data['year'] = year_elem.get_text(strip=True)
        else:
            data['year'] = year if year else ''

        # Extract metascore
        metascore_elem = soup.find('span', class_=re.compile(r'.*metacritic.*'))
        if metascore_elem:
            data['metascore'] = metascore_elem.get_text(strip=True)
        else:
            data['metascore'] = ''

        # Extract user reviews count
        reviews_elem = soup.find('div', {'data-testid': 'hero-rating-bar__user-rating'})
        if reviews_elem:
            count_elem = reviews_elem.find('div', class_=re.compile(r'.*'))
            if count_elem:
                reviews_text = count_elem.get_text(strip=True)
                # Extract just the number
                match = re.search(r'([\d.]+[KMB]?)', reviews_text)
                data['user_reviews'] = match.group(1) if match else ''
        else:
            data['user_reviews'] = ''

        # Extract cast
        cast_section = soup.find('div', {'data-testid': 'title-cast'})
        cast_names = []
        if cast_section:
            cast_items = cast_section.find_all('a', {'data-testid': 'title-cast-item__actor'})[:3]
            for item in cast_items:
                cast_names.append(item.get_text(strip=True))

        data['stars'] = str(cast_names)
        data['cast_number'] = len(cast_names)

        # Extract runtime
        runtime_elem = soup.find('li', {'data-testid': 'title-techspec_runtime'})
        if runtime_elem:
            runtime_text = runtime_elem.get_text(strip=True)
            # Extract just the minutes
            match = re.search(r'(\d+)\s*min', runtime_text)
            data['longitud'] = match.group(1) if match else ''
        else:
            data['longitud'] = ''

        # Set default values for fields we can't easily extract
        data['classification_rating'] = ''
        data['keywords'] = ''
        data['critic_reviews'] = ''
        data['budget'] = ''
        data['revenue'] = ''
        data['opening_weekend'] = ''

        logger.info(f"Successfully extracted data for: {title}")
        return data

    except ImportError:
        logger.error("Required libraries not installed. Install with: pip install requests beautifulsoup4")
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
        imdb_data = search_imdb_movie_http(title, year)

        if imdb_data:
            new_data.append(imdb_data)

        # Be polite - add a delay between requests
        if idx < len(missing_movies):
            time.sleep(2)  # 2 second delay between requests

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
        logger.warning(f"Not found: {', '.join(not_found)}")

    # Save summary to file
    with open('fetch_summary.json', 'w') as f:
        json.dump({
            'total_processed': len(new_data),
            'found': found_count,
            'not_found': not_found
        }, f, indent=2)


if __name__ == '__main__':
    main()
