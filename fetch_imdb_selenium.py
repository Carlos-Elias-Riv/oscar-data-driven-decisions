#!/usr/bin/env python3
"""
Script to fetch missing IMDb data for Oscar-nominated movies using Selenium.
This uses a headless browser to navigate IMDb and extract data.
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


def search_imdb_selenium(title: str, year: Optional[str] = None) -> Optional[Dict]:
    """
    Search for a movie on IMDb using Selenium and return its data.

    Args:
        title: Movie title to search for
        year: Optional year to help narrow down the search

    Returns:
        Dictionary containing IMDb data or None if not found
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.chrome.options import Options
        from selenium.common.exceptions import TimeoutException, NoSuchElementException

        logger.info(f"Searching for: {title} ({year})")

        # Set up Chrome options for headless browsing
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

        # Initialize the driver
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(30)

        try:
            # Build search URL
            search_query = f"{title} {year}" if year else title
            search_url = f"https://www.imdb.com/find/?q={search_query.replace(' ', '+')}&s=tt"

            logger.info(f"Navigating to: {search_url}")
            driver.get(search_url)

            # Wait for search results
            wait = WebDriverWait(driver, 10)

            # Try to find the first movie result
            try:
                first_result = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "li.find-title-result"))
                )
                movie_link = first_result.find_element(By.CSS_SELECTOR, "a")
                movie_url = movie_link.get_attribute('href')

                logger.info(f"Found movie: {movie_url}")

                # Navigate to the movie page
                driver.get(movie_url)
                time.sleep(2)  # Wait for page to load

                # Extract data from the movie page
                data = {
                    'Film': title,
                    'url_tried': search_url,
                    'url_used': movie_url,
                    'found': True,
                    'error': ''
                }

                # Extract rating
                try:
                    rating_elem = driver.find_element(By.CSS_SELECTOR, "[data-testid='hero-rating-bar__aggregate-rating__score'] span")
                    data['rating'] = rating_elem.text
                except NoSuchElementException:
                    data['rating'] = ''

                # Extract year
                try:
                    year_elem = driver.find_element(By.CSS_SELECTOR, "[data-testid='hero-title-block__metadata'] li:first-child")
                    data['year'] = year_elem.text
                except NoSuchElementException:
                    data['year'] = year if year else ''

                # Extract metascore
                try:
                    metascore_elem = driver.find_element(By.CSS_SELECTOR, "[data-testid='metacritic-score-box'] span")
                    data['metascore'] = metascore_elem.text
                except NoSuchElementException:
                    data['metascore'] = ''

                # Extract user reviews count
                try:
                    reviews_elem = driver.find_element(By.CSS_SELECTOR, "[data-testid='hero-rating-bar__user-rating'] div[role='button']")
                    reviews_text = reviews_elem.text
                    # Extract number from text like "2.5M"
                    data['user_reviews'] = reviews_text
                except NoSuchElementException:
                    data['user_reviews'] = ''

                # Extract cast
                try:
                    cast_section = driver.find_element(By.CSS_SELECTOR, "[data-testid='title-cast'] [data-testid='shoveler']")
                    cast_items = cast_section.find_elements(By.CSS_SELECTOR, "[data-testid='shoveler-item-wrapper']")[:3]
                    cast_names = []
                    for item in cast_items:
                        try:
                            name = item.find_element(By.CSS_SELECTOR, "[data-testid='title-cast-item__actor']").text
                            cast_names.append(name)
                        except:
                            pass
                    data['stars'] = str(cast_names)
                    data['cast_number'] = len(cast_items)
                except NoSuchElementException:
                    data['stars'] = ''
                    data['cast_number'] = ''

                # Set default values for fields we can't easily extract
                data['classification_rating'] = ''
                data['keywords'] = ''
                data['critic_reviews'] = ''
                data['budget'] = ''
                data['revenue'] = ''
                data['opening_weekend'] = ''
                data['longitud'] = ''
                data['imdb_url'] = movie_url

                logger.info(f"Successfully extracted data for: {title}")
                return data

            except TimeoutException:
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

        finally:
            driver.quit()

    except ImportError:
        logger.error("Selenium not installed. Install with: pip install selenium")
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

    # Fetch data for missing movies (limit to first 10 for testing)
    new_data = []
    # Process only first 10 movies to avoid running too long
    test_limit = min(10, len(missing_movies))
    for idx, movie in enumerate(missing_movies[:test_limit], 1):
        title = movie['Film']
        year = extract_year(movie.get('Year of Film Release', ''))

        logger.info(f"Processing {idx}/{test_limit}: {title}")

        # Fetch IMDb data
        imdb_data = search_imdb_selenium(title, year)

        if imdb_data:
            new_data.append(imdb_data)

        # Be polite - add a delay between requests
        if idx < test_limit:
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

    # Also write a summary
    found_count = sum(1 for d in new_data if d.get('found'))
    not_found = [d['Film'] for d in new_data if not d.get('found')]

    logger.info(f"Successfully found: {found_count}/{len(new_data)}")
    if not_found:
        logger.warning(f"Not found: {', '.join(not_found)}")


if __name__ == '__main__':
    main()
