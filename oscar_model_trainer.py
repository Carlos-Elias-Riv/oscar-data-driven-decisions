"""
Oscar Winner Prediction Model Trainer

This module provides functions to train a logistic regression model for predicting
Oscar Best Picture winners using cumulative training approach.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# Path to the data file
DATA_PATH = os.path.join(os.path.dirname(__file__), 'merged_df_final.csv')

# Features used in the model
NUMERICAL_FEATURES = [
    'rating',
    'user_reviews',
    'critic_reviews',
    'metascore',
    'cast_number',
    'rating_letterboxd',
    'number_of_lists',
    'number_of_watches',
    'number_of_likes',
    'director_nominated_before',
    'director_num_previous_nominations'
]


def load_and_preprocess_data(data_path=None):
    """
    Load the merged dataset and preprocess numeric columns.
    
    Parameters:
    - data_path: Path to CSV file (optional, uses default if not provided)
    
    Returns:
    - DataFrame with cleaned numeric columns
    """
    if data_path is None:
        data_path = DATA_PATH
    
    df = pd.read_csv(data_path)
    
    # Clean K/M suffixes from numeric columns
    columns_to_clean = ['number_of_lists', 'number_of_watches', 'number_of_likes', 
                        'user_reviews', 'critic_reviews', 'number_of_fans']
    
    df['number_of_lists'] = df['number_of_lists'].apply(lambda x: str(float(x.replace('K', '')) * 1000) if 'K' in x else x)
    df['number_of_lists'] = df['number_of_lists'].apply(lambda x: str(float(x.replace('M', '')) * 1000000) if 'M' in x else x)


    df['number_of_watches'] = df['number_of_watches'].apply(lambda x: str(float(x.replace('K', '')) * 1000) if 'K' in x else x)
    df['number_of_watches'] = df['number_of_watches'].apply(lambda x: str(float(x.replace('M', '')) * 1000000) if 'M' in x else x)

    df['number_of_likes'] = df['number_of_likes'].apply(lambda x: str(float(x.replace('K', '')) * 1000) if 'K' in x else x)
    df['number_of_likes'] = df['number_of_likes'].apply(lambda x: str(float(x.replace('M', '')) * 1000000) if 'M' in x else x)

    df['user_reviews'] = df['user_reviews'].apply(lambda x: str(float(str(x).replace('K', '')) * 1000) if 'K' in str(x) else x)
    df['user_reviews'] = df['user_reviews'].apply(lambda x: str(float(str(x).replace('M', '')) * 1000000) if 'M' in str(x) else x)

    df['critic_reviews'] = df['critic_reviews'].apply(lambda x: str(float(str(x).replace('K', '')) * 1000) if 'K' in str(x) else x)
    df['critic_reviews'] = df['critic_reviews'].apply(lambda x: str(float(str(x).replace('M', '')) * 1000000) if 'M' in str(x) else x)

    df['number_of_fans'] = df['number_of_fans'].apply(lambda x: str(x).replace(" FANS", ""))
    df['number_of_fans'] = df['number_of_fans'].apply(lambda x: str(x).replace(" FAN", ""))
    df['number_of_fans'] = df['number_of_fans'].apply(lambda x: str(float(x.replace('K', '')) * 1000) if 'K' in x else x)
    df['number_of_fans'] = df['number_of_fans'].apply(lambda x: str(float(x.replace('M', '')) * 1000000) if 'M' in x else x)

    
    # Convert to numeric
    numeric_cols = ['number_of_lists', 'number_of_watches', 'number_of_likes', 
                    'number_of_fans', 'user_reviews', 'critic_reviews']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df


def create_director_features(df):
    """
    Create director nomination history features.
    
    Parameters:
    - df: DataFrame with 'year' and 'name_of_director' columns
    
    Returns:
    - DataFrame with added director features
    """
    df = df.copy()
    df['director_nominated_before'] = 0
    df['director_num_previous_nominations'] = 0
    
    unique_years = sorted(df['year'].dropna().unique())
    
    for current_year in unique_years:
        current_year_mask = df['year'] == current_year
        current_year_indices = df[current_year_mask].index
        previous_years = [y for y in unique_years if y < current_year]
        
        for idx in current_year_indices:
            director = df.loc[idx, 'name_of_director']
            previous_nominations = df[
                (df['year'].isin(previous_years)) & 
                (df['name_of_director'] == director)
            ].shape[0]
            
            df.loc[idx, 'director_nominated_before'] = 1 if previous_nominations > 0 else 0
            df.loc[idx, 'director_num_previous_nominations'] = previous_nominations
    
    return df


def prepare_features(df, features=None):
    """
    Prepare feature matrix with imputation.
    
    Parameters:
    - df: DataFrame with features
    - features: List of feature names (optional, uses default if not provided)
    
    Returns:
    - DataFrame with imputed features
    """
    if features is None:
        features = NUMERICAL_FEATURES
    
    available_features = [f for f in features if f in df.columns]
    
    imputer = SimpleImputer(strategy='mean')
    df_features = df[available_features].copy()
    df[available_features] = imputer.fit_transform(df_features)
    
    return df, available_features


def cumulative_predict_logistic(X, y, years, films, min_train_years=20):
    """
    Train Logistic Regression models cumulatively for each year.
    
    Parameters:
    - X: feature matrix
    - y: target variable
    - years: year labels
    - films: film names
    - min_train_years: minimum number of years needed for training
    
    Returns:
    - DataFrame with predictions
    """
    results = []
    unique_years = sorted(years.unique())
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    for i, test_year in enumerate(unique_years):
        if i < min_train_years:
            continue
        
        train_mask = years < test_year
        test_mask = years == test_year
        
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        
        X_train, y_train = X_scaled[train_mask], y[train_mask]
        X_test, y_test = X_scaled[test_mask], y[test_mask]
        
        logistic_model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        
        logistic_model.fit(X_train, y_train)
        y_pred_proba = logistic_model.predict_proba(X_test)[:, 1]
        predicted_winner_idx = y_pred_proba.argmax()
        
        for j, (film_name, actual_winner, pred_prob) in enumerate(zip(
            films[test_mask], y_test, y_pred_proba)):
            
            results.append({
                'Year': test_year,
                'Film': film_name,
                'Actual_Winner': actual_winner,
                'Logistic_Probability': pred_prob,
                'Logistic_Predicted_Winner': (j == predicted_winner_idx),
                'Training_Size': train_mask.sum()
            })
    
    return pd.DataFrame(results)


def calculate_betting_proportions(df):
    """
    Calculate betting proportions based on fair odds.
    
    Parameters:
    - df: DataFrame with fair_odds column
    
    Returns:
    - DataFrame with implied_probability column
    """
    df = df.copy()
    df['implied_probability'] = 1 / df['fair_odds']
    return df


def add_belief_heterogeneity(df, lambda_param=0.14, noise_scale=0.05, n_bettors=1000):
    """
    Add belief heterogeneity using Tukey-Lambda distribution.
    Simulates a pool of bettors per year that distribute their bets.
    
    Parameters:
    - df: DataFrame with fair_odds and implied_probability
    - lambda_param: shape parameter of Tukey-Lambda (0.14 ~ normal)
    - noise_scale: scale of noise to add
    - n_bettors: number of simulated bettors per year
    
    Returns:
    - DataFrame with belief heterogeneity columns
    """
    df = df.copy()
    df = calculate_betting_proportions(df)
    
    np.random.seed(42)
    
    belief_means = []
    belief_stds = []
    prop_active_bettors = []
    
    for year in df['Year'].unique():
        year_mask = df['Year'] == year
        year_films = df[year_mask].copy()
        n_films = len(year_films)
        
        beliefs_matrix = np.zeros((n_bettors, n_films))
        
        for film_idx, (idx, row) in enumerate(year_films.iterrows()):
            center = row['implied_probability']
            tukey_samples = stats.tukeylambda.rvs(lambda_param, size=n_bettors)
            beliefs = center + noise_scale * tukey_samples
            beliefs = np.clip(beliefs, 0, 1)
            beliefs_matrix[:, film_idx] = beliefs
        
        odds_array = year_films['fair_odds'].values
        expected_returns = beliefs_matrix * odds_array - 1
        
        best_choices = np.argmax(expected_returns, axis=1)
        has_positive_return = np.max(expected_returns, axis=1) > 0
        
        bets_per_film = np.zeros(n_films)
        for bettor_idx in range(n_bettors):
            if has_positive_return[bettor_idx]:
                chosen_film = best_choices[bettor_idx]
                bets_per_film[chosen_film] += 1
        
        total_active_bettors = bets_per_film.sum()
        if total_active_bettors > 0:
            film_proportions = bets_per_film / total_active_bettors
        else:
            film_proportions = np.zeros(n_films)
        
        for film_idx, (idx, row) in enumerate(year_films.iterrows()):
            belief_means.append(beliefs_matrix[:, film_idx].mean())
            belief_stds.append(beliefs_matrix[:, film_idx].std())
            prop_active_bettors.append(film_proportions[film_idx])
    
    df['belief_mean'] = belief_means
    df['belief_std'] = belief_stds
    df['prop_active_bettors'] = prop_active_bettors
    
    return df


def get_predictions(years_back=5, min_train_years=20, data_path=None):
    """
    Main entry point: train the model and return predictions.
    
    Parameters:
    - years_back: Number of recent years to return predictions for
    - min_train_years: Minimum years for training
    - data_path: Path to data file (optional)
    
    Returns:
    - DataFrame with predictions and betting analysis columns
    """
    # Load and preprocess data
    df = load_and_preprocess_data(data_path)
    
    # Convert target
    df['is_winner'] = df['is_winner'].astype(int)
    
    # Create director features
    df = create_director_features(df)
    
    # Prepare features
    df, available_features = prepare_features(df)
    
    # Filter and sort data
    df = df[df['year'] > 1946]
    df = df.sort_values('year').reset_index(drop=True)
    
    X = df[available_features]
    y = df['is_winner']
    years = df['year']
    films = df['Film_wiki']
    
    # Run cumulative prediction
    results = cumulative_predict_logistic(X, y, years, films, min_train_years)
    
    # Get last N years
    max_year = results['Year'].max()
    min_year = max_year - years_back + 1
    predictions = results[results['Year'] >= min_year].copy()
    
    # Normalize probabilities per year
    predictions['Logistic_Probability'] = predictions.groupby('Year')['Logistic_Probability'].transform(
        lambda x: x / x.sum()
    )
    
    # Calculate fair odds
    predictions['fair_odds'] = 1 / predictions['Logistic_Probability']
    
    # Add belief heterogeneity
    predictions = add_belief_heterogeneity(predictions)
    
    return predictions


def calculate_bettors_with_elasticity(base_bettors, overround, elasticity=-0.7):
    """
    Calculate number of bettors given overround and elasticity.
    
    Parameters:
    - base_bettors: Base number of bettors at 0% overround
    - overround: Overround percentage (e.g., 0.1 for 10%)
    - elasticity: Price elasticity of demand (default -0.7)
    
    Returns:
    - Number of bettors
    """
    return int(base_bettors * (1 + elasticity * overround))


def calculate_betting_house_revenue(df, overround, total_bettors, bet_amount=10, fair_odds_col='fair_odds'):
    """
    Calculate actual betting house revenue for a given overround scenario.
    
    Parameters:
    - df: DataFrame with predictions and betting proportions
    - overround: Overround percentage (e.g., 0.10 for 10%)
    - total_bettors: Total number of bettors
    - bet_amount: Amount each bettor bets (default $10)
    - fair_odds_col: Column name containing fair odds values (default 'fair_odds')
    
    Returns:
    - DataFrame with revenue breakdown by year
    """
    results = []
    df = df.copy()
    
    # Calculate overrounded odds
    odds_col = f'{fair_odds_col}_{int(overround*100)}_overrounded'
    df[odds_col] = df[fair_odds_col] / (1 + overround)
    
    for year in sorted(df['Year'].unique()):
        year_data = df[df['Year'] == year].copy()
        total_stakes = total_bettors * bet_amount
        
        year_data['num_bettors'] = year_data['prop_active_bettors'] * total_bettors
        year_data['stakes_on_film'] = year_data['num_bettors'] * bet_amount
        
        winner_mask = year_data['Actual_Winner'] == 1
        winner_data = year_data[winner_mask].iloc[0]
        
        payout = winner_data['stakes_on_film'] * winner_data[odds_col]
        revenue = total_stakes - payout
        profit_margin = (revenue / total_stakes) * 100
        
        results.append({
            'Year': int(year),
            'Total_Stakes': total_stakes,
            'Winner': winner_data['Film'],
            'Winner_Stakes': winner_data['stakes_on_film'],
            'Winner_Odds': winner_data[odds_col],
            'Payout': payout,
            'Revenue': revenue,
            'Profit_Margin_Pct': profit_margin
        })
    
    return pd.DataFrame(results)


def calculate_expected_revenue(df, overround, total_bettors, bet_amount=10, probability_col='Logistic_Probability', fair_odds_col='fair_odds'):
    """
    Calculate expected revenue based on probability predictions.
    
    Expected payout for each film = P(win) * stakes_on_film * odds
    Total expected payout = sum over all films
    Expected revenue = Total stakes - Total expected payout
    
    Parameters:
    - df: DataFrame with predictions and betting proportions
    - overround: Overround percentage
    - total_bettors: Total number of bettors
    - bet_amount: Amount each bettor bets
    - probability_col: Column name containing probability values (default 'Logistic_Probability')
    - fair_odds_col: Column name containing fair odds values (default 'fair_odds')
    
    Returns:
    - DataFrame with expected revenue breakdown by year
    """
    results = []
    df = df.copy()
    
    # Calculate overrounded odds from the specified fair odds column
    odds_col = f'{fair_odds_col}_{int(overround*100)}_overrounded'
    df[odds_col] = df[fair_odds_col] / (1 + overround)
    
    for year in sorted(df['Year'].unique()):
        year_data = df[df['Year'] == year].copy()
        total_stakes = total_bettors * bet_amount
        
        year_data['num_bettors'] = year_data['prop_active_bettors'] * total_bettors
        year_data['stakes_on_film'] = year_data['num_bettors'] * bet_amount
        
        # Calculate expected payout using the specified probability column
        year_data['expected_payout'] = (
            year_data[probability_col] * 
            year_data['stakes_on_film'] * 
            year_data[odds_col]
        )
        
        total_expected_payout = year_data['expected_payout'].sum()
        expected_revenue = total_stakes - total_expected_payout
        profit_margin = (expected_revenue / total_stakes) * 100
        
        results.append({
            'Year': int(year),
            'Total_Stakes': total_stakes,
            'Expected_Payout': total_expected_payout,
            'Expected_Revenue': expected_revenue,
            'Expected_Profit_Margin_Pct': profit_margin
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test the module
    print("Loading and training Oscar prediction model...")
    predictions = get_predictions(years_back=5)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"\nYears covered: {sorted(predictions['Year'].unique())}")
    print(f"\nColumns: {list(predictions.columns)}")
    print(f"\nSample predictions:")
    print(predictions[['Year', 'Film', 'Actual_Winner', 'Logistic_Probability', 'fair_odds']].tail(10))

