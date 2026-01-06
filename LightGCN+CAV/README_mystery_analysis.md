# Mystery Genre Preference Analysis

This README explains how to use the provided scripts to analyze Mystery genre preferences by occupation in the MovieLens dataset.

## Problem Description

The original code in `hw3_explore.ipynb` had an error when analyzing Mystery genre preferences by occupation. Specifically, there was a KeyError: 'nm_users' when trying to access user counts during visualization.

## Solution Files

This fix provides three main files:

1. `visualization_fix.py` - Contains corrected visualization functions
2. `mystery_genre_analysis.py` - A standalone script that demonstrates usage
3. `README_mystery_analysis.md` - This documentation file

## Issue Fixed

The main issue was a key name mismatch. The code was trying to access a key named 'nm_users' when the actual key was 'num_users'. Additionally, there was a typo in the variable name 'conceupt' instead of 'concept'.

## How to Use

### Option 1: Use the standalone script

The simplest way to run the analysis is to use the standalone script:

```bash
python mystery_genre_analysis.py
```

This script will:
1. Load and process the MovieLens data
2. Load the pre-trained LightGCN model
3. Train Concept Activation Vectors (CAVs) for all genres
4. Visualize correlations between genres and user ratings
5. Specifically analyze Mystery genre preferences by occupation
6. Compare embedding-based preferences with actual ratings

### Option 2: Import the visualization functions

You can also import the visualization functions in your own code:

```python
from visualization_fix import (
    visualize_genre_correlations,
    visualize_mystery_by_occupation,
    analyze_mystery_ratings_by_occupation,
    compare_mystery_preference_and_ratings
)

# Then call them with your data
visualize_mystery_by_occupation(processor, proj_scores)
```

### Option 3: Fix the notebook directly

To fix the original notebook:

1. Find the code cell with the KeyError
2. Replace:
   ```python
   for conceupt, result in sorted_results:
       if not np.isnan(result['correlation']):
           valid_concepts.append(concept)
           correlation_values.append(result['correlation'])
           p_values.append(result['p_value'])
           user_counts.append(result['nm_users'])
   ```

3. With:
   ```python
   for concept, result in sorted_results:
       if not np.isnan(result['correlation']):
           valid_concepts.append(concept)
           correlation_values.append(result['correlation'])
           p_values.append(result['p_value'])
           user_counts.append(result['num_users'])
   ```

## Visualization Functions

The `visualization_fix.py` script provides four main functions:

1. `visualize_genre_correlations(sorted_results)` - Creates a bubble chart showing correlations between genre preferences and ratings

2. `visualize_mystery_by_occupation(processor, proj_scores)` - Shows Mystery genre preference by occupation based on embeddings

3. `analyze_mystery_ratings_by_occupation(processor)` - Analyzes actual ratings for Mystery movies by occupation

4. `compare_mystery_preference_and_ratings(processor, proj_scores)` - Compares embedding-based preferences with actual ratings

## Data Requirements

The script expects the MovieLens-1M dataset with:
- `ratings.dat` - User-movie ratings
- `movies.dat` - Movie information including genres
- `users.dat` - User demographics including occupation

The script also requires a pre-trained LightGCN model saved as `MovieLens_LightGCN.pth`.

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib
- SciPy
- Custom modules: `MovieDataProcessor`, `LightGCN`, `RecommenderCAV` 