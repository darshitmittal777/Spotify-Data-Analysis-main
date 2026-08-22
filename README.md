# Spotify / Billboard Data Analysis

A data science project analyzing Spotify song data and Billboard Year-End Hot 100 charts (2000-2018) to build:

1. **Song Recommender System** — two unsupervised approaches (nearest-neighbor by Euclidean distance, and k-means/hierarchical clustering) for recommending similar songs.
2. **Stream Count Prediction** — linear and ridge regression models predicting a song's number of Spotify streams from its audio features.
3. **Billboard Top 100 Classifier** — logistic/ridge/lasso classifiers predicting whether a song reaches the Billboard Year-End Hot 100 based solely on its audio characteristics ("Hit Song Science").

> **Note:** All R scripts have been converted to Jupyter notebooks (R kernel / `ir`) for consistency with the rest of the project. Original code and logic are unchanged — only the file format and section headers were added.

## Project Structure

```
.
├── Project_Report.pdf              Final write-up covering all three sub-projects
├── README.md
├── .gitattributes
│
├── Top100_Analysis_Codes/          Data pipeline: scrape -> merge -> feature-extract -> EDA -> classify
│   ├── 1_billboard_top100_scrape.ipynb        Scrape Billboard Year-End Hot 100 (2000-2018) from Wikipedia
│   ├── 2_combining_billboard_19000_datasets.ipynb   Merge Billboard + 19k-song dataset, create Top100 label
│   ├── 3_extracting_features_using_api.ipynb  Pull Spotify audio features via API (fuzzy-matched)
│   ├── eda_4.ipynb                            EDA on the fully-featured dataset
│   └── ML_top100.ipynb                        Logistic / Ridge / Lasso classifiers for Top100 prediction
│
├── Codes/                          Recommenders, stream-prediction, and supporting analysis
│   ├── billboard.csv
│   ├── stream_prediction.ipynb                Linear & Ridge regression for stream-count prediction
│   ├── Plots_Streams_prediction.ipynb         Diagnostic plots for stream prediction
│   ├── Recommender_model_1.ipynb              Nearest-neighbor (Euclidean distance) recommender
│   ├── Recommender_model_2.ipynb              Cluster-based (k-means/hierarchical) recommender
│   ├── Hypothesis_test.ipynb                  Chi-squared test: do same-artist songs cluster together?
│   └── ML_top100.ipynb                        (duplicate) Top100 classification models
│
├── Data/
│   ├── billboard.csv                Scraped Billboard Year-End Hot 100 (2000-2018)
│   ├── songs_complete_data.csv      Fully-featured dataset used for classification & EDA
│   └── spotify-2023.csv             Kaggle "Most Streamed Spotify Songs 2023" dataset
│
└── Images/                          Saved plots/figures referenced in the report
```

## Languages Used

- **Python** (`pandas`, `spotipy`, `scikit-learn`, `matplotlib`/`seaborn`): web scraping, Spotify API calls, data merging/cleaning, stream-count regression.
- **R** (`ggplot2`, `glmnet`, `caret`, `factoextra`, `ROCR`): statistical modeling, classification (logistic/ridge/lasso), clustering diagnostics, hypothesis testing, and most of the report's visualizations.

## Requirements

- **Python notebooks**: `pandas`, `numpy`, `spotipy`, `fuzzywuzzy`, `tqdm`, `scikit-learn`, `matplotlib`, `seaborn`
- **R notebooks** (run via the `IRkernel` in Jupyter, or in RStudio after stripping the notebook wrapper): `ggplot2`, `dplyr`, `stringi`, `factoextra`, `glmnet`, `caret`, `caTools`, `ROCR`, `pROC`, `car`, `vcd` (for `mosaic()`), `MASS`, `rpart`, `randomForest`, `gbm`
