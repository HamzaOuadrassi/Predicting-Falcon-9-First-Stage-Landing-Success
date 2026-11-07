# Predicting Falcon 9 First Stage Landing Success

Capstone project of the **IBM Applied Data Science** course (Coursera). I collected SpaceX launch data, analyzed it with Python and SQL, built an interactive dashboard, and trained classification models to predict whether the Falcon 9 first stage lands successfully.

![Project overview](Predicting%20Falcon%209%20First%20Stage%20Landing%20Success.png)

> **Zusammenfassung (Deutsch):** In diesem Projekt habe ich Startdaten von SpaceX über eine API und per Web Scraping gesammelt, bereinigt und ausgewertet, ein interaktives Dashboard mit Plotly Dash erstellt und Klassifikationsmodelle trainiert, die vorhersagen, ob die erste Raketenstufe erfolgreich landet. Logistische Regression, SVM und KNN erreichten etwa 84 % Genauigkeit auf einem kleinen Testdatensatz von 18 Starts.

## Problem

SpaceX saves money when the first stage of a Falcon 9 lands and can be reused. Whether a landing succeeds depends on factors such as payload mass, launch site, orbit type and booster version. The question of this project: **can a machine learning model predict the landing outcome from mission data?**

## Data

| Source | Used for |
|---|---|
| SpaceX REST API | Launch records: booster versions, launch dates, payload mass, orbit, landing outcome |
| SpaceX launch history on Wikipedia (BeautifulSoup) | Additional details the API does not provide: booster versions, landing pad, mission data |

After cleaning and merging, the dataset contains about 90 launches. Features include launch site, payload mass, orbit type, flight number, booster version, reuse information, and the landing outcome (the target: 0 = failure, 1 = success).

## Method

1. **Data collection** via API and web scraping.
2. **Data wrangling**: removed missing values and irrelevant columns, converted categorical variables (orbit, launch site, landing outcome) to numeric form, created the landing success flag, merged both sources.
3. **SQL analysis** with SQLite.
4. **Exploratory data analysis and visualization**: scatter plots, bar charts, pie charts, correlation heatmap and box plots.
5. **Interactive dashboard** with Plotly Dash: launch-site dropdown, success pie chart, payload range slider and success scatter plot.
6. **Machine learning**: feature selection, `StandardScaler`, 80/20 train/test split, hyperparameter tuning with `GridSearchCV`, evaluation with accuracy, confusion matrix, precision and recall.

## Results

| Model | Test accuracy |
|---|---|
| Logistic Regression | about 84 % |
| Support Vector Machine | about 84 % |
| K-Nearest Neighbors | about 84 % |
| Decision Tree | about 67 % |

Observations from the exploratory analysis (from the plots, not from a statistical test):
- Launches with higher payload mass tended to have a lower landing success rate.
- The launch site and the orbit type were associated with differences in success rate.
- Later flights succeeded more often than early flights.

For the SVM, the linear kernel performed best; the RBF and sigmoid kernels performed worse.

### Limitations

- The test set has only **18 launches**, so one launch changes the accuracy by about 5.6 percentage points. The three models that tie at about 84 % cannot be separated reliably, and the lower decision tree score (about 67 %) may also be a result of this small sample.
- The dataset is small and covers a limited period.
- Accuracy alone does not show how well failures are detected; check the confusion matrix in the notebook.

## Repository contents

| File | Content |
|---|---|
| `jupyter-labs-spacex-data-collection-api-v2.ipynb` | Data collection from the SpaceX REST API |
| `jupyter-labs-webscraping.ipynb` | Web scraping with BeautifulSoup |
| `labs-jupyter-spacex-Data wrangling-v2.ipynb` | Data cleaning and feature preparation |
| `jupyter-labs-eda-sql-coursera_sqllite.ipynb` | Exploratory analysis with SQL (SQLite) |
| `jupyter-labs-eda-dataviz-v2.ipynb` | Exploratory analysis and visualizations |
| `SpaceX-Machine-Learning-Prediction-Part-5-v1.ipynb` | Model training, tuning and evaluation |
| `spacex-dash-app.py` | Plotly Dash dashboard |
| `Data_Piplines.md` | Notes on the data pipeline |

Run the notebooks in the order shown above.

## How to run

```bash
pip install pandas numpy requests beautifulsoup4 matplotlib seaborn scikit-learn folium plotly dash jupyter
jupyter notebook          # open and run the notebooks in order
python spacex-dash-app.py # starts the dashboard, then open http://127.0.0.1:8050
```

## Possible improvements

- Add more recent Falcon 9 missions to enlarge the dataset and the test set.
- Add features such as weather conditions and booster age.
- Try Random Forest, XGBoost or ensemble models, and use cross-validated scores instead of a single small test set.
- Extend the dashboard with orbit and booster reuse filters.

## Author

Hamza El Khrouf Ouadrassi
[LinkedIn](https://www.linkedin.com/in/hamza-el-khrouf-ouadrassi-117a99352/) | [GitHub](https://github.com/HamzaOuadrassi)
