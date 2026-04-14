# Online Shoppers Purchase Intention Dashboard

An interactive Streamlit dashboard for data mining on online shopping sessions.
This app supports:
- Exploratory data analysis
- Purchase prediction with XGBoost
- High-potential customer filtering
- Customer segmentation clustering
- Association rule mining
- AI-powered business Q&A

## Main Application

The main app file is app.py.

## Features and Functions in app.py

### 1) Data Loading and Preparation

- load_data()
	- Reads online_shoppers_intention.csv
	- Removes duplicate rows
- prepare_encoded_data(df)
	- Encodes Month into ordered numeric values
	- Encodes VisitorType
	- Converts Weekend and Revenue to integer

### 2) Classification Model

- train_xgboost(df_encoded)
	- Splits data into train and test sets
	- Scales features with StandardScaler
	- Handles class imbalance using SMOTE
	- Trains XGBClassifier
	- Returns model, evaluation metrics, confusion matrix, and feature importance

Classification page output includes:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- High-potential customer detection with probability threshold slider
- CSV export of high-potential sessions

### 3) Clustering Analysis

- perform_clustering(df)
	- Uses KMeans with ProductRelated_Duration and BounceRates
	- Computes elbow values (WCSS)
	- Creates 2 clusters and evaluates with Adjusted Rand Index

Clustering page also supports upload of high-potential customer CSV and groups sessions into:
- High Intent Buyers
- Returning Buyers
- Window Shoppers

### 4) Association Rule Mining

- perform_association_rules(df)
- perform_association_rules_custom(data)
	- Builds binary transaction-like features
	- Generates frequent itemsets with Apriori
	- Generates association rules
	- Filters rules where consequent is Is_Revenue

Association Rules page shows:
- Top rules leading to purchases
- Support, confidence, and lift comparisons
- Feature ranking style interpretation

### 5) AI Business Analyst

The AI Analyst page allows natural-language questions about the dataset.
Flow:
- User asks question in chat
- PandasAI generates a data-grounded response
- LLM produces concise business interpretation

## Screenshots Section (Add Your Function Screenshots Here)

Create a folder named screenshots in your project root, then save image files and reference them like below.

### Home / Overview
![Data Overview](screenshots/data-overview.png)

### Classification Dashboard
![Classification Metrics](screenshots/classification-dashboard.png)

### High Potential Customers
![High Potential Customers](screenshots/high-potential-customers.png)

### Clustering Result
![Clustering](screenshots/clustering-dashboard.png)

### Association Rules
![Association Rules](screenshots/association-rules.png)

### AI Analyst Chat
![AI Analyst](screenshots/ai-analyst.png)

## How to Run

1. Activate your virtual environment.
2. Install dependencies:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn mlxtend pandasai litellm pandasai-litellm
```

3. Run the app:

```bash
streamlit run app.py
```

## Notes

- Do not name your script streamlit.py because it conflicts with the Streamlit package import.
- If you see imbalanced-learn and scikit-learn compatibility errors, upgrade both packages together.