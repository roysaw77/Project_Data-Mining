import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import pandasai as pai
from pandasai_litellm.litellm import LiteLLM
from litellm import completion
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from mlxtend.frequent_patterns import apriori, association_rules

# Page configuration
st.set_page_config(
    page_title="Online Shoppers Purchase Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .insight-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING & PREPROCESSING ====================
@st.cache_data
def load_data():
    df = pd.read_csv("online_shoppers_intention.csv")
    df = df.drop_duplicates().reset_index(drop=True)
    return df

@st.cache_data
def prepare_encoded_data(df):
    """Prepare encoded data for modeling with highly correlated features dropped"""
    df_encoded = df.copy()
    
    # Drop highly correlated features (correlation > 0.85)
    # BounceRates <-> ExitRates: 0.90 (drop BounceRates, keep ExitRates - higher Revenue corr)
    # ProductRelated <-> ProductRelated_Duration: 0.86 (drop ProductRelated_Duration, keep ProductRelated)
    features_to_drop = ['BounceRates', 'ProductRelated_Duration']
    df_encoded = df_encoded.drop(columns=features_to_drop)
    
    # Encode Month
    month_order = {'Feb': 1, 'Mar': 2, 'May': 3, 'June': 4, 'Jul': 5, 
                   'Aug': 6, 'Sep': 7, 'Oct': 8, 'Nov': 9, 'Dec': 10}
    df_encoded['Month'] = df_encoded['Month'].map(month_order)
    
    # VisitorType
    le = LabelEncoder()
    df_encoded['VisitorType'] = le.fit_transform(df_encoded['VisitorType'])
    
    # Weekend & Revenue
    df_encoded['Weekend'] = df_encoded['Weekend'].astype(int)
    df_encoded['Revenue'] = df_encoded['Revenue'].astype(int)
    
    return df_encoded

@st.cache_resource
def train_xgboost(df_encoded):
    """Train XGBoost model with best parameters and return it with the scaler"""
    X = df_encoded.drop('Revenue', axis=1)
    y = df_encoded['Revenue']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                          random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    # Train XGBoost with best parameters from GridSearchCV
    model = XGBClassifier(
        n_estimators=50,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=1.0,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Confusion matrix and feature importance
    cm = confusion_matrix(y_test, y_pred)
    feature_importance = model.feature_importances_
    
    return model, results, scaler, X.columns.tolist(), cm, feature_importance, X_test_scaled, y_test

@st.cache_data
def perform_clustering(df):
    """Perform K-Means clustering"""
    x = df[['ProductRelated_Duration', 'BounceRates']].values
    
    # Elbow method
    wcss = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        km.fit(x)
        wcss.append(km.inertia_)
    
    # Final clustering with 2 clusters
    km = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_means = km.fit_predict(x)
    
    # Evaluate clustering
    le = LabelEncoder()
    labels_true = le.fit_transform(df['Revenue'])
    ari_score = adjusted_rand_score(labels_true, y_means)
    cm = confusion_matrix(labels_true, y_means)
    
    return x, y_means, km.cluster_centers_, wcss, ari_score, cm

@st.cache_data
def perform_association_rules(df):
    """Perform Association Rule Mining"""
    arm_df = pd.DataFrame()
    
    # Discretize features
    arm_df['High_PageValue'] = df['PageValues'] > df['PageValues'].median()
    arm_df['High_ExitRate'] = df['ExitRates'] > df['ExitRates'].median()
    arm_df['Is_Revenue'] = df['Revenue']
    arm_df['Is_Weekend'] = df['Weekend']
    arm_df['Is_Returning_Visitor'] = df['VisitorType'] == 'Returning_Visitor'
    arm_df['Is_New_Visitor'] = df['VisitorType'] == 'New_Visitor'
    
    # Add Month (One-hot encoding)
    month_dummies = pd.get_dummies(df['Month'], prefix='Month')
    arm_df = pd.concat([arm_df, month_dummies], axis=1)
    arm_df = arm_df.astype(bool)
    
    # Generate frequent itemsets and rules
    frequent_itemsets = apriori(arm_df, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    
    # Filter for rules where consequent is ONLY Is_Revenue (avoid duplicates)
    revenue_rules = rules[rules['consequents'].apply(lambda x: x == frozenset({'Is_Revenue'}))]
    revenue_rules = revenue_rules.sort_values(by='lift', ascending=False)
    
    return frequent_itemsets, rules, revenue_rules

# Load data
df = load_data()
df_encoded = prepare_encoded_data(df)
xgb_model, model_results, scaler, feature_columns, confusion_matrix_result, feature_importance, X_test_scaled, y_test = train_xgboost(df_encoded)

# ==================== SIDEBAR ====================
st.sidebar.markdown("## 🛒 Shopping Analytics")
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Select a Page:",
    ["📊 Data Overview", "🤖 Classification Dashboard", "🎯 Clustering", "🔗 Association Rules", "💬 AI Analyst"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Quick Stats")
st.sidebar.metric("Total Sessions", f"{len(df):,}")
st.sidebar.metric("Purchase Rate", f"{(df['Revenue'].sum() / len(df)) * 100:.1f}%")
st.sidebar.metric("Model", "XGBoost")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use this dashboard to understand customer behavior and predict purchase intentions.")

# ==================== MAIN TITLE ====================
st.markdown('<p class="main-header">🛒 Online Shoppers Purchase Intention Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">A Data Mining Dashboard for Business Managers & Data Analysts</p>', unsafe_allow_html=True)

# ==================== PAGE 1: Data Overview ====================
if page == "📊 Data Overview":
    st.header("📊 Data Overview & Exploratory Analysis")
    st.markdown("*Understand your customer data at a glance*")
    
    # Key Metrics Row
    st.subheader("📈 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📋 Total Sessions", f"{len(df):,}")
    with col2:
        st.metric("🔢 Features", f"{df.shape[1]}")
    with col3:
        purchase_rate = (df['Revenue'].sum() / len(df)) * 100
        st.metric("✅ Purchase Rate", f"{purchase_rate:.1f}%")
    with col4:
        st.metric("❌ Non-Purchase Rate", f"{100 - purchase_rate:.1f}%")
    with col5:
        st.metric("👥 Returning Visitors", f"{(df['VisitorType'] == 'Returning_Visitor').sum():,}")
    
    st.markdown("---")
    
    # Data Quality Section
    st.subheader("🔍 Data Quality Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Missing Values Analysis**")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            st.success("✅ No missing values found in the dataset!")
        else:
            st.dataframe(missing[missing > 0])
    
    with col2:
        st.markdown("**Dataset Sample**")
        st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    
    # Target Distribution
    st.subheader("🎯 Target Variable Distribution")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        target_counts = df['Revenue'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#FF6B6B", "#4ECDC4"]
        bars = ax.bar(["No Purchase", "Purchase"], target_counts.values, 
                     color=colors, edgecolor='white', linewidth=2)
        ax.bar_label(bars, labels=[f"{v:,}\n({v/len(df)*100:.1f}%)" for v in target_counts.values], fontsize=12, fontweight='bold')
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Distribution of Purchase Intention", fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, max(target_counts.values) * 1.2)
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>💡 Business Insight</h4>
        <p>The dataset shows a <strong>significant class imbalance</strong>:</p>
        <ul>
            <li>~84% of sessions do NOT result in purchase</li>
            <li>Only ~16% of sessions lead to actual purchases</li>
        </ul>
        <p><strong>Implication:</strong> We use SMOTE (Synthetic Minority Oversampling) to balance the training data for better model performance.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature Categories
    st.subheader("📊 Feature Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📈 Numerical Features", "📋 Categorical Features", "🔥 Correlation Matrix"])
    
    with tab1:
        numerical_features = ['Administrative', 'Administrative_Duration', 'Informational', 
                              'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                              'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']
        
        selected_feature = st.selectbox("Select a numerical feature to visualize:", numerical_features)
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(df[selected_feature], bins=50, color='#4ECDC4', edgecolor='white', alpha=0.8)
            ax.set_title(f'Distribution of {selected_feature}', fontsize=14, fontweight='bold')
            ax.set_xlabel(selected_feature, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
        
        with col2:
            st.markdown(f"**Statistics for {selected_feature}:**")
            stats_df = df[selected_feature].describe().to_frame().T
            st.dataframe(stats_df, use_container_width=True)
            
            st.markdown(f"""
            **Key Observations:**
            - Mean: {df[selected_feature].mean():.2f}
            - Median: {df[selected_feature].median():.2f}
            - Std Dev: {df[selected_feature].std():.2f}
            """)
    
    with tab2:
        cat_columns = ['Month', 'VisitorType', 'Weekend', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']
        selected_cat = st.selectbox("Select a categorical feature:", cat_columns)
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            df[selected_cat].value_counts().plot(kind='bar', color='#667eea', edgecolor='white', ax=ax)
            ax.set_title(f'Distribution of {selected_cat}', fontsize=14, fontweight='bold')
            ax.set_xlabel(selected_cat, fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
        
        with col2:
            # Conversion rate by category
            st.markdown(f"**Conversion Rate by {selected_cat}:**")
            conversion = df.groupby(selected_cat)['Revenue'].mean() * 100
            conversion_df = conversion.reset_index()
            conversion_df.columns = [selected_cat, 'Conversion Rate (%)']
            conversion_df = conversion_df.sort_values('Conversion Rate (%)', ascending=False)
            st.dataframe(conversion_df, use_container_width=True)
    
    with tab3:
        fig, ax = plt.subplots(figsize=(12, 10))
        correlation_matrix = df[numerical_features + ['Revenue']].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                    linewidths=0.5, square=True, ax=ax, center=0)
        ax.set_title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        
        st.markdown("""
        <div class="insight-box">
        <h4>💡 Correlation Insights</h4>
        <ul>
            <li><strong>PageValues</strong> has the strongest positive correlation with Revenue</li>
            <li><strong>BounceRates</strong> and <strong>ExitRates</strong> are negatively correlated with purchases</li>
            <li>Duration features are highly correlated with their respective page count features</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ==================== PAGE 2: Classification ====================
elif page == "🤖 Classification Dashboard":
    st.header("🤖 XGBoost Classification Model")
    st.markdown("*Predict whether a customer will make a purchase using XGBoost*")
    
    # Model Info Box
    st.info("""🚀 **Model:** XGBoost (Extreme Gradient Boosting)
    
**Optimized Hyperparameters (from GridSearchCV with SMOTE):**
- `n_estimators`: 50 | `max_depth`: 7 | `learning_rate`: 0.1 | `subsample`: 0.8 | `colsample_bytree`: 1.0""")
    
    st.markdown("---")
    
    # Model Performance Metrics
    st.subheader("📊 XGBoost Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("🎯 Accuracy", f"{model_results['Accuracy']:.4f}")
    with col2:
        st.metric("✅ Precision", f"{model_results['Precision']:.4f}")
    with col3:
        st.metric("🔍 Recall", f"{model_results['Recall']:.4f}")
    with col4:
        st.metric("⚖️ F1-Score", f"{model_results['F1-Score']:.4f}")
    with col5:
        st.metric("📈 ROC-AUC", f"{model_results['ROC-AUC']:.4f}")
    
    st.markdown("---")
  
    # Create importance dataframe sorted by importance 
    importance_data = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    # Get top 10 features
    top_10 = importance_data.head(10)
    
    
    # ==================== HIGH POTENTIAL CUSTOMERS DETECTION ====================
    st.subheader("🎯 High Potential Customers Detection")
    st.markdown("*Identify sessions with high purchase probability from your own dataset*")
    
    # File uploader for custom CSV
    st.markdown("#### 📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with customer session data",
        type=['csv'],
        help="Upload your own CSV file with the same columns as the original dataset to predict purchase probability for each session."
    )
    
    if uploaded_file is not None:
        try:
            # Load uploaded data
            df_upload = pd.read_csv(uploaded_file)
            df_upload = df_upload.drop_duplicates().reset_index(drop=True)
            
            # Prepare encoded data for the uploaded file
            df_upload_encoded = df_upload.copy()
            
            # Drop highly correlated features
            features_to_drop = ['BounceRates', 'ProductRelated_Duration']
            df_upload_encoded = df_upload_encoded.drop(columns=[f for f in features_to_drop if f in df_upload_encoded.columns])
            
            # Encode Month
            month_order = {'Feb': 1, 'Mar': 2, 'May': 3, 'June': 4, 'Jul': 5, 
                           'Aug': 6, 'Sep': 7, 'Oct': 8, 'Nov': 9, 'Dec': 10}
            if 'Month' in df_upload_encoded.columns:
                df_upload_encoded['Month'] = df_upload_encoded['Month'].map(month_order)
            
            # VisitorType
            if 'VisitorType' in df_upload_encoded.columns:
                le = LabelEncoder()
                df_upload_encoded['VisitorType'] = le.fit_transform(df_upload_encoded['VisitorType'])
            
            # Weekend & Revenue
            if 'Weekend' in df_upload_encoded.columns:
                df_upload_encoded['Weekend'] = df_upload_encoded['Weekend'].astype(int)
            if 'Revenue' in df_upload_encoded.columns:
                df_upload_encoded['Revenue'] = df_upload_encoded['Revenue'].astype(int)
            
            st.success(f"✅ Successfully loaded {len(df_upload):,} sessions from your file!")
            
            # Use uploaded data for prediction
            df_for_prediction = df_upload
            df_encoded_for_prediction = df_upload_encoded
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("Using the default dataset instead.")
            df_for_prediction = df
            df_encoded_for_prediction = df_encoded
    else:
        st.info("💡 **No file uploaded.** Using the built-in dataset for demonstration. Upload your own CSV to analyze your customer data!")
        df_for_prediction = df
        df_encoded_for_prediction = df_encoded
    
    st.markdown("---")
    
    # Prepare dataset for prediction
    df_predict = df_encoded_for_prediction.copy()
    
    # Check if Revenue column exists
    has_revenue = 'Revenue' in df_predict.columns
    
    if has_revenue:
        X_full = df_predict.drop('Revenue', axis=1)
        y_actual = df_predict['Revenue']
    else:
        X_full = df_predict.copy()
        y_actual = None
    
    # Scale the full dataset
    X_full_scaled = scaler.transform(X_full)
    
    # Predict probabilities for all sessions
    all_probabilities = xgb_model.predict_proba(X_full_scaled)[:, 1]
    all_predictions = xgb_model.predict(X_full_scaled)
    
    # Add predictions to dataframe
    df_with_predictions = df_for_prediction.copy()
    df_with_predictions['Purchase_Probability'] = all_probabilities
    df_with_predictions['Predicted_Purchase'] = all_predictions
    df_with_predictions['Session_ID'] = range(1, len(df_for_prediction) + 1)
    
    # Threshold explanation and selector
    st.markdown("#### 🎚️Probability Threshold?")
    st.markdown("""
    The **Probability Threshold** determines which customers are considered "high potential" buyers:
    
    - **Filtering:** Only sessions with probability **≥ threshold** are shown as high potential
 
    **💡 Choosing a threshold:**
    - **Higher threshold (80-95%):** Fewer customers, but **very likely** to buy 
    - **Lower threshold (50-60%):** More customers identified, but **less certain**
    """)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        threshold = st.slider(
            "🎚️ Select Threshold",
            min_value=0.5, max_value=0.95, value=0.7, step=0.05,
            help="Sessions with purchase probability above this threshold are considered high potential"
        )
    
    # Filter high potential customers
    high_potential = df_with_predictions[df_with_predictions['Purchase_Probability'] >= threshold].copy()
    high_potential = high_potential.sort_values('Purchase_Probability', ascending=False)
    
    with col2:
        # Metrics
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("🎯 High Potential", f"{len(high_potential):,}")
        with metric_cols[1]:
            st.metric("📊 Total Sessions", f"{len(df_for_prediction):,}")
        with metric_cols[2]:
            percentage = len(high_potential) / len(df_for_prediction) * 100 if len(df_for_prediction) > 0 else 0
            st.metric("📈 Percentage", f"{percentage:.1f}%")
        with metric_cols[3]:
            # How many of these actually purchased? (only if Revenue column exists)
            if len(high_potential) > 0 and 'Revenue' in high_potential.columns:
                actual_buyers = high_potential['Revenue'].sum()
                accuracy = actual_buyers / len(high_potential) * 100
                st.metric("✅ Actual Buyers", f"{accuracy:.1f}%")
            else:
                st.metric("✅ Actual Buyers", "N/A")
    
    st.markdown("---")
    
    if len(high_potential) > 0:
        # Display high potential customers list
        st.markdown(f"### 📋 High Potential Customer List (Top {min(50, len(high_potential))} Sessions)")
        
        # Select columns to display
        display_cols = ['Session_ID', 'Purchase_Probability', 'Revenue', 'PageValues', 'Month', 
                        'ExitRates', 'ProductRelated', 'Administrative', 'VisitorType', 'TrafficType']
        
        # Filter available columns
        available_cols = [col for col in display_cols if col in high_potential.columns]
        
        # Create display dataframe
        display_df = high_potential[available_cols].head(50).copy()
        display_df['Purchase_Probability'] = display_df['Purchase_Probability'].apply(lambda x: f"{x*100:.1f}%")
        if 'Revenue' in display_df.columns:
            display_df['Revenue'] = display_df['Revenue'].apply(lambda x: "✅ Yes" if x else "❌ No")
        
        # Rename for better display
        rename_cols = {
            'Session_ID': 'Session #',
            'Purchase_Probability': 'Buy Probability',
            'PageValues': 'Page Value',
            'ExitRates': 'Exit Rate',
            'ProductRelated': 'Product Pages',
            'Administrative': 'Admin Pages',
            'VisitorType': 'Visitor Type',
            'TrafficType': 'Traffic Type'
        }
        if 'Revenue' in display_df.columns:
            rename_cols['Revenue'] = 'Actually Bought'
        
        display_df = display_df.rename(columns=rename_cols)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Export functionality
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            # Prepare export data - only include columns that exist
            export_cols = ['Session_ID', 'Purchase_Probability']
            optional_cols = ['Revenue', 'PageValues', 'Month', 'VisitorType', 'TrafficType', 'Region']
            export_cols.extend([c for c in optional_cols if c in high_potential.columns])
            
            export_df = high_potential[export_cols].copy()
            export_df['Purchase_Probability'] = export_df['Purchase_Probability'].apply(lambda x: f"{x*100:.2f}%")
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download High Potential Customers (CSV)",
                data=csv,
                file_name=f"high_potential_customers_threshold_{int(threshold*100)}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Build summary statistics dynamically
            summary_lines = [
                f"- **Threshold:** {threshold*100:.0f}% probability",
                f"- **Identified:** {len(high_potential):,} high potential sessions"
            ]
            if 'Revenue' in high_potential.columns:
                actual_buyers = high_potential['Revenue'].sum()
                summary_lines.append(f"- **Actual Buyers:** {actual_buyers:,} ({actual_buyers/len(high_potential)*100:.1f}%)")
            if 'PageValues' in high_potential.columns:
                summary_lines.append(f"- **Avg Page Value:** {high_potential['PageValues'].mean():.2f}")
            
            st.markdown("**📊 Summary Statistics:**\n" + "\n".join(summary_lines))
        
      
    
    st.markdown("---")

# ==================== PAGE 3: Clustering ====================
elif page == "🎯 Clustering":
    st.header("🎯 Customer Segmentation (Clustering)")
    st.markdown("*Segment customers based on browsing behavior*")
    
    # Perform clustering
    x, y_means, centers, wcss, ari_score, cm = perform_clustering(df)
    
    # Clustering Overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Number of Clusters", "2")
    with col2:
        st.metric("📊 Adjusted Rand Index", f"{ari_score:.4f}")
    with col3:
        st.metric("📈 Total Data Points", f"{len(x):,}")
    
    st.markdown("---")
    
    # Elbow Method
    st.subheader("📉 Elbow Method (Optimal Clusters)")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, 11), wcss, 'bo-', linewidth=2, markersize=8)
        ax.axvline(x=2, color='red', linestyle='--', label='Optimal k=2')
        ax.set_title('The Elbow Method', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Clusters', fontsize=12)
        ax.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>💡 Elbow Method Explanation</h4>
        <p>The "elbow" in the curve indicates the optimal number of clusters where adding more clusters doesn't significantly reduce WCSS.</p>
        <p><strong>Result:</strong> k=2 is optimal for this dataset.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Clustering Visualization
    st.subheader("🔵 Customer Segments Visualization")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter1 = ax.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s=50, c='#FFD700', 
                             label='Uninterested Customers', alpha=0.6, edgecolor='white')
        scatter2 = ax.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s=50, c='#FF69B4', 
                             label='Target Customers', alpha=0.6, edgecolor='white')
        ax.scatter(centers[:, 0], centers[:, 1], s=200, c='blue', marker='X', 
                  label='Centroids', edgecolor='black', linewidth=2)
        ax.set_title('Customer Segmentation: ProductRelated Duration vs Bounce Rate', fontsize=14, fontweight='bold')
        ax.set_xlabel('ProductRelated Duration', fontsize=12)
        ax.set_ylabel('Bounce Rates', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        ### 📊 Cluster Interpretation
        
        **🟡 Yellow - Uninterested Customers:**
        - Lower product page duration
        - Higher bounce rates
        - Less engaged visitors
        
        **🩷 Pink - Target Customers:**
        - Higher product page duration
        - Lower bounce rates
        - More engaged & likely to purchase
        
        **🔵 Blue X - Centroids:**
        - Center points of each cluster
        """)
    
    st.markdown("---")
    
    # Clustering Evaluation
    st.subheader("📋 Clustering Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Cluster 0', 'Cluster 1'],
                    yticklabels=['No Purchase', 'Purchase'])
        ax.set_title('Clustering vs Actual Purchase (Confusion Matrix)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Actual Revenue', fontsize=12)
        ax.set_xlabel('Predicted Cluster', fontsize=12)
        st.pyplot(fig)
    
    with col2:
        st.markdown(f"""
        <div class="warning-box">
        <h4>⚠️ Clustering Limitation</h4>
        <p><strong>Adjusted Rand Index: {ari_score:.4f}</strong></p>
        <p>The low ARI score indicates that clustering based only on ProductRelated_Duration and BounceRates doesn't align well with actual purchase behavior.</p>
        <p><strong>Key Findings:</strong></p>
        <ul>
            <li>Many purchasers were clustered as "uninterested"</li>
            <li>High bounce rate doesn't always mean no purchase</li>
            <li>Additional features needed for better segmentation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Business Insights for Clustering
    st.subheader("💼 Business Insights from Customer Segmentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 What This Means for Your Business
        
        **Key Finding:** Customers who spend **more time on product pages** and have **lower bounce rates** 
        are significantly more engaged and likely to purchase.
        
        | Segment | Behavior | Business Value | Priority |
        |---------|----------|---------------|----------|
        | **Target Customers** | High engagement, low bounce | High conversion potential | 🔴 High |
        | **Uninterested Visitors** | Quick exits, high bounce | Low immediate value | 🟢 Low |
        
        ### 📊 Segment Size Analysis
        """)
        
        # Calculate segment sizes
        cluster_0_size = (y_means == 0).sum()
        cluster_1_size = (y_means == 1).sum()
        total = len(y_means)
        
        segment_data = {
            "Segment": ["Target Customers", "Uninterested Visitors"],
            "Count": [cluster_1_size, cluster_0_size],
            "Percentage": [f"{cluster_1_size/total*100:.1f}%", f"{cluster_0_size/total*100:.1f}%"]
        }
        st.dataframe(pd.DataFrame(segment_data), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("""
        ### 💡 Actionable Strategies
        
        **For Target Customers (Engaged):**
        - ✅ Prioritize for premium offers
        - ✅ Upselling and cross-selling campaigns
        - ✅ Loyalty program enrollment
        - ✅ Personalized product recommendations
        
        **For Uninterested Visitors (High Bounce):**
        - 🔧 Improve landing page relevance
        - 🔧 Faster page load times
        - 🔧 Compelling above-the-fold content
        - 🔧 Exit-intent offers to re-engage
        
        ### ⚠️ Important Caveat
        
        The clustering alone **should not be used for purchase prediction**. 
        As our analysis shows, many actual purchasers have high bounce rates. 
        
        **Recommendation:** Use clustering for **engagement segmentation** 
        but rely on **Classification models** for purchase prediction.
        """)
    
    st.markdown("""
    <div class="insight-box">
    <h4>🔑 Key Business Takeaway</h4>
    <p><strong>Engagement ≠ Purchase Intent</strong></p>
    <p>While engaged customers (high duration, low bounce) are valuable, our data shows that 
    purchase behavior is influenced by many more factors. Use this clustering to:</p>
    <ul>
        <li>Identify website UX issues (why do some visitors bounce immediately?)</li>
        <li>Segment for engagement-based marketing</li>
        <li>Complement with classification models for full purchase prediction</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ==================== PAGE 4: Association Rule Mining ====================
elif page == "🔗 Association Rules":
    st.header("🔗 Association Rule Mining")
    st.markdown("*Discover patterns that lead to purchases*")
    
    # ==================== FILE UPLOAD FOR CUSTOM DATA ====================
    st.subheader("📁 Upload Your Data for Feature Ranking")
    st.markdown("*Upload your own dataset to discover which features are most important for driving purchases*")
    
    uploaded_arm_file = st.file_uploader(
        "Upload a CSV file with customer session data",
        type=['csv'],
        key="arm_file_uploader",
        help="Upload your own CSV file to analyze feature importance based on association rules"
    )
    
    # Determine which dataset to use
    if uploaded_arm_file is not None:
        try:
            df_arm = pd.read_csv(uploaded_arm_file)
            df_arm = df_arm.drop_duplicates().reset_index(drop=True)
            st.success(f"✅ Successfully loaded {len(df_arm):,} sessions from your file!")
            use_uploaded_data = True
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("Using the default dataset instead.")
            df_arm = df
            use_uploaded_data = False
    else:
        st.info("💡 **No file uploaded.** Using the built-in dataset for demonstration.")
        df_arm = df
        use_uploaded_data = False
    
    st.markdown("---")
    
    # Perform ARM on the selected dataset
    @st.cache_data
    def perform_association_rules_custom(data):
        """Perform Association Rule Mining on custom data"""
        arm_df = pd.DataFrame()
        
        # Discretize features
        if 'PageValues' in data.columns:
            arm_df['High_PageValue'] = data['PageValues'] > data['PageValues'].median()
        if 'ExitRates' in data.columns:
            arm_df['High_ExitRate'] = data['ExitRates'] > data['ExitRates'].median()
        if 'Revenue' in data.columns:
            arm_df['Is_Revenue'] = data['Revenue']
        if 'Weekend' in data.columns:
            arm_df['Is_Weekend'] = data['Weekend']
        if 'VisitorType' in data.columns:
            arm_df['Is_Returning_Visitor'] = data['VisitorType'] == 'Returning_Visitor'
            arm_df['Is_New_Visitor'] = data['VisitorType'] == 'New_Visitor'
        
        # Add Month (One-hot encoding)
        if 'Month' in data.columns:
            month_dummies = pd.get_dummies(data['Month'], prefix='Month')
            arm_df = pd.concat([arm_df, month_dummies], axis=1)
        arm_df = arm_df.astype(bool)
        
        # Generate frequent itemsets and rules
        frequent_itemsets = apriori(arm_df, min_support=0.05, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
        
        # Filter for rules where consequent is ONLY Is_Revenue
        revenue_rules = rules[rules['consequents'].apply(lambda x: x == frozenset({'Is_Revenue'}))]
        revenue_rules = revenue_rules.sort_values(by='lift', ascending=False)
        
        return frequent_itemsets, rules, revenue_rules
    
    # Perform ARM on the selected data
    frequent_itemsets, all_rules, revenue_rules = perform_association_rules_custom(df_arm)
    
    # Overview Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Frequent Itemsets", f"{len(frequent_itemsets):,}")
    with col2:
        st.metric("🔗 Total Rules", f"{len(all_rules):,}")
    with col3:
        st.metric("💰 Revenue Rules", f"{len(revenue_rules):,}")
    
    st.markdown("---")
    
    # Top Rules Leading to Purchase
    st.subheader("🏆 Top 4 Rules Leading to Purchases")
    
    if len(revenue_rules) > 0:
        # Get top 4 unique rules (drop duplicate antecedents, keep highest lift)
        top_4_rules = revenue_rules.drop_duplicates(subset=['antecedents'], keep='first').head(5).copy()
        
        # Create clean, readable rule labels
        top_4_rules['rule_label'] = top_4_rules['antecedents'].apply(
            lambda x: ' + '.join([str(item).replace('_', ' ').replace('Is ', '').replace('High ', 'High ') 
                                  for item in x])
        )
        top_4_rules['antecedents_str'] = top_4_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Grouped bar chart with Support, Confidence, and Lift
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Bar positions
            y_pos = range(len(top_4_rules))
            bar_height = 0.25
            
            # Create grouped bars
            bars1 = ax.barh([y - bar_height for y in y_pos], top_4_rules['support'], 
                            height=bar_height, color='#3498db', edgecolor='black', label='Support')
            bars2 = ax.barh(y_pos, top_4_rules['confidence'], 
                            height=bar_height, color='#2ecc71', edgecolor='black', label='Confidence')
            bars3 = ax.barh([y + bar_height for y in y_pos], top_4_rules['lift'] / top_4_rules['lift'].max(), 
                            height=bar_height, color='#e74c3c', edgecolor='black', label='Lift (normalized)')
            
            # Add value labels
            for i, (s, c, l) in enumerate(zip(top_4_rules['support'], top_4_rules['confidence'], top_4_rules['lift'])):
                ax.text(s + 0.01, i - bar_height, f'{s:.3f}', va='center', fontsize=9, color='#2c3e50')
                ax.text(c + 0.01, i, f'{c:.1%}', va='center', fontsize=9, color='#2c3e50')
                ax.text(l/top_4_rules['lift'].max() + 0.01, i + bar_height, f'{l:.2f}', va='center', fontsize=9, color='#2c3e50')
            
            # Styling
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_4_rules['rule_label'], fontsize=11, fontweight='bold')
            ax.set_xlabel('Metric Value', fontsize=12, fontweight='bold')
            ax.set_title('Top 4 Association Rules Leading to Purchases\n(Unique Antecedents Only)', 
                         fontsize=14, fontweight='bold', pad=15)
            ax.legend(loc='lower right', fontsize=10)
            ax.set_xlim(0, 1.15)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
            <h4>💡 Understanding the Metrics</h4>
            <p><strong>🔵 Support:</strong> How often this pattern appears in all transactions.</p>
            <p><strong>🟢 Confidence:</strong> When we see the antecedent, how often does purchase occur?</p>
            <p><strong>🔴 Lift:</strong> How much more likely is purchase compared to random chance?</p>
            <p><em>Lift > 1 means the pattern increases purchase probability!</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed Rules Table
        st.subheader("📋 Top 4 Rules - Detailed View")
        
        display_df = top_4_rules[['antecedents_str', 'support', 'confidence', 'lift']].copy()
        display_df.columns = ['Antecedents (IF)', 'Support', 'Confidence', 'Lift']
        display_df = display_df.reset_index(drop=True)
        display_df.index = display_df.index + 1
        
        st.dataframe(display_df.style.format({
            'Support': '{:.4f}',
            'Confidence': '{:.4f}',
            'Lift': '{:.4f}'
        }).background_gradient(subset=['Lift'], cmap='Greens'), use_container_width=True)

    else:
        st.warning("No association rules found leading to purchase with current parameters.")

# ==================== PAGE 5: AI Business Analyst ====================
elif page == "💬 AI Analyst":
    st.header("💬 AI Business Analyst")
    st.markdown("*Ask questions about your data in natural language*")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if question := st.chat_input("Ask a question about the dataset..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    # Setup LLM and DataFrame
                    llm = LiteLLM(
                        model="nvidia_nim/meta/llama3-70b-instruct",
                        api_key="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",
                        stream=False,
                    )
                    pai.config.set({"llm": llm})
                    df_shoppers = pd.read_csv("online_shoppers_intention.csv")
                    df_pai = pai.DataFrame(df_shoppers, config={"llm": llm})
                    
                    # Get pandasai response
                    pai_response = df_pai.chat(question)
                    
                    # Get detailed analysis
                    analysis = completion(
                        model="nvidia_nim/google/gemma-2-27b-it",
                        messages=[{"role": "user", "content": f"You are an expert business analyst. Dataset sample: {df_shoppers.head(5).to_dict()}. User question: {question}. Data answer: {pai_response}. Provide a concise business analysis."}],
                        api_key="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",
                    )
                    
                    response_text = analysis.choices[0].message.content
                    st.markdown(response_text)
                    
                    # Add to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p><strong>🛒 Online Shoppers Purchase Intention Analysis Dashboard</strong></p>
    <p>Built with Streamlit | Data Mining Project</p>
    <p>For Business Managers & Data Analysts</p>
</div>
""", unsafe_allow_html=True)
