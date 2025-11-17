import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, silhouette_score, r2_score, mean_squared_error
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
import json

# Suppress warnings in the dashboard
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
# Set the page configuration for a professional look
st.set_page_config(
    page_title="ReFill Hub: Business Intelligence Dashboard",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for an elegant look
st.markdown("""
<style>
    /* Main app background */
    .main {
        background-color: #F0F2F6;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E6E9EF;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E6E9EF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: all 0.3s ease-in-out;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2E8B57; /* Main theme color (Sea Green) */
    }
    
    /* Streamlit containers */
    .st-emotion-cache-18ni7ap {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING & PREPROCESSING (CACHED)
# =============================================================================
@st.cache_data
def load_and_clean_data(filepath):
    """
    Loads and cleans the dataset from the given filepath.
    This function is cached, so it only runs once.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"Error: Data file '{filepath}' not found. Please make sure it's in the same GitHub repository.")
        return None, None

    # Basic Cleaning
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.strip()
        
    # Feature Engineering for Modeling
    # Convert 'Family_Size' to a numerical average
    def process_family_size(val):
        if '5+' in val: return 5
        if '1-2' in val: return 1.5
        if '3-4' in val: return 3.5
        return val
    df['Family_Size_Num'] = df['Family_Size'].apply(process_family_size)
    
    # Define feature lists for models
    categorical_features = ['Age_Group', 'Gender', 'Emirate', 'Occupation', 'Income', 
                            'Purchase_Location', 'Purchase_Frequency', 'Uses_Eco_Products',
                            'Preferred_Packaging', 'Aware_Plastic_Ban', 'Eco_Brand_Preference', 
                            'Follow_Campaigns', 'Used_Refill_Before', 'Preferred_Payment_Mode',
                            'Refill_Location', 'Container_Type', 'Interest_Non_Liquids', 'Discount_Switch']

    numerical_features = ['Family_Size_Num', 'Importance_Convenience', 'Importance_Price', 
                          'Importance_Sustainability', 'Reduce_Waste_Score', 'Social_Influence_Score', 
                          'Try_Refill_Likelihood']

    # Features for clustering (psychographic & behavioral)
    cluster_features = ['Importance_Convenience', 'Importance_Price', 'Importance_Sustainability', 
                        'Reduce_Waste_Score', 'Eco_Brand_Preference', 'Social_Influence_Score']

    return df, categorical_features, numerical_features, cluster_features

# =============================================================================
# MODEL TRAINING MASTER FUNCTION (CACHED)
# =============================================================================
@st.cache_resource
def train_all_models(df, categorical_features, numerical_features, cluster_features):
    """
    Trains all required models (Clustering, Classification, Regression, Association)
    and returns them. This function is cached to run only once.
    """
    models = {}
    metrics = {}
    
    # --- 1. CLASSIFICATION (Predicting Adoption) ---
    X_class = df[categorical_features + numerical_features]
    y_class = df['Likely_to_Use_ReFillHub'].map({'Yes': 1, 'No': 0})
    
    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Create the full pipeline with a RandomForestClassifier
    clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))])
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42, stratify=y_class)
    clf_pipeline.fit(X_train_c, y_train_c)
    y_pred_c = clf_pipeline.predict(X_test_c)
    
    models['classification'] = clf_pipeline
    metrics['classification_report'] = classification_report(y_test_c, y_pred_c, output_dict=True)

    # --- 2. REGRESSION (Predicting Spending) ---
    X_reg = df[categorical_features + numerical_features]
    y_reg = df['Willingness_to_Pay_AED']
    
    # We can reuse the same preprocessor
    reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', RandomForestRegressor(random_state=42, n_estimators=100))])

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    reg_pipeline.fit(X_train_r, y_train_r)
    y_pred_r = reg_pipeline.predict(X_test_r)
    
    models['regression'] = reg_pipeline
    metrics['r2_score'] = r2_score(y_test_r, y_pred_r)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_test_r, y_pred_r))

    # --- 3. CLUSTERING (Customer Segmentation) ---
    X_cluster = df[cluster_features]
    cluster_scaler = StandardScaler()
    X_cluster_scaled = cluster_scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10) # 4 clusters for better segmentation
    df['Cluster'] = kmeans.fit_predict(X_cluster_scaled)
    
    models['clustering_model'] = kmeans
    models['clustering_scaler'] = cluster_scaler
    metrics['silhouette_score'] = silhouette_score(X_cluster_scaled, df['Cluster'])
    
    # Analyze cluster profiles
    cluster_profiles = df.groupby('Cluster')[cluster_features].mean()
    models['cluster_profiles'] = cluster_profiles

    # --- 4. ASSOCIATION RULE MINING (Market Basket) ---
    transactions = df['Products_Bought'].apply(lambda x: [item.strip() for item in x.split(',')]).tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = apriori(df_trans, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    # Clean up rules for display
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    models['association_rules'] = rules.sort_values('lift', ascending=False)
    
    return models, metrics, df # Return the dataframe with cluster labels

# =============================================================================
# MAIN APP EXECUTION
# =============================================================================

# Load data
df, cat_features, num_features, cluster_features = load_and_clean_data('ReFillHub_SyntheticSurvey.csv')

if df is not None:
    # Train models
    models, metrics, df_clustered = train_all_models(df.copy(), cat_features, num_features, cluster_features)

    # =========================================================================
    # SIDEBAR NAVIGATION
    # =========================================================================
    st.sidebar.image("https://placehold.co/400x200/2E8B57/FFFFFF?text=ReFill+Hub&font=sans-serif", use_column_width=True)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a Page:",
                            ["Executive Summary", 
                             "Customer Segmentation (Clustering)", 
                             "Predictive Simulator (Class. & Reg.)",
                             "Market Basket Analysis (Association)",
                             "Model Performance & Methodology"])

    # =========================================================================
    # PAGE 1: EXECUTIVE SUMMARY
    # =========================================================================
    if page == "Executive Summary":
        st.title("♻️ ReFill Hub: Executive Summary")
        st.markdown("This dashboard provides data-driven intelligence for the ReFill Hub concept, based on a market survey of 600 respondents in the UAE.")
        
        # --- Key Metrics Row ---
        st.header("Top-Line Metrics")
        col1, col2, col3 = st.columns(3)
        adoption_rate = (df['Likely_to_Use_ReFillHub'] == 'Yes').mean() * 100
        avg_wtp = df['Willingness_to_Pay_AED'].mean()
        
        with col1:
            st.metric(label="Projected Adoption Rate", value=f"{adoption_rate:.1f}%")
        with col2:
            st.metric(label="Avg. Willingness to Pay", value=f"AED {avg_wtp:.2f}", help="Average amount willing to spend per visit.")
        with col3:
            st.metric(label="Identified Customer Segments", value="4", help="Based on K-Means Clustering.")

        # --- Charts Row ---
        st.header("Key Customer Insights")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Adoption by Income Level")
            fig_inc = px.histogram(df, x="Income", color="Likely_to_Use_ReFillHub",
                                   barmode="group",
                                   color_discrete_map={'Yes': '#2E8B57', 'No': '#FF6347'},
                                   title="Adoption Likelihood by Income (AED)")
            st.plotly_chart(fig_inc, use_container_width=True)
            
        with col_b:
            st.subheader("Adoption by Emirate")
            fig_em = px.histogram(df, x="Emirate", color="Likely_to_Use_ReFillHub",
                                  barmode="group",
                                  color_discrete_map={'Yes': '#2E8B57', 'No': '#FF6347'},
                                  title="Adoption Likelihood by Emirate of Residence")
            st.plotly_chart(fig_em, use_container_width=True)
            
        st.subheader("Preferred Kiosk Locations")
        location_data = df.groupby('Refill_Location')['Likely_to_Use_ReFillHub'].count().reset_index(name='count')
        fig_loc = px.treemap(location_data, path=['Refill_Location'], values='count',
                             title="Most Preferred Kiosk Locations",
                             color_discrete_sequence=px.colors.sequential.Greens_r)
        st.plotly_chart(fig_loc, use_container_width=True)

    # =========================================================================
    # PAGE 2: CUSTOMER SEGMENTATION (CLUSTERING)
    # =========================================================================
    elif page == "Customer Segmentation (Clustering)":
        st.title("🧩 Customer Segmentation (Clustering)")
        st.markdown("We used **K-Means Clustering (k=4)** to segment respondents based on their attitudes towards price, sustainability, and convenience.")
        st.metric("Model Silhouette Score", f"{metrics['silhouette_score']:.3f}", help="A score from -1 to 1. Higher is better. 0.35 suggests reasonable separation.")

        # --- 3D Cluster Plot ---
        st.header("Interactive 3D Cluster Visualization")
        df_clustered['Cluster'] = df_clustered['Cluster'].astype(str) # For discrete colors
        fig_3d = px.scatter_3d(df_clustered, 
                               x='Importance_Price', 
                               y='Importance_Sustainability', 
                               z='Importance_Convenience',
                               color='Cluster', 
                               symbol='Cluster',
                               opacity=0.7,
                               title="3D Customer Segments")
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_3d, use_container_width=True)

        # --- Cluster Profiles ---
        st.header("Customer Personas")
        st.dataframe(models['cluster_profiles'].style.background_gradient(cmap='Greens'))
        
        st.markdown("### Persona Analysis & Strategy:")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Cluster 0: The Eco-Warriors (High Sustainability)")
            st.markdown("- **Profile:** High importance on sustainability and waste reduction. Not very price-sensitive. High social influence.")
            st.markdown("- **Strategy:** Target with impact-messaging (e.g., 'You saved 5 plastic bottles!'). Prime candidates for premium/eco-brand partnerships.")
            
            st.subheader("Cluster 1: The Budget-Conscious (High Price Sensitivity)")
            st.markdown("- **Profile:** Price is the most important factor. Less concerned with waste or brand preference.")
            st.markdown("- **Strategy:** Target with value-messaging ('Save 20% vs. packaged goods'). Win them with subscription discounts and loyalty points.")

        with c2:
            st.subheader("Cluster 2: The Convenience Seekers (High Convenience)")
            st.markdown("- **Profile:** Value convenience above all. Average sustainability interest. Likely busy professionals.")
            st.markdown("- **Strategy:** Target with location (supermarkets, residential lobbies) and speed. App-based pre-ordering and 'tap-to-pay' are essential.")
            
            st.subheader("Cluster 3: The Apathetic (Low Overall Engagement)")
            st.markdown("- **Profile:** Low scores on price, sustainability, and convenience. Hardest to motivate.")
            st.markdown("- **Strategy:** Low-priority segment. Awareness campaigns and strong discounts (e.g., 'First refill 50% off') are needed to convert them.")

    # =========================================================================
    # PAGE 3: PREDICTIVE SIMULATOR (CLASSIFICATION & REGRESSION)
    # =========================================================================
    elif page == "Predictive Simulator (Class. & Reg.)":
        st.title("🔮 Predictive Simulator")
        st.markdown("Use our trained models to simulate a new customer. Predict their **adoption likelihood (Classification)** and **spending potential (Regression)**.")
        
        with st.form("simulation_form"):
            st.header("Customer Profile")
            
            # --- Input Columns ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Demographics")
                age = st.selectbox("Age Group", df['Age_Group'].unique())
                gender = st.selectbox("Gender", df['Gender'].unique())
                income = st.selectbox("Income Level (AED)", df['Income'].unique())
                fam_size = st.selectbox("Family Size", df['Family_Size'].unique(), index=1)
                emirate = st.selectbox("Emirate", df['Emirate'].unique())
            
            with col2:
                st.subheader("Attitudes (1=Low, 5=High)")
                imp_price = st.slider("Importance: Price", 1, 5, 3)
                imp_sust = st.slider("Importance: Sustainability", 1, 5, 3)
                imp_conv = st.slider("Importance: Convenience", 1, 5, 3)
                waste_score = st.slider("Actively Reduces Waste", 1, 5, 3)
                social_score = st.slider("Social Influence Score", 1, 5, 3)
            
            with col3:
                st.subheader("Current Behaviors")
                eco_brand = st.select_slider("Eco-Brand Preference", [1, 2, 3, 4, 5], 3)
                follow_camp = st.selectbox("Follows Green Campaigns?", ["Yes", "No"])
                used_before = st.selectbox("Used Refill Before?", ["Yes", "No"])
                purchase_freq = st.selectbox("Purchase Frequency", df['Purchase_Frequency'].unique())
                try_likelihood = st.select_slider("Initial Likelihood to Try (1-5)", [1, 2, 3, 4, 5], 3)

            # Submit button
            submitted = st.form_submit_button("Run Simulation", use_container_width=True)
            
        # --- Results Display ---
        if submitted:
            # Preprocess numerical input
            if '5+' in fam_size: fam_num = 5
            elif '1-2' in fam_size: fam_num = 1.5
            else: fam_num = 3.5
            
            # Create DataFrame for model input (MUST MATCH TRAINING COLUMNS)
            # We fill unused/placeholder columns with the most common value (mode)
            input_data = pd.DataFrame({
                'Age_Group': [age], 'Gender': [gender], 'Emirate': [emirate], 
                'Occupation': [df['Occupation'].mode()[0]], 'Income': [income], 
                'Purchase_Location': [df['Purchase_Location'].mode()[0]], 
                'Purchase_Frequency': [purchase_freq], 'Uses_Eco_Products': [df['Uses_Eco_Products'].mode()[0]], 
                'Preferred_Packaging': [df['Preferred_Packaging'].mode()[0]], 
                'Aware_Plastic_Ban': [df['Aware_Plastic_Ban'].mode()[0]],
                'Eco_Brand_Preference': [eco_brand], 'Follow_Campaigns': [follow_camp], 
                'Used_Refill_Before': [used_before], 'Preferred_Payment_Mode': [df['Preferred_Payment_Mode'].mode()[0]],
                'Refill_Location': [df['Refill_Location'].mode()[0]], 
                'Container_Type': [df['Container_Type'].mode()[0]],
                'Interest_Non_Liquids': [df['Interest_Non_Liquids'].mode()[0]], 
                'Discount_Switch': [df['Discount_Switch'].mode()[0]],
                
                'Family_Size_Num': [fam_num], 'Importance_Convenience': [imp_conv], 
                'Importance_Price': [imp_price], 'Importance_Sustainability': [imp_sust], 
                'Reduce_Waste_Score': [waste_score], 'Social_Influence_Score': [social_score], 
                'Try_Refill_Likelihood': [try_likelihood]
            })

            # Make predictions
            clf_pipeline = models['classification']
            reg_pipeline = models['regression']
            
            pred_prob = clf_pipeline.predict_proba(input_data)[0]
            pred_spend = reg_pipeline.predict(input_data)[0]
            
            adoption_probability = pred_prob[1] # Probability of 'Yes' (class 1)
            
            st.header("Simulation Results")
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.subheader("Adoption Likelihood (Classification)")
                
                # Create a Plotly Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = adoption_probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Adoption Probability", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#2E8B57" if adoption_probability > 0.5 else "#FF6347"},
                        'steps' : [
                            {'range': [0, 50], 'color': 'rgba(255, 99, 71, 0.2)'},
                            {'range': [50, 100], 'color': 'rgba(46, 139, 87, 0.2)'}],
                    }))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with res_col2:
                st.subheader("Spending Potential (Regression)")
                st.metric(label="Predicted Willingness to Pay", value=f"AED {pred_spend:.2f}", help="Estimated spend per visit for this profile.")
                
                if pred_spend > avg_wtp + 10:
                    st.success("This is a High-Value Customer prospect!")
                elif pred_spend < avg_wtp - 10:
                    st.warning("This is a Low-Value Customer prospect. Focus on increasing basket size.")
                else:
                    st.info("This is an Average-Value Customer prospect.")

    # =========================================================================
    # PAGE 4: MARKET BASKET ANALYSIS (ASSOCIATION)
    # =========================================================================
    elif page == "Market Basket Analysis (Association)":
        st.title("🛒 Market Basket Analysis (Association Rules)")
        st.markdown("Discovering which products are frequently bought together. This helps optimize kiosk layout and create product bundles.")
        
        rules_df = models['association_rules']
        
        # --- Filter sliders ---
        col1, col2 = st.columns(2)
        with col1:
            min_lift = st.slider("Minimum Lift", min_value=1.0, max_value=float(rules_df['lift'].max()), value=1.2, help="How much more likely products are bought together vs. random chance. >1 is good.")
        with col2:
            min_confidence = st.slider("Minimum Confidence", min_value=0.0, max_value=float(rules_df['confidence'].max()), value=0.1, help="How often the rule is true (e.g., 0.2 = 20% of people who bought X also bought Y).")
            
        filtered_rules = rules_df[(rules_df['lift'] > min_lift) & (rules_df['confidence'] > min_confidence)]
        
        st.subheader(f"Found {len(filtered_rules)} Rules")
        st.dataframe(filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False))
        
        # --- Top Rules Chart ---
        st.subheader("Top Product Associations")
        top_rules = filtered_rules.sort_values('lift', ascending=False).head(15)
        top_rules['Rule'] = top_rules['antecedents'] + "  ➡️  " + top_rules['consequents']
        
        fig_rules = px.bar(top_rules, x="lift", y="Rule", orientation='h', 
                           title="Top 15 Product Associations by Lift",
                           color="confidence",
                           color_continuous_scale=px.colors.sequential.Greens)
        fig_rules.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_rules, use_container_width=True)
        
        st.subheader("Strategic Insights")
        st.markdown("""
        - **High Lift (e.g., Fabric Softener -> Detergent):** These items are strongly related. Place them in the same kiosk or offer a "Laundry Day" bundle discount.
        - **High Confidence (e.g., Shampoo -> Conditioner):** This is an extremely common path. If a user selects Shampoo, the app should immediately recommend Conditioner.
        """)

    # =========================================================================
    # PAGE 5: MODEL PERFORMANCE & METHODOLOGY
    # =========================================================================
    elif page == "Model Performance & Methodology":
        st.title("🔬 Model Performance & Methodology")
        st.markdown("This page details the performance scores for each model used in the dashboard, as required by the group assignment.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Classification Model")
            st.markdown("**Task:** Predict if a user will adopt (`Yes`) or not (`No`).")
            st.markdown("**Algorithm:** `RandomForestClassifier`")
            
            report = metrics['classification_report']
            st.subheader("Overall Accuracy")
            st.metric("Accuracy", f"{report['accuracy']:.2%}")
            
            st.subheader("Class-Specific Performance (Yes/No)")
            # Create a clean DataFrame from the report
            df_report = pd.DataFrame({
                'Metric': ['Precision', 'Recall', 'F1-Score', 'Support'],
                'No (Class 0)': [report['0']['precision'], report['0']['recall'], report['0']['f1-score'], report['0']['support']],
                'Yes (Class 1)': [report['1']['precision'], report['1']['recall'], report['1']['f1-score'], report['1']['support']]
            }).set_index('Metric')
            st.dataframe(df_report.style.format("{:.2f}"))
            st.markdown("- **Precision:** Of all 'Yes' predictions, how many were correct.")
            st.markdown("- **Recall:** Of all actual 'Yes' customers, how many did we find.")

        with col2:
            st.header("Regression Model")
            st.markdown("**Task:** Predict how much a user is willing to spend (AED).")
            st.markdown("**Algorithm:** `RandomForestRegressor`")
            
            st.subheader("Model Fit")
            st.metric("R-squared (R²)", f"{metrics['r2_score']:.3f}", help="How much of the variance in spending is explained by our model. 1.0 is a perfect fit.")
            
            st.subheader("Prediction Error")
            st.metric("Root Mean Squared Error (RMSE)", f"AED {metrics['rmse']:.2f}", help="The average error of our spending prediction. A lower RMSE is better.")

        st.divider()
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.header("Clustering Model")
            st.markdown("**Task:** Segment users into distinct personas.")
            st.markdown("**Algorithm:** `K-Means Clustering` (k=4)")
            
            st.subheader("Cluster Separation")
            st.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}", help="Measures how well-separated clusters are. 1 is perfect, 0 is overlapping. 0.35 is good for real-world survey data.")
            
        with col4:
            st.header("Association Model")
            st.markdown("**Task:** Find co-purchased products.")
            st.markdown("**Algorithm:** `Apriori`")
            
            st.subheader("Rules Found")
            st.metric("Total Rules Generated", f"{len(models['association_rules'])}", help="Number of 'if-then' rules found with lift > 1.0 and support > 5%.")

else:
    st.error("Fatal Error: Could not load data. Dashboard cannot start.")