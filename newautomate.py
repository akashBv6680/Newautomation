import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder, OneHotEncoder, MinMaxScaler, Binarizer, Normalizer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from imblearn.over_sampling import SMOTE
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import xgboost as xgb
import smtplib
from email.message import EmailMessage
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import itertools # Import itertools for cycling through API keys

# === Agent Names ===
AGENT_NAMES = {
    'ingestion': "DataScout AI",
    'preprocess': "Cleanser AI",
    'visualize': "InsightLens AI",
    'model': "SmartModeler AI",
    'predict': "Verifier AI"
}

# --- Initialize Session State Variables ---
# This is crucial for controlling execution flow and preserving state across reruns.
if 'df_original' not in st.session_state:
    st.session_state['df_original'] = None
if 'X_processed' not in st.session_state:
    st.session_state['X_processed'] = None
if 'y_processed' not in st.session_state:
    st.session_state['y_processed'] = None
if 'target_column_name' not in st.session_state:
    st.session_state['target_column_name'] = None
if 'preprocessor_le_dict' not in st.session_state: # Stores LabelEncoders for features
    st.session_state['preprocessor_le_dict'] = {}
if 'target_label_encoder' not in st.session_state: # Stores LabelEncoder for the target
    st.session_state['target_label_encoder'] = None
if 'best_model' not in st.session_state:
    st.session_state['best_model'] = None
if 'best_model_info' not in st.session_state: # Stores model info and scaler
    st.session_state['best_model_info'] = None
if 'model_training_completed' not in st.session_state:
    st.session_state['model_training_completed'] = False
if 'together_key_cycler' not in st.session_state:
    st.session_state.together_key_cycler = None # Initialize to None, set later

# === Email Notification ===
def send_email_report(subject, body, to, attachment_paths=None):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = st.secrets["EMAIL_ADDRESS"]
    msg['To'] = to
    msg.set_content(body)

    if attachment_paths:
        for path in attachment_paths:
            try:
                with open(path, 'rb') as f:
                    img_data = f.read()
                msg.add_attachment(img_data, maintype='image', subtype='png', filename=os.path.basename(path))
            except FileNotFoundError:
                st.warning(f"Attachment file not found: {path}")
            except Exception as e:
                st.warning(f"Could not attach file {path}: {e}")

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_ADDRESS"], st.secrets["EMAIL_PASSWORD"])
            smtp.send_message(msg)
        st.success(f"Email report sent to {to}")
    except Exception as e:
        st.error(f"Failed to send email report: {e}")

# === Helper: Ask TogetherAI (for InsightLens AI) ===
def get_together_api_key_cycler():
    """Retrieves Together AI API keys from Streamlit secrets and creates a cyclic iterator."""
    keys = []
    # Attempt to load keys, allowing for multiple keys named sequentially
    for i in range(1, 10): # Check for TOGETHER_API_KEY_1, TOGETHER_API_KEY_2, etc.
        key_name = f"TOGETHER_API_KEY_{i}"
        if key_name in st.secrets:
            keys.append(st.secrets[key_name])
        else:
            break # Stop if a key in the sequence is not found

    if not keys:
        st.error("Together AI API keys not found in Streamlit secrets. Please configure them (e.g., TOGETHER_API_KEY_1).")
        return iter([]) # Return an empty iterator if no keys

    return itertools.cycle(keys)

# Initialize the key cycler once
if st.session_state.together_key_cycler is None:
    st.session_state.together_key_cycler = get_together_api_key_cycler()

def ask_agent(prompt, model="mistralai/Mistral-7B-Instruct-v0.1"):
    """
    Sends a prompt to the Together AI API using a cyclically selected API key.
    """
    try:
        current_api_key = next(st.session_state.together_key_cycler)
    except StopIteration:
        st.error("No Together AI API keys available. Please check your Streamlit secrets configuration.")
        return "Insight generation failed: No API key."
    except Exception as e:
        st.error(f"Error getting Together AI API key: {e}")
        return "Insight generation failed: API key access error."

    if not current_api_key:
        st.error("Together AI API key is empty. Please check your Streamlit secrets.")
        return "Insight generation failed: Empty API key."

    try:
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {current_api_key}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]}
        )
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        st.error(f"Together AI API request failed: {e}. Check API key and network.")
        return "Insight generation failed."
    except KeyError:
        st.error("Unexpected response from Together AI API. Check API documentation.")
        return "Insight generation failed."


# === Web Scraper using Selenium ===
@st.cache_data(ttl=3600) # Cache the scraped data for 1 hour
def scrape_web_table(url, column_name):
    """
    Scrapes a table from a given URL and extracts a specific column.
    Uses Selenium with headless Chrome.
    """
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu') # Often needed in headless environments

    try:
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(3) # Give time for the page to load

        tables = driver.find_elements(By.TAG_NAME, "table")
        for table in tables:
            try:
                df = pd.read_html(table.get_attribute('outerHTML'))[0]
                if column_name in df.columns:
                    driver.quit()
                    st.success(f"Successfully scraped '{column_name}' from web table.")
                    return df[[column_name]]
            except Exception as e:
                # Catch specific read_html errors or parsing errors for this table
                continue # Try the next table

        driver.quit()
        st.error(f"Could not find a table containing the column '{column_name}' on the page.")
        return None
    except Exception as e:
        st.error(f"Error during web scraping with Selenium: {e}. Make sure Chrome driver is compatible and available.")
        if 'driver' in locals() and driver:
            driver.quit()
        return None

# === Ingestion Agent ===
def ingest_data():
    """
    Handles data ingestion from various sources (CSV, Excel, JSON, Web URL).
    """
    st.subheader(f"🌐 {AGENT_NAMES['ingestion']}: Data Ingestion")
    file_type = st.selectbox("What type of dataset do you have?", ["CSV", "Excel", "JSON", "Web URL"], key="file_type_select")

    df = None
    if file_type == "Web URL":
        url = st.text_input("Enter the webpage URL", key="web_url_input")
        column = st.text_input("Which column do you want to extract from the web table?", key="web_column_input")
        if url and column:
            if st.button("Scrape Data", key="scrape_button"):
                with st.spinner(f"Scraping data from {url}... This might take a moment."):
                    df = scrape_web_table(url, column)
                    if df is not None:
                        st.write("First 5 rows of scraped data:")
                        st.dataframe(df.head())
                        st.session_state['df_original'] = df # Store it in session state
                        st.session_state['model_training_completed'] = False # Reset state
    else:
        uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "json"])
        if uploaded_file:
            try:
                if file_type == "CSV":
                    df = pd.read_csv(uploaded_file)
                elif file_type == "Excel":
                    df = pd.read_excel(uploaded_file)
                elif file_type == "JSON":
                    df = pd.read_json(uploaded_file)
                st.success("File uploaded successfully!")
                st.write("First 5 rows of your dataset:")
                st.dataframe(df.head())
                st.session_state['df_original'] = df # Store it in session state
                st.session_state['model_training_completed'] = False # Reset state
            except Exception as e:
                st.error(f"Error reading the file: {e}. Please check the file format.")
    return df


# === Preprocessing Agent ===
def preprocess_data(df):
    """
    Handles data preprocessing: missing values, encoding, and imbalance handling.
    """
    st.subheader(f"🧹 {AGENT_NAMES['preprocess']}: Data Preprocessing")
    st.write("Inferring target variable and performing basic preprocessing...")

    if df is None or df.empty:
        st.error("Cannot preprocess an empty DataFrame. Please ingest data first.")
        return None, None, None

    # Let the user select the target column
    target_col = st.selectbox("Which column is your target variable?", df.columns, key="target_col_select")

    y = df[target_col].copy() # Use .copy() to avoid SettingWithCopyWarning
    X = df.drop(columns=[target_col]).copy()

    # Handle missing values
    st.info("Handling missing values (forward fill for now, more advanced options could be added)...")
    X.fillna(method='ffill', inplace=True)
    y.fillna(method='ffill', inplace=True) # Important if target has missing values

    # Encode categorical features in X
    st.info("Encoding categorical features...")
    current_le_dict = {} # Dictionary to store LabelEncoders for feature columns
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str)) # Ensure string conversion
        current_le_dict[col] = le
    st.session_state['preprocessor_le_dict'] = current_le_dict # Store the encoders

    # Encode target variable if it's categorical
    is_classification = False
    if y.dtype == 'object' or (y.nunique() <= 20 and y.dtype not in ['int64', 'float64']): # Heuristic
        st.info("Target variable appears categorical. Applying Label Encoding to target.")
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))
        st.session_state['target_label_encoder'] = le_target # Store for later inverse transform if needed
        is_classification = True
    else:
        st.info("Target variable appears numerical. No encoding applied to target.")
        is_classification = False # Explicitly set for regression

    # Handle class imbalance for classification tasks
    if is_classification:
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) > 1:
            min_class_count = counts.min()
            max_class_count = counts.max()
            # Adjusted threshold for imbalance, you can tune this
            if min_class_count / max_class_count < 0.3:
                st.warning(f"Target class is imbalanced (Min count: {min_class_count}, Max count: {max_class_count}). Applying SMOTE...")
                try:
                    sm = SMOTE(random_state=42)
                    X_resampled, y_resampled = sm.fit_resample(X, y)
                    X, y = X_resampled, y_resampled
                    st.success(f"SMOTE applied. New class counts: {np.bincount(y)}")
                except ValueError as e:
                    st.warning(f"SMOTE could not be applied: {e}. This might happen if there's only one sample in a class after encoding or if all features are categorical and not enough samples.")
            else:
                st.info("Target class is reasonably balanced. No SMOTE applied.")
        else:
            st.warning("Only one unique class found in target. SMOTE not applicable.")
    else:
        st.info("Target variable is numerical (regression task). SMOTE not applicable.")

    st.success("Preprocessing complete!")
    st.write("Processed data (first 5 rows of features):")
    st.dataframe(X.head())
    st.write("Processed target (first 5 values):")
    st.write(y[:5])
    return X, y, target_col, is_classification

# === Visualization Agent ===
def visualize_and_insight(df_original):
    """
    Generates visualizations and AI-driven insights, saving them to a PDF.
    """
    pdf_path = "eda_report.pdf"
    st.subheader(f"📊 {AGENT_NAMES['visualize']}: Visual Insights")
    st.info("Generating visual insights and explanations... This might take a moment.")

    if df_original is None or df_original.empty:
        st.warning("No data available for visualization.")
        return None

    try:
        with PdfPages(pdf_path) as pdf:
            # Create a copy for plotting to avoid any in-place changes by plotting libraries
            df_plot = df_original.copy()

            cat_cols = df_plot.select_dtypes(include='object').columns
            num_cols = df_plot.select_dtypes(include=np.number).columns

            # Overview of the dataset
            fig_overview = plt.figure(figsize=(10, 2))
            gs_overview = GridSpec(1, 1)
            ax_overview = fig_overview.add_subplot(gs_overview[0])
            ax_overview.axis('off')
            ax_overview.text(0.05, 0.9, f"Dataset Overview:", fontsize=14, weight='bold')
            ax_overview.text(0.05, 0.7, f"Shape: {df_plot.shape[0]} rows, {df_plot.shape[1]} columns", fontsize=12)
            ax_overview.text(0.05, 0.5, f"Numerical Columns: {len(num_cols)}", fontsize=12)
            ax_overview.text(0.05, 0.3, f"Categorical Columns: {len(cat_cols)}", fontsize=12)
            pdf.savefig(fig_overview)
            plt.close(fig_overview)

            # Numerical columns histograms and insights
            for col in num_cols:
                fig = plt.figure(figsize=(11, 5))
                gs = GridSpec(1, 2, width_ratios=[2, 1])
                ax1 = fig.add_subplot(gs[0])
                df_plot[col].hist(ax=ax1, bins=20, color='skyblue', edgecolor='black')
                ax1.set_title(f"Histogram of {col}", fontsize=14)
                ax1.set_xlabel(col)
                ax1.set_ylabel("Frequency")
                ax1.grid(axis='y', alpha=0.75)

                ax2 = fig.add_subplot(gs[1])
                ax2.axis('off') # Turn off axes for text
                ax2.set_title("AI Insights", fontsize=14, weight='bold')

                prompt = f"Given a histogram of the numerical column '{col}', describe its distribution (e.g., normal, skewed, multimodal) and what that implies for business clients. Provide 3 short, concise bullet points targeted at non-technical business clients. Focus on actionable insights or key observations about the data's spread."
                raw_insights = ask_agent(prompt)
                points = '\n'.join([f"• {line.strip()}" for line in raw_insights.strip().split('\n') if line.strip()])
                ax2.text(0, 1, points, wrap=True, fontsize=10, verticalalignment='top', transform=ax2.transAxes)

                pdf.savefig(fig, bbox_inches='tight') # bbox_inches='tight' prevents labels from being cut off
                st.pyplot(fig)
                plt.close(fig)

            # Categorical columns bar charts and insights
            for col in cat_cols:
                fig = plt.figure(figsize=(11, 5))
                gs = GridSpec(1, 2, width_ratios=[2, 1])
                ax1 = fig.add_subplot(gs[0])
                df_plot[col].value_counts().plot(kind='bar', ax=ax1, color='orange', edgecolor='black')
                ax1.set_title(f"Bar Chart of {col}", fontsize=14)
                ax1.set_xlabel(col)
                ax1.set_ylabel("Count")
                plt.xticks(rotation=45, ha='right') # Rotate labels for readability
                ax1.grid(axis='y', alpha=0.75)

                ax2 = fig.add_subplot(gs[1])
                ax2.axis('off')
                ax2.set_title("AI Insights", fontsize=14, weight='bold')

                prompt = f"Given a bar chart of the categorical column '{col}', describe the key categories and their relative frequencies. What does this tell us about the data in terms of customer segments or popular choices? Provide 3 short, concise bullet points targeted at non-technical business clients."
                raw_insights = ask_agent(prompt)
                points = '\n'.join([f"• {line.strip()}" for line in raw_insights.strip().split('\n') if line.strip()])
                ax2.text(0, 1, points, wrap=True, fontsize=10, verticalalignment='top', transform=ax2.transAxes)

                pdf.savefig(fig, bbox_inches='tight')
                st.pyplot(fig)
                plt.close(fig)

            st.success("Visual insights generated and saved to PDF!")
            return pdf_path
    except Exception as e:
        st.error(f"Error during visualization: {e}")
        return None


# === Model Runner with Multiple Test Sizes ===
class ModelRunner:
    def __init__(self, X, y, is_classification):
        self.X = X
        self.y = y
        self.is_classification = is_classification

        if self.is_classification:
            self.models = [
                LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
                RandomForestClassifier(random_state=42),
                GradientBoostingClassifier(random_state=42),
                SVC(probability=True, random_state=42),
                KNeighborsClassifier(),
                xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            ]
            self.metric_name = "Accuracy"
            self.metric_func = accuracy_score
        else: # Regression
            self.models = [
                LinearRegression(),
                Lasso(random_state=42),
                Ridge(random_state=42),
                ElasticNet(random_state=42),
                DecisionTreeRegressor(random_state=42),
                RandomForestRegressor(random_state=42),
                GradientBoostingRegressor(random_state=42),
                KNeighborsRegressor(),
                SVR(),
                xgb.XGBRegressor(random_state=42)
            ]
            self.metric_name = "R2 Score"
            self.metric_func = r2_score

        self.best_model = None
        self.best_info = {}
        self.scaler = StandardScaler() # Initialize a scaler to be used and stored

    def run(self):
        """Trains and evaluates models across different test sizes, selecting the best one."""
        best_score = -np.inf # Initialize for maximization
        best_info = {}
        best_model = None

        st.info(f"Training and evaluating models for {'Classification' if self.is_classification else 'Regression'}...")
        progress_bar = st.progress(0)
        total_runs = len([0.1, 0.2, 0.25, 0.3]) * len(self.models)
        run_count = 0

        # Fit the scaler once on the full dataset (or train set if careful about data leakage)
        # For simplicity here, fitting on full X for consistent scaling for prediction
        # In a real pipeline, you'd fit on X_train only and transform X_test
        self.scaler.fit(self.X)

        for size in [0.1, 0.2, 0.25, 0.3]:
            # Use stratify for classification if target has more than one class
            stratify_y = self.y if self.is_classification and self.y.nunique() > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=size, random_state=42, stratify=stratify_y)

            # Scale data for current split
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            for model in self.models:
                run_count += 1
                progress_bar.progress(run_count / total_runs)
                model_name = model.__class__.__name__

                # Use scaled data for models that require it, otherwise use original
                if isinstance(model, (LogisticRegression, SVC, KNeighborsClassifier, SVR, KNeighborsRegressor, LinearRegression, Lasso, Ridge, ElasticNet)):
                    current_X_train = X_train_scaled
                    current_X_test = X_test_scaled
                else:
                    current_X_train = X_train
                    current_X_test = X_test

                try:
                    model.fit(current_X_train, y_train)
                    y_pred = model.predict(current_X_test)

                    score = self.metric_func(y_test, y_pred)
                    st.write(f"Model: {model_name}, Test Size: {int(size*100)}%, {self.metric_name}: {score:.4f}")

                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_info = {
                            'Model': model_name,
                            'Score': score,
                            'Type': 'Classification' if self.is_classification else 'Regression',
                            'Test Size': f"{int(size*100)}%",
                            'Scaler': self.scaler # Store the scaler used
                        }

                except Exception as e:
                    st.warning(f"Error training {model_name} with test size {int(size*100)}%: {e}")
                    continue

        progress_bar.empty()

        self.best_model = best_model
        self.best_info = best_info
        return best_model, best_info

    def save_best_model(self, filename="best_model.pkl"):
        """Saves the best performing model and its associated scaler and info."""
        if self.best_model:
            try:
                # Store model, scaler, and info together
                model_package = {
                    'model': self.best_model,
                    'scaler': self.scaler,
                    'info': self.best_info,
                    'is_classification': self.is_classification,
                    'target_label_encoder': st.session_state.get('target_label_encoder', None)
                }
                with open(filename, "wb") as f:
                    pickle.dump(model_package, f)
                st.success(f"Best model and scaler saved as '{filename}'")
            except Exception as e:
                st.error(f"Failed to save the best model: {e}")
        else:
            st.warning("No best model found to save.")

# === Prediction Interface ===
def prediction_interface():
    """
    Provides a simple interface for making predictions with the trained model.
    Allows manual input for a single prediction or uses existing data.
    """
    st.subheader(f"🔮 {AGENT_NAMES['predict']}: Prediction Interface")

    if st.session_state.best_model is None:
        st.warning("No model has been trained yet. Please complete previous steps.")
        return

    model = st.session_state.best_model
    model_info = st.session_state.best_model_info
    target_column_name = st.session_state.target_column_name
    X_sample_data = st.session_state.X_processed # Use processed X for feature names and types
    is_classification = model_info['Type'] == 'Classification'

    st.write("Enter values for features to get a prediction:")

    # Use st.form to group inputs and prevent reruns on every character
    with st.form("prediction_form"):
        input_data = {}
        st.write("Input values for prediction:")

        # Get the feature names
        feature_columns = X_sample_data.columns

        # Create input widgets dynamically in columns
        num_cols_per_row = 2
        cols = st.columns(num_cols_per_row)
        for i, col_name in enumerate(feature_columns):
            current_col = cols[i % num_cols_per_row]
            sample_value = X_sample_data[col_name].iloc[0]

            if pd.api.types.is_numeric_dtype(sample_value):
                # Use type coercion for number_input
                input_data[col_name] = current_col.number_input(f"Enter value for {col_name}:", value=float(sample_value), key=f"input_{col_name}")
            else:
                # This case should be rare if preprocessing converts all object dtypes
                input_data[col_name] = current_col.text_input(f"Enter value for {col_name}:", value=str(sample_value), key=f"input_{col_name}")

        submit_button = st.form_submit_button("Get Prediction")

        if submit_button:
            try:
                # Create a DataFrame for prediction, ensuring column order
                input_df = pd.DataFrame([input_data], columns=feature_columns)

                # Apply LabelEncoders for feature columns that were originally categorical
                # Use the stored preprocessor_le_dict from session state
                for col, le in st.session_state['preprocessor_le_dict'].items():
                    if col in input_df.columns and input_df[col].dtype == 'object': # Check if it's still object (might be if user input)
                        # Ensure the input value is known to the encoder, or handle new values
                        # For simplicity, if a value is new, it will raise an error.
                        # In production, you might map unknown values to a specific placeholder or average.
                        try:
                            input_df[col] = le.transform(input_df[col].astype(str))
                        except ValueError as ve:
                            st.error(f"Input error for categorical column '{col}': {ve}. Value might be new/unseen during training.")
                            return

                # Apply scaling using the scaler stored with the best model
                scaled_input_data = input_df
                if model_info['Scaler'] is not None:
                    scaler = model_info['Scaler']
                    scaled_input_data = scaler.transform(input_df)

                prediction = model.predict(scaled_input_data)

                if is_classification:
                    # If target was label encoded, inverse transform the prediction
                    if st.session_state.target_label_encoder is not None:
                        decoded_prediction = st.session_state.target_label_encoder.inverse_transform(prediction)
                        st.success(f"Predicted {target_column_name}: **{decoded_prediction[0]}**")
                    else:
                        st.success(f"Predicted {target_column_name} (numeric class): **{prediction[0]}**")

                    # If the model supports probability, show probabilities
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(scaled_input_data)[0]
                        st.write("Class Probabilities:")
                        # Ensure class names are available for display
                        class_names = getattr(model, 'classes_', range(len(probabilities)))
                        prob_df = pd.DataFrame({'Class': class_names, 'Probability': probabilities})
                        st.dataframe(prob_df.style.format({'Probability': "{:.2%}"}))

                else: # Regression
                    st.success(f"Predicted {target_column_name}: **{prediction[0]:.2f}**")

            except Exception as e:
                st.error(f"Error during prediction: {e}. Please ensure input data matches the model's expectations and types.")
                st.warning("Common issues: New categorical values, incorrect numeric formats, or missing feature inputs.")


# === App Runner ===
st.set_page_config(page_title="Agentic AutoML AI", layout="wide", initial_sidebar_state="expanded")
st.title("🤖 Multi-Agent AutoML System")
st.markdown("---")

# Client email input in the sidebar
with st.sidebar:
    st.header("Client Communication")
    client_email = st.text_input("Enter Client Email for Reports", help="Reports will be sent to this email address.", key="client_email_input")
    if not client_email:
        st.warning("Please enter a client email to receive automated reports.")

st.sidebar.markdown("---")
st.sidebar.header("System Status")

# --- Agent 1: Ingestion ---
st.sidebar.write(f"**{AGENT_NAMES['ingestion']} Status:**")
# Call ingest_data which will update st.session_state['df_original']
ingest_data()
if st.session_state.df_original is not None and not st.session_state.df_original.empty:
    st.sidebar.success("Data ingested successfully!")
else:
    st.sidebar.warning("Waiting for data ingestion...")


if st.session_state.df_original is not None:
    st.markdown("---")
    # --- Agent 2: Preprocessing ---
    st.sidebar.write(f"**{AGENT_NAMES['preprocess']} Status:**")
    # Only run preprocessing if data is new or not yet processed
    if st.session_state.X_processed is None or st.button("Re-run Preprocessing", key="rerun_preprocess_button"):
        X, y, target_col, is_classification = preprocess_data(st.session_state.df_original.copy())
        if X is not None and y is not None:
            st.sidebar.success("Data preprocessing complete!")
            st.session_state['X_processed'] = X
            st.session_state['y_processed'] = y
            st.session_state['target_column_name'] = target_col
            st.session_state['is_classification_task'] = is_classification # Store classification type
            st.session_state['model_training_completed'] = False # Reset model training status
        else:
            st.sidebar.warning("Preprocessing failed or not yet started.")
    else:
        st.sidebar.info("Data already preprocessed. Click 'Re-run Preprocessing' if you made changes.")


    if st.session_state.X_processed is not None and st.session_state.y_processed is not None:
        st.markdown("---")
        # --- Agent 3: Visualization ---
        st.sidebar.write(f"**{AGENT_NAMES['visualize']} Status:**")
        # Only re-generate visuals if original DF changes or if forced
        if 'pdf_report_path' not in st.session_state or st.button("Re-generate Visuals", key="regen_visuals_button"):
            pdf_path = visualize_and_insight(st.session_state.df_original)
            if pdf_path:
                st.sidebar.success("Visual insights generated!")
                st.session_state['pdf_report_path'] = pdf_path
            else:
                st.sidebar.warning("Visual insights generation failed.")

        if 'pdf_report_path' in st.session_state and st.session_state['pdf_report_path']:
            with open(st.session_state['pdf_report_path'], "rb") as f:
                st.download_button("📥 Download Visual Report", f, file_name="Insights_Report.pdf", use_container_width=True, key="download_visual_report")

            if client_email and 'initial_report_sent' not in st.session_state:
                eda_summary = f"""
Dear Client,

Our system has completed the initial analysis of your dataset. Please find the attached visual insights report for your review.

Key observations from the initial data scan include:
- The dataset has {st.session_state.df_original.shape[0]} rows and {st.session_state.df_original.shape[1]} columns.
- We have identified numerical and categorical features within your data.
- Initial data quality checks may indicate missing values or potential outliers, which have been addressed in preprocessing.

The attached report provides detailed visualizations (histograms for numerical data, bar charts for categorical data) along with AI-generated explanations to help you understand your data at a glance.

Please confirm if you'd like us to proceed with advanced data cleaning and model training based on these insights.

Regards,
The Agentic AutoML AI Team
"""
                send_email_report("Initial Data Quality & Visual Report", eda_summary, client_email, [st.session_state['pdf_report_path']])
                st.warning("Initial report emailed to client for confirmation before continuing.")
                st.session_state['initial_report_sent'] = True # Mark as sent

        # Client confirmation to proceed for model training
        proceed = st.checkbox("✅ Client confirmed. Proceed with model training?", key="client_proceed_checkbox")

        if proceed:
            st.markdown("---")
            # --- Agent 4: Model Training ---
            st.sidebar.write(f"**{AGENT_NAMES['model']} Status:**")

            # Only run model training if it hasn't been completed yet for the current data
            if not st.session_state.model_training_completed:
                st.info("Starting model training and selection...")
                model_runner = ModelRunner(st.session_state.X_processed, st.session_state.y_processed, st.session_state.is_classification_task)
                best_model, best_info = model_runner.run()
                if best_model:
                    st.sidebar.success("Model training complete!")
                    st.success(f"Best Model: **{best_info['Model']}** | Score: **{best_info['Score']:.4f}** | Type: **{best_info['Type']}** | Test Size: **{best_info['Test Size']}**")
                    model_runner.save_best_model("best_model.pkl")
                    st.session_state['best_model'] = best_model
                    st.session_state['best_model_info'] = best_info
                    st.session_state['model_training_completed'] = True # Mark as completed

                    if client_email and 'final_report_sent' not in st.session_state:
                        model_summary = f"""
Dear Client,

The AutoML process is complete, and we have identified the best-performing model for your dataset.

Here are the details of the selected model:
- **Best Model**: {best_info['Model']}
- **Performance Score ({best_info['Type']})**: {best_info['Score']:.4f}
- **Data Split (Test Size)**: {best_info['Test Size']}

This model is now ready for predictions.

Thank for using our AI service.

Regards,
The Agentic AutoML AI Team
"""
                        send_email_report("Final AutoML Model Report", model_summary, client_email)
                        st.info("📬 Final report emailed to client.")
                        st.session_state['final_report_sent'] = True
                else:
                    st.sidebar.warning("Model training failed or no best model found.")
                    st.session_state['model_training_completed'] = False # Ensure false if failed
            else:
                st.sidebar.success("Model training already completed for current data.")
                st.info("Model training already completed. Details:")
                st.write(f"Best Model: **{st.session_state.best_model_info['Model']}** | Score: **{st.session_state.best_model_info['Score']:.4f}** | Type: **{st.session_state.best_model_info['Type']}** | Test Size: **{st.session_state.best_model_info['Test Size']}**")

            if st.session_state.best_model is not None:
                st.markdown("---")
                # --- Agent 5: Prediction Interface ---
                st.sidebar.write(f"**{AGENT_NAMES['predict']} Status:**")
                prediction_interface() # Call prediction_interface without passing args, it uses session_state
                st.sidebar.info("Prediction interface ready.")
            else:
                st.sidebar.warning("Prediction interface not available: No model trained.")
        else:
            st.sidebar.info("Awaiting client confirmation for model training.")
