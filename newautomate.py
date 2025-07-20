# === Agentic AutoML Multi-Agent System ===

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
import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
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

# === Agent Names ===
AGENT_NAMES = {
    'ingestion': "DataScout AI",
    'preprocess': "Cleanser AI",
    'visualize': "InsightLens AI",
    'model': "SmartModeler AI",
    'predict': "Verifier AI"
}

# === Email Notification ===
def send_email_report(subject, body, to, attachment_paths=None):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = st.secrets["EMAIL_ADDRESS"]
    msg['To'] = to
    msg.set_content(body)

    if attachment_paths:
        for path in attachment_paths:
            with open(path, 'rb') as f:
                img_data = f.read()
            msg.add_attachment(img_data, maintype='image', subtype='png', filename=os.path.basename(path))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(st.secrets["EMAIL_ADDRESS"], st.secrets["EMAIL_PASSWORD"])
        smtp.send_message(msg)

# === Helper: Ask TogetherAI (for InsightLens AI) ===
def ask_agent(prompt, model="mistralai/Mistral-7B-Instruct-v0.1"):
    response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={"Authorization": f"Bearer {st.secrets['TOGETHER_API_KEY']}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}]}
    )
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    return "Insight generation failed."
# === Web Scraper using Selenium ===
def scrape_web_table(url, column_name):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(3)
    tables = driver.find_elements(By.TAG_NAME, "table")
    for table in tables:
        try:
            df = pd.read_html(table.get_attribute('outerHTML'))[0]
            if column_name in df.columns:
                driver.quit()
                return df[[column_name]]
        except:
            continue
    driver.quit()
    return None

# === Ingestion Agent ===
def ingest_data():
    file_type = st.selectbox("What type of dataset do you have?", ["CSV", "Excel", "JSON", "Web URL"])
    if file_type == "Web URL":
        url = st.text_input("Enter the webpage URL")
        column = st.text_input("Which column do you want to extract from the web table?")
        if url and column:
            df = scrape_web_table(url, column)
            if df is not None:
                st.success("Web data loaded successfully!")
                return df
            else:
                st.error("Could not find the column in any table on the page.")
        return None
    else:
        uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "json"])
        if uploaded_file:
            if file_type == "CSV":
                return pd.read_csv(uploaded_file)
            elif file_type == "Excel":
                return pd.read_excel(uploaded_file)
            elif file_type == "JSON":
                return pd.read_json(uploaded_file)
    return None


# === Preprocessing Agent ===
def preprocess_data(df):
    target_col = st.selectbox("Which column is your target variable?", df.columns)
    y = df[target_col]
    X = df.drop(columns=[target_col])

    st.write("Handling missing values, encoding, and balancing...")
    df.fillna(method='ffill', inplace=True)

    le = LabelEncoder()
    for col in X.select_dtypes(include='object').columns:
        X[col] = le.fit_transform(X[col].astype(str))

    if y.dtype == 'object':
        y = le.fit_transform(y)

    balanced = len(np.unique(y)) > 1 and np.bincount(y).min() / np.bincount(y).max() > 0.3

    if not balanced:
        st.warning("Target class is imbalanced. Applying SMOTE...")
        sm = SMOTE()
        X, y = sm.fit_resample(X, y)

    return X, y, target_col

# === Visualization Agent ===
def visualize_and_insight(df):
    pdf_path = "eda_report.pdf"
    with PdfPages(pdf_path) as pdf:
        st.write("Generating visual insights...")
        insights = []
        cat_cols = df.select_dtypes(include='object').columns
        num_cols = df.select_dtypes(include=np.number).columns

        for col in num_cols:
            fig = plt.figure(figsize=(11, 5))
            gs = GridSpec(1, 2, width_ratios=[2, 1])
            ax1 = fig.add_subplot(gs[0])
            df[col].hist(ax=ax1, bins=20, color='skyblue')
            ax1.set_title(f"Histogram of {col}")
            ax2 = fig.add_subplot(gs[1])
            ax2.axis('off')
            raw = ask_agent(f"Explain histogram of column '{col}' in 3 short bullet points for business clients.")
            points = '\n'.join([f"• {line}" for line in raw.strip().split('\n') if line])
            ax2.text(0, 1, points, wrap=True, fontsize=10, verticalalignment='top')
            pdf.savefig(fig)
            st.pyplot(fig)
            plt.close()

        for col in cat_cols:
            fig = plt.figure(figsize=(11, 5))
            gs = GridSpec(1, 2, width_ratios=[2, 1])
            ax1 = fig.add_subplot(gs[0])
            df[col].value_counts().plot(kind='bar', ax=ax1, color='orange')
            ax1.set_title(f"Bar Chart of {col}")
            ax2 = fig.add_subplot(gs[1])
            ax2.axis('off')
            raw = ask_agent(f"Explain bar chart of column '{col}' in 3 short bullet points for clients.")
            points = '\n'.join([f"• {line}" for line in raw.strip().split('\n') if line])
            ax2.text(0, 1, points, wrap=True, fontsize=10, verticalalignment='top')
            pdf.savefig(fig)
            st.pyplot(fig)
            plt.close()

    return pdf_path

# === Model Runner with Multiple Test Sizes ===
class ModelRunner:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.models = [
            LogisticRegression(),
            RandomForestClassifier(),
            GradientBoostingClassifier(),
            SVC(),
            KNeighborsClassifier(),
            xgb.XGBClassifier()
        ]
        self.classification = len(np.unique(y)) <= 10
        self.best_model = None
        self.best_info = {}

    def run(self):
        best_score = 0
        best_info = {}
        best_model = None
        for size in [0.1, 0.2, 0.25, 0.3]:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=size, random_state=42)
            for model in self.models:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_info = {
                        'Model': model.__class__.__name__,
                        'Score': score,
                        'Type': 'Classification' if self.classification else 'Regression',
                        'Test Size': f"{int(size*100)}%"
                    }
        self.best_model = best_model
        self.best_info = best_info
        return best_model, best_info

    def save_best_model(self):
        with open("best_model.pkl", "wb") as f:
            pickle.dump(self.best_model, f)

# === Dummy Prediction Interface ===
def prediction_interface(model, X):
    st.write("Prediction interface is under development.")

# === App Runner ===
st.set_page_config(page_title="Agentic AutoML AI", layout="wide")
st.title("🤖 Multi-Agent AutoML System")

client_email = st.sidebar.text_input("Enter Client Email")

st.markdown(f"**Agent 1 - {AGENT_NAMES['ingestion']}**")
df = ingest_data()

if df is not None:
    st.markdown(f"**Agent 2 - {AGENT_NAMES['preprocess']}**")
    X, y, target = preprocess_data(df)

    st.markdown(f"**Agent 3 - {AGENT_NAMES['visualize']}**")
    pdf_path = visualize_and_insight(df)
    with open(pdf_path, "rb") as f:
        st.download_button("📥 Download Visual Report", f, file_name="Insights_Report.pdf")

    if client_email:
        eda_summary = f"""
Dear Client,

Our system has completed the initial analysis of your dataset. Here are the key observations:

- ❗ Potential data quality issues found (missing values or outliers)
- 🧹 Visuals attached for your review (see insights)

Please confirm if you'd like us to proceed with data cleaning and model training.

Regards,
Akash
        """
        send_email_report("Initial Data Quality Report", eda_summary, client_email, [pdf_path])
        st.warning("Initial report emailed to client for confirmation before continuing.")

    proceed = st.checkbox("✅ Client confirmed. Proceed with model training?")
    if proceed:
        st.markdown(f"**Agent 4 - {AGENT_NAMES['model']}**")
        model_runner = ModelRunner(X, y)
        best_model, best_info = model_runner.run()
        st.success(f"Best Model: {best_info['Model']} | Score: {best_info['Score']:.2f} | Test Size: {best_info['Test Size']}")
        model_runner.save_best_model()

        if client_email:
            model_summary = f"""
Dear Client,

The AutoML process is complete. Here are the results:

✅ Best Model: {best_info['Model']}
📈 Score: {best_info['Score']:.2f}
📊 Type: {best_info['Type']}
🔎 Test Size: {best_info['Test Size']}

Thank you for using our AI service.

Regards,
Akash
"""
            send_email_report("Final AutoML Model Report", model_summary, client_email)
            st.info("📬 Final report emailed to client.")

        st.markdown(f"**Agent 5 - {AGENT_NAMES['predict']}**")
        prediction_interface(best_model, X)
