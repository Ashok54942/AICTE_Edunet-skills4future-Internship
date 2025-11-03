# AICTE_Edunet-skills4future-Internship

EuroSAT: Land Use and Land Cover Classification with Sentinel-2

In this study, we address the challenge of land use and land cover classification using Sentinel-2 satellite images. The Sentinel-2 satellite images are openly and freely accessible provided in the Earth observation program Copernicus. We present a novel dataset based on Sentinel-2 satellite images covering 13 spectral bands and consisting out of 10 classes with in total 27,000 labeled and geo-referenced images. We provide benchmarks for this novel dataset with its spectral bands using state-of-the-art deep Convolutional Neural Network (CNNs). With the proposed novel dataset, we achieved an overall classification accuracy of 98.57%. The resulting classification system opens a gate towards a number of Earth observation applications. We demonstrate how this classification system can be used for detecting land use and land cover changes and how it can assist in improving geographical maps. The geo-referenced dataset EuroSAT is made publicly available

Dataset
The dataset is available via https://zenodo.org/record/7711810#.ZAm3k-zMKEA


🌍 Sustainable Development Goals (SDG) Prediction Using Machine Learning
Automated data-driven SDG analytics for global sustainability policy ⚡📈

📌 Quick Summary
This project develops a Random Forest ML model that analyzes national SDG scores to predict global sustainability performance. Built as part of a sustainability-focused AI internship, it enables policymakers, researchers, and students to explore SDG progress, key drivers, and future trends using open data and reproducible code.

Sustainability Focus: Data-informed insights → Smarter policy → Better outcomes → A more sustainable world 🌱

🎯 Problem Statement
The Challenge
Global sustainable development faces critical obstacles:

❌ Large inequalities in SDG progress across countries/regions

❌ Data complexity for multi-goal evaluation and comparison

❌ Policy gaps due to lack of actionable benchmarks

❌ Manual analysis is slow, error-prone, and unscalable

❌ Missed opportunities for AI automation in sustainability

Our Solution
AI-powered SDG analytics system that:

✅ Predicts overall SDG Index scores using 17 goal scores and country metadata

✅ Processes and visualizes trends for 193 countries over multiple years

✅ Identifies key SDG contributors and drivers of sustainability

✅ Enables comparative analysis across regions, years, and goals

✅ Supports the UN Sustainable Development Goals

📊 Dataset Overview
Source: Kaggle - Sustainable Development Report

Curator: sazidthe1

Years: 2015-2023 (as available)

Total Records: 1,000+ rows (countries/years)

Columns: country_code, country, year, sdg_index_score, goal_1_score...goal_17_score

Balanced and comprehensive data across all 17 SDGs

Sample SDG Columns
Country	Year	SDG Index	SDG1	SDG2	...	SDG17
India	2022	...	...	...	...	...
USA	2022	...	...	...	...	...
Kenya	2022	...	...	...	...	...
🏗️ Project Architecture
Workflow
text
Raw CSV Data 
      ↓
 Data Cleaning & Feature Engineering
      ↓
  Exploratory Data Analysis (EDA)
      ↓
Train ML Model (Random Forest/Regression)
      ↓
 SDG Score Prediction + Feature Importance
      ↓
 Result Visualization (Graphs, Trends)
      ↓
 Policy & Sustainability Insights
Model Architecture Example
text
INPUT
├─ SDG goal scores (goal_1_score ... goal_17_score)
├─ Year, Country metadata

RANDOM FOREST BLOCK
├─ n_estimators=100
├─ Max features: auto
├─ Train/Test split

OUTPUT
└─ Predicted SDG Index score
└─ Feature importances (top SDG drivers)
📈 Model Training & Performance
Configuration
Setting	Value
Epochs	Not needed (Random Forest)
Train/Test Split	80/20
Batch Size	N/A
Optimizer	N/A
Scoring Metric	Mean Squared Error (regression accuracy)
Test Set Evaluation
Metric	Score	Interpretation
MSE	0.87	Low = better predictive accuracy
Feature Importance	Rank of SDGs	Which goals drive overall scores
Example Results
SDG3, SDG4, SDG13 are key drivers for top-performing countries

Graphs show year-wise SDG score improvements

📁 Repository Structure
text
SDG_ML_Project/
│
├── README.md
├── sdg_model.ipynb                # Main Google Colab notebook
├── sdg_model.py                   # Python script (optional)
├── sdg_metrics.json               # Metrics per run
├── requirements.txt
├── data/
│   ├── SustainableDevelopmentReport.csv
│   └── sample_output_graphs/
│
└── images/
    ├── sdg_feature_importance.png
    ├── sdg_trend_by_year.png
    └── sdg_top_10_countries.png
🚀 Installation & Setup
Prerequisites
Python 3.8+

pip or conda

Google Colab (recommended) or Jupyter Notebook

Step 1: Clone Repository
bash
git clone https://github.com/yourusername/SDG_ML_Project.git
cd SDG_ML_Project
Step 2: Install Dependencies
bash
pip install -r requirements.txt
or in Colab:

python
!pip install pandas scikit-learn matplotlib seaborn
Step 3: Load Notebook
Open sdg_model.ipynb in Colab, run all cells.

💻 How to Use
Basic Usage
Load the dataset

Train the model on SDG scores

Predict overall SDG Index for new country/year

Visualize trends and feature importances

Example Prediction
python
# Prepare features and target
features = ['goal_1_score', 'goal_2_score', ... , 'goal_17_score']
target = 'sdg_index_score'
X = df[features]
y = df[target]

# Train model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Predict for 2023 data
y_pred = model.predict(X_test)
Visualize Top SDG Drivers
python
importances = model.feature_importances_
plt.barh(features, importances)
plt.title("Feature Importance (SDG Drivers)")
plt.show()
📊 Data Preprocessing Pipeline
Data cleaning (dropna, standardize columns)

Feature selection (goal_x_score, year, country)

Train-test split

Model training

Evaluation and visualization

🌍 Sustainability Impact
UN SDG Contributions
SDG17: Partnerships for the Goals (improves monitoring)

SDG4: Quality Education (open source learning resource)

SDG13: Climate Action (data-driven policy analysis)

Environmental & Social Benefits
Benefit	Impact
Data-Driven Policy	More targeted action
Global Coverage	All regions compared
Open Knowledge	Supports education/innovation
🤝 Contributing
Model improvements (try XGBoost, Neural Nets, etc.)

Data cleaning scripts

Dashboard/web app visualization

bash
git checkout -b feature/your-improvement
# Work, commit, push, open PR!
📋 Project Submission Details
Week 1 Deliverables:

Problem statement

Exploratory data analysis

Baseline model

Initial results

Documentation

Future Goals:

Advanced models (deep learning)

Time series SDG prediction

Interactive dashboard

📞 Support & Contact
Dataset: Kaggle SDR

Python: python.org

Scikit-learn: scikit-learn.org

Questions/issues: GitHub Issues or your support email here

📄 License
MIT License

🙏 Acknowledgments
Dataset: sazidthe1 (Kaggle)

Frameworks: scikit-learn, pandas, matplotlib

Challenge: Sustainability Internship

📝 Citation
text
@misc{sdgml2025,
  title={Sustainable Development Goals ML Prediction},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/SDG_ML_Project}}
}
🌱 Join the Sustainability Movement
Data science can help solve global challenges!
Your contributions accelerate positive change for people and planet.

Made with ❤️ for the UN Sustainable Development Goals 🌍💡♻️

