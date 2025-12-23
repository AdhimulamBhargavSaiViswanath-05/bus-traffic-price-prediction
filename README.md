# 🚌 Improving Bus Service with Predictions for Prices and Passenger Traffic

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

**A Machine Learning-Based Analytics Project for Bus Transportation Optimization**

[View Project](https://github.com/AdhimulamBhargavSaiViswanath-05/bus-traffic-price-prediction) · [Report Issue](https://github.com/AdhimulamBhargavSaiViswanath-05/bus-traffic-price-prediction/issues)

</div>

---

## 📌 Project Overview

This project aims to improve public bus services by analyzing and predicting: 
- **Passenger demand patterns** based on historical travel data
- **Ticket price variations** influenced by multiple factors
- **Key insights** into transportation analytics

By applying machine learning techniques, the system provides transparency into how ticket prices are calculated and how passenger traffic fluctuates over time, helping improve decision-making for both service providers and passengers.

> **Personal Learning Outcome:**  
> *This project helped me understand the stages of machine learning, including data preparation, model training, and prediction, and how ML models learn patterns to make predictions.*

---

## 📂 Project Structure

```
bus-traffic-price-prediction/
│
├── Bus_Service_Predictions/          # Main project directory
│   ├── data/                         # Dataset files
│   │   ├── raw/                      # Raw data files
│   │   └── processed/                # Cleaned and processed data
│   │
│   ├── notebooks/                    # Jupyter notebooks
│   │   ├── 01_data_exploration.ipynb        # EDA and visualization
│   │   ├── 02_data_preprocessing.ipynb      # Data cleaning
│   │   ├── 03_traffic_prediction.ipynb      # Passenger traffic model
│   │   ├── 04_price_prediction.ipynb        # Price prediction model
│   │   └── 05_dashboard.ipynb               # Interactive dashboard
│   │
│   ├── models/                       # Saved ML models
│   │   ├── traffic_model.pkl
│   │   └── price_model.pkl
│   │
│   ├── src/                          # Source code
│   │   ├── data_preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── model_training. py
│   │   └── utils.py
│   │
│   └── visualizations/               # Output plots and charts
│       ├── traffic_trends.png
│       ├── price_analysis.png
│       └── correlation_heatmap.png
│
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
├── . gitignore                        # Git ignore file
└── LICENSE                           # License information
```

---

## 🎯 Objectives

- ✅ Predict passenger traffic using historical travel data  
- ✅ Forecast bus ticket prices using machine learning models  
- ✅ Analyze key factors affecting pricing and demand  
- ✅ Provide interactive visual insights using a Jupyter dashboard  
- ✅ Improve public understanding of ticket price finalization  
- ✅ Learn end-to-end ML lifecycle implementation

---

## 🛠️ Tech Stack

### Programming & Environment
- **Language:** Python 3.8+
- **IDE:** Jupyter Notebook
- **Version Control:** Git & GitHub

### Core Libraries

| Library | Purpose |
|---------|---------|
| **NumPy** | Numerical computations and array operations |
| **Pandas** | Data manipulation and analysis |
| **Matplotlib** | Static data visualization |
| **Seaborn** | Statistical data visualization |
| **Scikit-learn** | Machine learning algorithms and tools |

### ML Algorithms Implemented
- **Linear Regression** - Baseline model
- **Ridge Regression** - Regularized linear model
- **Decision Tree Regressor** - Non-linear tree-based model
- **Random Forest Regressor** - Ensemble learning method
- **Support Vector Regressor (SVR)** - Kernel-based regression

### Techniques Applied
- Data preprocessing & normalization
- Feature engineering & selection
- Hyperparameter tuning (GridSearchCV)
- Cross-validation (5-fold CV)
- Model evaluation (MAE, MSE, RMSE, R²)

---

## 📊 Project Workflow

```
┌─────────────────────────────────────────────────────────┐
│                  1. DATA COLLECTION                     │
│         Historical bus travel and pricing data          │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│              2. DATA PREPROCESSING                      │
│   • Handle missing values                               │
│   • Remove duplicates and outliers                      │
│   • Data type conversions                               │
│   • Normalization and scaling                           │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│          3. EXPLORATORY DATA ANALYSIS (EDA)             │
│   • Passenger traffic trends                            │
│   • Price variation patterns                            │
│   • Correlation analysis                                │
│   • Feature distribution analysis                       │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│            4. FEATURE ENGINEERING                       │
│   • Create new features                                 │
│   • Feature selection                                   │
│   • Encode categorical variables                        │
│   • Train-test split                                    │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│              5. MODEL TRAINING                          │
│   • Train multiple regression models                    │
│   • Hyperparameter tuning                               │
│   • Cross-validation                                    │
│   • Model comparison                                    │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│          6. PREDICTION & EVALUATION                     │
│   • Make predictions on test data                       │
│   • Evaluate using MAE, MSE, RMSE, R²                   │
│   • Error analysis                                      │
│   • Model optimization                                  │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│           7. VISUALIZATION & INSIGHTS                   │
│   • Interactive dashboard                               │
│   • Prediction vs Actual plots                          │
│   • Feature importance charts                           │
│   • Business insights                                   │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Basic understanding of machine learning concepts

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AdhimulamBhargavSaiViswanath-05/bus-traffic-price-prediction. git
   cd bus-traffic-price-prediction
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

5. **Navigate to the notebooks folder** and run them in sequence: 
   - Start with `01_data_exploration.ipynb`
   - Follow through `02_data_preprocessing.ipynb`
   - Continue with traffic and price prediction notebooks
   - Explore the interactive dashboard in `05_dashboard.ipynb`

---

## 📈 Key Features

### 1. **Passenger Traffic Prediction** 🚶
- Predicts daily/weekly passenger counts
- Identifies peak travel times
- Analyzes seasonal trends
- Helps optimize bus frequency

### 2. **Ticket Price Forecasting** 💰
- Predicts ticket prices based on multiple factors
- Factors considered: 
  - Route distance
  - Time of day
  - Day of week
  - Season
  - Demand patterns
- Provides transparent pricing insights

### 3. **Interactive Dashboard** 📊
- Real-time prediction interface
- Visual comparison of actual vs predicted values
- Feature importance analysis
- Model performance metrics

### 4. **Data Insights** 🔍
- Correlation heatmaps
- Distribution plots
- Time-series analysis
- Statistical summaries

---

## 📉 Model Evaluation Summary

### 🏆 Best Performing Models

The **Random Forest Regressor** emerged as the best model with exceptional performance:
- **R² Score:  0.9685** (Before Tuning) / **0.9521** (After Tuning)
- **RMSE: 11.77** (Before Tuning) / **14.52** (After Tuning)
- Explains **96.85%** of the variance in the data

### 📊 Complete Model Comparison

#### Before Tuning (Initial Training)

| Model | MAE | MSE | RMSE | R² Score | Performance |
|-------|-----|-----|------|----------|-------------|
| **Random Forest Regressor** 🥇 | **8.76** | **138.66** | **11.78** | **0.9685** | ⭐⭐⭐⭐⭐ Excellent |
| **Decision Tree Regressor** 🥈 | 12.62 | 318.80 | 17.86 | 0.9275 | ⭐⭐⭐⭐ Very Good |
| **Linear Regression** 🥉 | 22.17 | 946.41 | 30.76 | 0.7849 | ⭐⭐⭐ Good |
| **Support Vector Regressor** | 45.60 | 3162.49 | 56.24 | 0.2812 | ⭐ Poor |

#### After Hyperparameter Tuning

| Model | MAE | MSE | RMSE | R² Score | Performance |
|-------|-----|-----|------|----------|-------------|
| **Random Forest** 🥇 | **10.92** | **210.80** | **14.52** | **0.9521** | ⭐⭐⭐⭐⭐ Excellent |
| **Decision Tree** 🥈 | 14.79 | 435.92 | 20.88 | 0.9009 | ⭐⭐⭐⭐ Very Good |
| **Ridge Regression** 🥉 | 22.16 | 946.08 | 30.76 | 0.7850 | ⭐⭐⭐ Good |
| **Support Vector Regressor** | 438.45 | 393334.34 | 627.16 | -88.3973 | ❌ Failed |

#### Cross-Validation Results (5-Fold)

| Model | MAE (CV) | MSE (CV) | RMSE (CV) | R² Score (CV) | Stability |
|-------|----------|----------|-----------|---------------|-----------|
| **Random Forest** 🥇 | **8.72** | **159.91** | **12.65** | **0.9620** | ✅ Highly Stable |
| **Decision Tree** 🥈 | 12.56 | 322.02 | 17.95 | 0.9235 | ✅ Stable |
| **Linear Regression** 🥉 | 22.35 | 962.79 | 31.03 | 0.7710 | ✅ Stable |
| **Support Vector Regressor** | 44.67 | 3030.69 | 55.05 | 0.2789 | ⚠️ Unstable |

---

### 📊 Evaluation Metrics Explained

| Metric | Description | Best Value | Interpretation |
|--------|-------------|------------|----------------|
| **MAE** (Mean Absolute Error) | Average absolute difference between predicted and actual values | Lower is better | Average prediction error in original units |
| **MSE** (Mean Squared Error) | Average of squared differences | Lower is better | Penalizes larger errors more heavily |
| **RMSE** (Root Mean Squared Error) | Square root of MSE | Lower is better | Standard deviation of prediction errors |
| **R² Score** | Proportion of variance explained by the model | Closer to 1 is better | 1.0 = Perfect fit, 0.0 = No predictive power |

---

### 🔍 Key Insights from Model Evaluation

1. **Random Forest Dominance** 🌲
   - Consistently outperformed all other models
   - Achieved **96.85% accuracy** in explaining data variance
   - Most reliable for both traffic and price prediction

2. **Decision Tree Performance** 🌿
   - Second-best performer with **92.75% accuracy**
   - Good balance between complexity and interpretability
   - Useful for understanding feature importance

3. **Linear Models** 📏
   - Linear and Ridge Regression showed moderate performance (~78% R²)
   - Useful for baseline comparison
   - Limited by linear assumption of relationships

4. **SVR Failure** ⚠️
   - Support Vector Regressor performed poorly
   - Negative R² score after tuning indicates catastrophic failure
   - Not suitable for this dataset/problem

5. **Model Stability** ✅
   - Cross-validation results show Random Forest is highly stable
   - Minimal variance between training and CV scores
   - Indicates good generalization capability

---

### 🎯 Final Model Selection

**Chosen Model:  Random Forest Regressor**

**Justification:**
- ✅ Highest R² score (0.9685)
- ✅ Lowest MAE (8.76) and RMSE (11.78)
- ✅ Excellent cross-validation performance
- ✅ Stable across different data splits
- ✅ Handles non-linear relationships well
- ✅ Robust to outliers

**Practical Implications:**
- The model can predict bus ticket prices with an average error of only **₹8.76**
- Explains **96.85%** of price variations
- Reliable enough for production deployment

---

## 📚 Key Learnings

Through this project, I gained hands-on experience in:

### Technical Skills:
- ✅ **End-to-end machine learning lifecycle** implementation
- ✅ **Data preprocessing** techniques (handling missing data, outliers, normalization)
- ✅ **Feature engineering** and selection strategies
- ✅ **Model training, tuning, and evaluation**
- ✅ **Hyperparameter optimization** using GridSearchCV
- ✅ **Cross-validation** for robust model assessment
- ✅ **Real-world application** of predictive analytics
- ✅ **Dashboard-based data visualization** using Jupyter widgets

### ML Concepts: 
- ✅ Understanding how **ML models learn patterns** from data
- ✅ Importance of **data quality** in model performance
- ✅ **Overfitting vs Underfitting** and regularization
- ✅ **Ensemble methods** (Random Forest) vs single models
- ✅ **Model comparison** and selection strategies
- ✅ **Evaluation metrics** interpretation (MAE, MSE, RMSE, R²)
- ✅ Why some models fail (SVR case study)

### Practical Insights:
- ✅ **Random Forest** consistently outperforms other algorithms for tabular data
- ✅ **Hyperparameter tuning** doesn't always improve performance
- ✅ **Cross-validation** is crucial for assessing model stability
- ✅ **Feature engineering** is more important than algorithm selection
- ✅ **Domain knowledge** helps in feature creation and interpretation

### Soft Skills:
- ✅ Problem-solving and analytical thinking
- ✅ Documentation and code organization
- ✅ Communicating technical insights to non-technical audiences
- ✅ Systematic experimentation and result tracking

---

## 🔍 Motivation

The motivation behind this project was to address the lack of transparency in: 

1. **How bus ticket prices are determined**  
   Many passengers don't understand why prices vary for the same route at different times.

2. **What factors influence passenger demand**  
   Understanding demand patterns can help optimize bus schedules and reduce wait times.

3. **Data-driven decision making in public transport**  
   By presenting insights in a simple and visual manner, this project makes transport analytics accessible to everyone.

---

## 🏫 Academic Details

- **Institution:** Vasireddy Venkatadri Institute of Technology (VVIT), Nambur  
- **Duration:** May 2025 – July 2025  
- **Course:** Machine Learning / Data Science Project

### Project Guide
**Mrs. B. Lalitha Rajeswari, M.Tech (Ph.D)**  
*Assistant Professor*  
Vasireddy Venkatadri Institute of Technology (VVIT)

---

## 🔮 Future Enhancements

- [ ] **Real-time Data Integration**  
  Connect with live bus data APIs for real-time predictions

- [ ] **Advanced Time-Series Models**  
  Implement ARIMA, LSTM, and Prophet for better forecasting

- [ ] **Deep Learning Models**  
  Explore Neural Networks for complex pattern recognition

- [ ] **Route-Level Analysis**  
  Extend predictions to specific routes and cities

- [ ] **Web Application Deployment**  
  Create a Flask/Django web app with interactive UI

- [ ] **Mobile App Integration**  
  Develop companion mobile app for passengers

- [ ] **Government Integration**  
  Partner with transport authorities for system implementation

- [ ] **Weather Data Integration**  
  Include weather conditions as a predictive feature

- [ ] **Dynamic Pricing Model**  
  Implement surge pricing algorithms based on demand

- [ ] **Explainable AI (XAI)**  
  Add SHAP values for model interpretability

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project: 

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is intended for **academic and educational purposes**.   
Feel free to use the code for learning and research. 

---

## 📞 Contact

**Adhimulam Bhargav Sai Viswanath**

- 🐱 GitHub: [@AdhimulamBhargavSaiViswanath-05](https://github.com/AdhimulamBhargavSaiViswanath-05)
- 💼 LinkedIn: [adhimulambhargavsaiviswanath](https://www.linkedin.com/in/adhimulambhargavsaiviswanath/)
- 📧 Email: bhargavsaiadhimulam12@gmail.com

**Project Link:** [https://github.com/AdhimulamBhargavSaiViswanath-05/bus-traffic-price-prediction](https://github.com/AdhimulamBhargavSaiViswanath-05/bus-traffic-price-prediction)

---

## 🙏 Acknowledgments

- **Mrs. B. Lalitha Rajeswari** for project guidance and mentorship
- **VVIT** for providing the opportunity and resources
- **Scikit-learn documentation** for comprehensive ML tutorials
- **Kaggle community** for dataset inspiration
- **Stack Overflow** for troubleshooting support

---

<div align="center">

**Version:** 1.0.0  
**Last Updated:** July 2025

---

**Made with 🧠 and 💻 for advancing public transportation analytics**

⭐ Star this repository if you found it helpful!

</div>
