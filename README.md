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
├── .gitignore                        # Git ignore file
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

### ML Techniques Applied
- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- Data preprocessing & normalization
- Feature engineering & selection
- Cross-validation
- Model evaluation (RMSE, MAE, R²)

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
│   • Evaluate using RMSE, MAE, R²                        │
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

## 📉 Model Performance

| Model | Traffic Prediction | Price Prediction |
|-------|-------------------|------------------|
| **Linear Regression** | R² = 0.XX | R² = 0.XX |
| **Decision Tree** | R² = 0.XX | R² = 0.XX |
| **Random Forest** | R² = 0.XX | R² = 0.XX |

*Note: Replace XX with actual scores from your models*

### Evaluation Metrics Used: 
- **R² Score** - Coefficient of determination
- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error
- **Cross-Validation Score** - 5-fold CV

---

## 📚 Key Learnings

Through this project, I gained hands-on experience in: 

### Technical Skills:
- ✅ **End-to-end machine learning lifecycle** implementation
- ✅ **Data preprocessing** techniques (handling missing data, outliers, normalization)
- ✅ **Feature engineering** and selection strategies
- ✅ **Model training, tuning, and evaluation**
- ✅ **Real-world application** of predictive analytics
- ✅ **Dashboard-based data visualization** using Jupyter widgets

### ML Concepts:
- ✅ Understanding how **ML models learn patterns** from data
- ✅ Importance of **data quality** in model performance
- ✅ **Overfitting vs Underfitting** and regularization
- ✅ **Model comparison** and selection strategies
- ✅ **Hyperparameter tuning** for optimization

### Soft Skills:
- ✅ Problem-solving and analytical thinking
- ✅ Documentation and code organization
- ✅ Communicating technical insights to non-technical audiences

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
