# Air-Quality-Index-AQI-Prediction-System
End-to-End Automated Machine Learning Pipeline.

## Objective

Predict the **Air Quality Index (AQI) in Karachi** for the next **3 days (72 hours)** using a **100% serverless architecture**.

This project demonstrates a complete **end-to-end machine learning pipeline** for AQI forecasting, including:

* Automated data collection
* Feature engineering
* Model training and registry management
* Real-time AQI prediction through a Streamlit dashboard

---

## Project Architecture

The project is implemented using a fully **serverless MLOps pipeline**, consisting of:

* **Open-Meteo API** – Weather and pollutant data source
* **Hopsworks Feature Store** – Feature storage and model registry
* **GitHub Actions** – Automated data ingestion and model training
* **Streamlit** – Interactive web dashboard for AQI visualization

---

## Repository Structure

| File / Folder                              | Description                                                                                                                                     |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `.github/workflows/`                       | GitHub Actions workflows for automated ingestion and training                                                                                   |
| `open_metro_1_year.py`                     | Fetches one year of historical weather and pollutant data (Oct 2024 – Oct 2025) from Open-Meteo API and saves it as a CSV file                  |
| `open_metro_hopswork_pushed_v3.py`         | Pushes the CSV data to Hopsworks Feature Store and periodically fetches new hourly weather and pollutant records (automated via GitHub Actions) |
| `AQI_Model_training_v2.py`                 | Trains a Random Forest model for 72-hour AQI forecasting and stores the model in Hopsworks Model Registry and locally                           |
| `app_streamlit_aqi_v6.py`                  | Streamlit application that displays current AQI and 72-hour AQI predictions                                                                     |
| `karachi_weather_pollutants_2024_2025.csv` | Historical dataset generated from Open-Meteo API                                                                                                |
| `aqi_model_push/`                          | Directory containing locally saved trained models                                                                                               |
| `requirements.txt`                         | Python dependencies                                                                                                                             |
| `.gitignore`                               | Ignored files and folders                                                                                                                       |
| `.gitattributes`                           | Git LFS configuration for large files                                                                                                           |

---

## Automation Overview

| Process        | Script                             | Frequency      | Trigger        |
| -------------- | ---------------------------------- | -------------- | -------------- |
| Data Ingestion | `open_metro_hopswork_pushed_v3.py` | Every 1 hour   | GitHub Actions |
| Model Training | `AQI_Model_training_v2.py`         | Every 12 hours | GitHub Actions |

---

## Technologies Used

* Python
* Hopsworks (Feature Store and Model Registry)
* Open-Meteo API
* Streamlit
* scikit-learn (Random Forest Regressor)
* GitHub Actions (CI/CD automation)

---

## Implementation Details

### 1. Data Collection

Weather and pollutant data (temperature, humidity, PM2.5, PM10, NO₂, CO) is collected using the **Open-Meteo API**.

The script `open_metro_1_year.py` fetches one year of historical data (Oct 2024 – Oct 2025) and stores it as:

```
karachi_weather_pollutants_2024_2025.csv
```

---

### 2. Data Ingestion and Feature Store Management

The script `open_metro_hopswork_pushed_v3.py`:

* Pushes historical data to the **Hopsworks Feature Store**
* Fetches new hourly data updates
* Appends the latest records automatically

This script is executed **hourly via GitHub Actions**, ensuring the system always has up-to-date features.

---

### 3. Model Training

Model training is handled by `AQI_Model_training_v2.py`:

* Pulls data from the Hopsworks Feature Store
* Trains a **Random Forest Regressor**
* Predicts AQI for the next **72 hours**
* Stores trained models:

  * Locally in `aqi_model_push/`
  * In the **Hopsworks Model Registry**

Training is automated and runs every **12 hours** via GitHub Actions.

---

### 4. Deployment and Visualization

The trained model is deployed using a **Streamlit web application** (`app_streamlit_aqi_v6.py`).

The dashboard displays:

* Current AQI values
* AQI predictions for the next 72 hours
* Graphical trends for better interpretability

---

### 5. Continuous Automation

The entire pipeline is automated using **GitHub Actions**, enabling:

* Hourly data ingestion
* Periodic model retraining
* Fully serverless and maintenance-free operation

---

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/72004/AirQualityPrediction.git
cd AirQualityPrediction

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app_streamlit_aqi_v6.py
```

---

## Output

The Streamlit dashboard provides:

* Current AQI value
* 72-hour AQI forecast
* Visual AQI trends and insights

<img width="1202" height="460" alt="image" src="https://github.com/user-attachments/assets/b2b45f18-c73c-4b64-8e0a-4469f64f995a" />

<img width="1050" height="327" alt="image" src="https://github.com/user-attachments/assets/da070706-851c-423f-b5bf-2f9be0b6d545" />


---

