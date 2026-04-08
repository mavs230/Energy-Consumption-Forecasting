# Energy Consumption Prediction using Linear Regression

## Project Overview
This repository contains a predictive model for building energy consumption (kWh). It focuses on rigorous data cleaning and feature engineering to improve the accuracy of a Linear Regression model.

## Technical Highlights
- **Feature Engineering**: Implemented **One-Hot Encoding** for categorical variables to prevent false numerical ordering.
- **Leakage Prevention**: Feature scaling was fit strictly on training data and applied to the test set.
- **Evaluation**: Utilized R² and MAE (Mean Absolute Error) for performance benchmarking.

## Real-World Data Pipeline Integration
For a specialized energy management system, this script acts as the analytical engine:
- **Streaming Data**: Real-time energy consumption metrics would be streamed via MQTT or Kafka from IoT sensors in the building.
- **Feature Store**: Categorical data (Building Type) would be fetched from a Feature Store to ensure consistency across different models.
- **Automated Retraining**: A CI/CD pipeline would monitor the **MAE (Mean Absolute Error)**; if accuracy drops below a certain threshold due to seasonal changes, the pipeline automatically triggers a re-run of this script to update coefficients.

## How to Run
```bash
python "LR energy consumption.py"
