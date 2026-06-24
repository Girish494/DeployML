# Health Insurance Multiclass Classification System

## Overview

This project is an end-to-end Machine Learning application that predicts the most suitable health insurance category for a customer using demographic and health-related information. The solution uses a Deep Learning Artificial Neural Network (ANN) model built with PyTorch and is deployed as a REST API using FastAPI.

The project follows a production-oriented workflow including data preprocessing, model training, API development, Docker containerization, and AWS deployment.

---

## Dataset

* Dataset Size: 100,000 Records
* Problem Type: Multiclass Classification
* Features:

  * Age
  * Gender
  * BMI
  * Income Level
  * Smoking Status
  * Medical History
  * Occupation
  * City Tier
  * Other demographic and health-related attributes

---

## Tech Stack

### Machine Learning & Deep Learning

* Python
* PyTorch
* Scikit-Learn
* Pandas
* NumPy

### Backend

* FastAPI
* Uvicorn
* Pydantic

### Deployment

* Docker
* AWS

---

## Model Performance

| Metric            | Score  |
| ----------------- | ------ |
| Accuracy          | 95.14% |
| Macro F1 Score    | 0.95   |
| Weighted F1 Score | 0.95   |
| Training Loss     | 0.0910 |
| Testing Loss      | 0.0926 |

---

## Project Workflow

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Data Preprocessing
5. Feature Scaling
6. ANN Model Training
7. Model Evaluation
8. API Development with FastAPI
9. Docker Containerization
10. AWS Deployment

---

## API Features

### Health Check Endpoint

Verifies that the API service is running successfully.

### Single Prediction Endpoint

Predicts the insurance category for a single customer.

### Batch Prediction Endpoint

Supports predictions for multiple customer records in a single request.

### Model Information Endpoint

Returns model details and performance metrics.

### Swagger Documentation

Interactive API testing using FastAPI's built-in Swagger UI.

---

## Project Structure

```text
HealthInsuranceProject/
│
├── API/
│   ├── main.py
│   ├── model.py
│   ├── utils.py
│   └── schemas.py
│
├── Model/
│   ├── best_model.pt
│   ├── preprocessor.pkl
│   └── model_info.pkl
│
├── notebooks/
│   └── model_training.ipynb
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

### Clone Repository

```bash
git clone https://github.com/Girish494/DeployML.git
```

### Navigate to Project

```bash
cd DeployML/HealthInsuranceProject
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run FastAPI Application

```bash
uvicorn API.main:app --reload
```

### Open Swagger Documentation

```text
http://127.0.0.1:8000/docs
```

---

## Docker Deployment

### Build Docker Image

```bash
docker build -t health-insurance-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 health-insurance-api
```

---

## AWS Deployment

The application is containerized using Docker and deployed on AWS for scalable and production-ready access.

---

## Future Improvements

* React Frontend Dashboard
* JWT Authentication
* Database Integration
* Prediction History Tracking
* CI/CD Pipeline
* Kubernetes Deployment
* Model Monitoring
* MLOps Workflow Automation

---

## GitHub Repository

Repository Link:

https://github.com/Girish494/DeployML/tree/main/HealthInsuranceProject

---

## Author

**Girish Pawar**

Full Stack Engineer | Machine Learning Enthusiast | Deep Learning Practitioner

Focused on building scalable AI-powered applications using Machine Learning, Deep Learning, FastAPI, Docker, and Cloud Technologies.
