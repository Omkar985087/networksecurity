🛡️ Network Security Phishing Detection System (MLOps)

An end-to-end production-grade Machine Learning system for detecting phishing websites using MLOps practices. This project includes data ingestion, training pipeline, model tracking, Dockerization, CI/CD with GitHub Actions, AWS ECR deployment, and FastAPI serving.

🚀 Features

✅ End-to-end ML pipeline (Ingestion → Validation → Transformation → Training → Evaluation → Prediction)

✅ MongoDB for dataset storage

✅ MLflow + DagsHub for experiment tracking

✅ FastAPI backend for training & prediction

✅ Dockerized application

✅ CI/CD using GitHub Actions

✅ AWS ECR for container registry

✅ Self-hosted runner on AWS EC2

✅ S3 bucket support for artifact storage

✅ Production-ready architecture

🏗️ Architecture

Data (MongoDB)
     ↓
Data Ingestion
     ↓
Validation → Drift Check
     ↓
Transformation → Preprocessing
     ↓
Model Training → Model Selection
     ↓
MLflow Tracking
     ↓
Artifacts → S3
     ↓
FastAPI App
     ↓
Docker → AWS ECR → EC2 Deployment

🧰 Tech Stack

Python

FastAPI

Scikit-learn

MongoDB

MLflow

DagsHub

Docker

GitHub Actions

AWS (EC2, ECR, S3)

Boto3

Uvicorn

📁 Project Structure

networksecurity/
│── components/
│── pipeline/
│── entity/
│── constants/
│── utils/
│── logging/
│── exception/
│── app.py
│── Dockerfile
│── requirements.txt
│── main.yml (GitHub Actions)
