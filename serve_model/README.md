# ML Model Serving with MLflow, Docker, and Kubernetes

This project demonstrates how to serve a machine learning model using MLflow, Docker, and Kubernetes. It includes steps for training a model, registering it with MLflow, building a Docker image, and deploying it to a Kubernetes cluster.

## Installation

To set up the project locally, follow these steps:

1. **Setup Minikube and kubectl**:

   ```bash
   make setup
   ```
2. **Train Model**:

   ```bash
   python train.py
   ```
3. **Register the best model at MLFlow and build as Docker image**:

   ```bash
   python register_best_model.py
   ```

3. **Push Docker image to Docker hub**:

   ```bash
   make push_image
   ```
   
4. **Start Minikube and Setup Cluster**:

     
   ```bash
   make start_minikube
   make setup_cluster
   ```

