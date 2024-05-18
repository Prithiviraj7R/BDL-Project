## Image Classification Project
The goal of this project is to build a deep learning model tailored specifically for image classification task. Our approach includes the integration of cutting-edge MLOps tools such as DVC, Git, MLflow, Docker, FastAPI, Prometheus, and Grafana.

This project encompasses several key components:

1. **Deep Learning Model:** We will develop a robust deep learning architecture capable of accurately categorizing images into distinct classes. This involves implementing neural network models, such as convolutional neural networks (CNNs).

2. **MLOps Tools Integration:**
   - **DVC (Data Version Control):** DVC will be utilized for building data processing pipelines which includes image resizing and normalization.
   - **Git:** Git version control ensures the reproducibility and traceability of our model development process, allowing for easy collaboration and rollback to previous iterations if necessary.
   - **MLflow:** MLflow will serve as a comprehensive platform for experiment tracking and model tuning. 
   - **Docker:** Docker containers will be leveraged to encapsulate our deep learning environment, ensuring consistency and portability across different systems. It will be used to host the FastAPI app to use the trained model for classification.
   - **FastAPI:** FastAPI will enable us to build robust and scalable REST APIs for our deep learning model, facilitating seamless integration with other systems or applications.
   - **Prometheus and Grafana:** These monitoring tools will provide valuable insights into the performance and health of our deployed model, allowing for proactive optimization and troubleshooting.

3. **Deployment and Monitoring:** Once the model is trained and validated, we will deploy it using Docker containers and FastAPI endpoints for real-time inference. Prometheus and Grafana will then monitor the model's performance metrics, ensuring reliability and performance optimization over time.

## Deployment

### Cloning the Repository
1. Clone this repository to your local machine: `git clone https://github.com/Prithiviraj7R/BDL-Project.git`
2. Navigate to the project directory: `cd BDL_Project`
3. For Model selection using MLflow, run model_tuning.ipynb and `mlflow server --host 127.0.0.1 --port 8080`
   
### Building and Running the Docker Image
1. Build the Docker image using Docker Compose: `docker compose build`
2. Run and host the built image using Docker Compose: `docker compose up`


### Accessing Services
- FastAPI will be available at: `http://localhost:8002/docs`
- Prometheus can be accessed at: `http://localhost:9090`
- Grafana is accessible via: `http://localhost:3000`
