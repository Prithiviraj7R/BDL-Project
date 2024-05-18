from fastapi import FastAPI, UploadFile, File, Request
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from io import BytesIO
import os
import time
from src.app.util import CNNModel1

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
import psutil

app = FastAPI()
Instrumentator().instrument(app).expose(app)

# Custom metrics for tracking
ip_counter = Gauge('api_requests_by_ip', 'Number of requests by IP address', ['ip'])
api_runtime_gauge = Gauge('api_runtime', 'API runtime in milliseconds')
api_tl_time_gauge = Gauge('api_tl_time', 'API T/L time in microseconds per character')
api_memory_usage_gauge = Gauge('api_memory_usage', 'API memory usage in bytes')
api_cpu_usage_gauge = Gauge('api_cpu_usage', 'API CPU utilization rate')
api_network_bytes_gauge = Gauge('api_network_bytes', 'API network I/O bytes')
api_network_bytes_rate_gauge = Gauge('api_network_bytes_rate', 'API network I/O bytes rate')


def load_model(path: str):
    """
    A function to load the model
    """
    model_path = os.path.join(os.getcwd(), 'src', 'app', path)
    model = CNNModel1()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def format_image(image: bytes) -> np.array:
    """
    A function to process the uploaded image.
    It resizes the image into 224*224
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_pil = Image.open(BytesIO(image))
    img_transformed = transform(img_pil)

    img_with_batch = img_transformed.unsqueeze(0)

    height, width = img_pil.size
    input_length = height * width

    return img_with_batch, input_length

def predict_class(model, data: list) -> str:
    """
    A function to predict the image class given
    an array representaing the image.
    """
    class_labels = ['triceratops', 'arctic wolf', 'rhinoceros beetle', 'gymnastics', 'soup']

    with torch.no_grad():
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)

    prediction = class_labels[predicted.item()]
    return prediction

@app.post('/predict/')
async def digit_classification(request: Request, file: UploadFile = File(...)):
    """
    Accepts the uploaded image and passes 
    the array returned by format_image function
    to predict_digit function for prediction. 
    """

    start_time = time.time()

    image = await file.read()
    data, input_length = format_image(image)

    model = load_model('best_model.pth')
    prediction = predict_class(model, data)

     # ip counter for api requests
    client_ip = request.client.host
    ip_counter.labels(ip=client_ip).inc()
    
    # runtime calculation
    final_time = time.time()
    elapsed_time = (final_time - start_time) * 1000 
    api_runtime_gauge.set(elapsed_time)
    if input_length != 0:
        tl_time = elapsed_time / input_length * 1000  
        api_tl_time_gauge.set(tl_time)

    memory_usage = (psutil.virtual_memory().used)/(1024**3)
    api_memory_usage_gauge.set(memory_usage)

    # CPU usage rate
    cpu_usage = psutil.cpu_percent(interval=1)
    api_cpu_usage_gauge.set(cpu_usage)

    # Network I/O bytes
    network_io_counters = psutil.net_io_counters()
    bytes_in = network_io_counters.bytes_recv
    bytes_out = network_io_counters.bytes_sent
    api_network_bytes_gauge.set((bytes_in + bytes_out)/1024)

    # Network I/O bytes rate
    bytes_rate_in = network_io_counters.bytes_recv / elapsed_time if elapsed_time > 0 else 0
    bytes_rate_out = network_io_counters.bytes_sent / elapsed_time if elapsed_time > 0 else 0
    api_network_bytes_rate_gauge.set((bytes_rate_in + bytes_rate_out)/1024)

    # returned class is displayed
    return {"class": prediction}
