# SnapEat Neural Network

This project is an API implemented with FastAPI and Docker. It is deployed on an EC2 instance.

## Endpoint for Predictions

The endpoint to make predictions is `http://3.19.223.112/file/`.

### Parameters

- `file`: Required parameter.
- `x`: Optional parameter.
- `y`: Optional parameter.
- `width`: Optional parameter.
- `height`: Optional parameter.

## Deployment

The application is containerized using Docker and deployed on an EC2 instance for scalability and easy management.

## Usage

To make a prediction, send a POST request to the endpoint with the required and optional parameters as form data.


 ## Install triton
 pip install https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/triton-2.0.0-cp310-cp310-win_amd64.whl

