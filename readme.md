# Terrain Detection Web App

This is a Flask-based AI project for classifying terrain types from images using a trained deep learning model. 

## Features
- Image upload
- Terrain classification using a `.h5` model
- Roughness and slipperiness prediction
- DevOps enabled via GitHub Actions and Docker

## Usage
```bash
docker build -t terrain-detector .
docker run -p 5000:5000 terrain-detector
