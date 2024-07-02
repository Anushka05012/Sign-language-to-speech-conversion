#Inference
import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)