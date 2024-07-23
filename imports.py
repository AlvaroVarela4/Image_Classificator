
# LIBRERIAS UTILIZADAS EN EL PROYECTO #
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np  
from PIL import Image 
import os 
import random
from skimpy import skim  
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 # Modelo base para el proyecto 
from tensorflow.keras.models import Model, load_model 
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout # Capas de Pooling para la CNN 
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data Augmentation
from tensorflow.keras.optimizers import Adam  # optimizadores 
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_curve, auc  # Metricas de evaluacion 
from sklearn.model_selection import train_test_split  # Cross Validation 
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical 
import numpy as np
from tqdm import tqdm 
import imageio
from imgaug import augmenters as iaa
