
# LIBRERIAS UTILIZADAS EN EL PROYECTO #

# PRE- MODELIZACION (EDA + CONFIGURACIONES ADICIONALES)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np  
from PIL import Image 
import os 
import random
from skimpy import skim  # Es como un describe pero mucho mas ordenado y con mas informacion
import cv2


# MODELIZACION 
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 # Modelo base para el proyecto 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # Capas de Pooling para la CNN 
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data Augmentation
from tensorflow.keras.optimizers import Adam  # optimizadores 
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_curve, auc  # Metricas de evaluacion 
from sklearn.model_selection import train_test_split  # Cross Validation 
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical # Label Encoding 
import numpy as np
from tqdm import tqdm # barra de progreso