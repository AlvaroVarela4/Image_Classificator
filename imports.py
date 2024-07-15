
# Librerias

# Parte preanalisis
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
from sklearn.preprocessing import LabelEncoder
# Parte mas de modelizacion 
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data Augmentation
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # Label Encoding 
import numpy as np
