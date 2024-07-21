
from typing import *
from imports import *

## FUNCIONES FASE MODELIZACION ##

# 1. Funcion que verifica que las rutas de las imagenes estan correctamente construidas

def check_path(data: pd.DataFrame) -> Tuple:
    t = 0 
    f = 0  
    for path in tqdm(data['path_files'], desc='Procesando rutas'): 
        try:
            Image.open(path)
            t += 1
        except Exception as e:
            f += 1
            
    return f"{t} fotos correctas", f"{f} fotos no leidas/invalidas"

# 2. Funcion que crea rutas de las imagenes a partir

def create_path(data: pd.DataFrame) -> List: 
    created_paths = []
    for id in tqdm(data["photo_id"], desc= "Creando rutas"): 
        image_path = os.path.join("photos", f"{id}.jpg")
        try: 
            Image.open(image_path)
            created_paths.append(image_path)
        except Exception as e:
            print(f"Error al crear ruta de imagen {id}")
    return created_paths

# 3. Funcion que formatea las imagenes para que MobileNetV2 pued procesarlas. Procesa tanto las imagenes como sus categorias
#    y devuelve una tupla de dos arrays: images de dim = 4 (muestras, h, w, channels) y labels de dim = 2 (muestras, categorias)

def image_label_load(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]: 
    images = []
    labels = []
    h_format = 244  # Altura 
    w_format = 244  # Anchura 
    for image_path, label in tqdm(zip(data['path_files'].tolist(), data['label'].tolist()), desc="Codificando imagenes"):
        try: 
            img = Image.open(image_path).resize((h_format,w_format)).convert('RGB')
            images.append(np.array(img))
            labels.append(np.array(label))
        except Exception as e: 
            print(f"{e} Levantado tras ejecutar la ruta {image_path}")
    
    return np.array(images), np.array(labels)

## FUNCIONES FASE PRE-MODELIZACION ##

# 1. Funcion para contar elementos de un documento espefico
def counter(documento):
    count = 0  
    with open(documento, "r") as output: 
        for _ in output: 
            count +=1 
    return f"{count} items"

# 2. Funcion que depura el dataset escogiendo solo las imagenes validas
def valid(data: str) -> List[str]:
    imagenes_validas = []
    photos = pd.read_json("photos.json", lines=True)
    for img in tqdm(os.listdir(data), desc="Procesando Imagenes"):
        ruta_img = os.path.join(data, img)
        try: 
            Image.open(ruta_img)
            imagenes_validas.append(img)
        except Exception:
            print(f"Error al procesar la imagen {img}")
            pass
    f"{len(imagenes_validas)} imagenes validas de {photos.shape[0]}"
    return imagenes_validas
      