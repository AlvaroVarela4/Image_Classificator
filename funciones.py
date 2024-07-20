# Funcion check_path: verifica que todas las imagenes se abren correctamente mediante la columna "path_files"
from typing import *
from tqdm import tqdm 

## FUNCIONES FASE MODELIZACION ##
def check_path(data) -> Tuple:
    t = 0 
    f = 0  
    for path in tqdm(data['path_files'], desc='Procesando rutas'): 
        try:
            Image.open(path)
            t += 1
        except Exception as e:
            f += 1
            
    return f"{t} fotos correctas", f"{f} fotos no leidas/invalidas"

def create_path(data) -> List: 
    created_paths = []
    for id in tqdm(data["photo_id"], desc= "Creando rutas"): 
        image_path = os.path.join("photos", f"{id}.jpg")
        try: 
            Image.open(image_path)
            created_paths.append(image_path)
        except Exception as e:
            print(f"Error al crear ruta de imagen {id}")

    return created_paths
## FUNCIONES FASE PRE-MODELIZACION ##

# 1. Funcion para contar elementos de un documento espefico
def counter(documento):
    count = 0  
    with open(documento, "r") as output: 
        for _ in output: 
            count +=1 
    return f"{count} items"

# 2. Funcion que depura el dataset escogiendo solo las imagenes validas
def valid(data):
    imagenes_validas = []
    photos = pd.read_json("photos.json", lines=True)
    for img in os.listdir(data):
        ruta_img = os.path.join(data, img)
        try: 
            Image.open(ruta_img)
            imagenes_validas.append(img)
        except Exception:
            print(f"Error al procesar la imagen {img}")
            pass
    f"{len(imagenes_validas)} imagenes validas de {photos.shape[0]}"
    return imagenes_validas
      