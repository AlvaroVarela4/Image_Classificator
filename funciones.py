
from typing import *
from imports import *

## FUNCIONES FASE MODELIZACION ##

# 1. Funcion que verifica que las rutas de las imagenes estan correctamente construidas
def check_path(data:list) -> Tuple[str,str]:
    t = 0 
    f = 0  
    for path in tqdm(data, desc='Procesando rutas'): 
        try:
            Image.open(path)
            t += 1
        except Exception as e:
            f += 1
    print(f"{t} fotos correctas", f"{f} fotos no leidas/invalidas")

# 2. Funcion que crea rutas de las imagenes a partir
def create_path(data: pd.DataFrame) -> list: 
    created_paths = []
    for id in tqdm(data["photo_id"], desc= "Creando rutas"): 
        image_path = os.path.join("photos", f"{id}.jpg")
        try: 
            Image.open(image_path)
            created_paths.append(image_path)
        except Exception as e:
            print(f"Error al crear ruta de imagen {id}")
    check_path(created_paths)
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

# 5. Reduce el numero de muestras por categoria a una misma cantidad n 
def dim_reduction(data: pd.DataFrame, n:int, category = "label") -> pd.DataFrame:
    # Split en la n-esima parte 
    dimension = len(data) // n
    df_reduced, _ = train_test_split(data, train_size=dimension, stratify=data[category], random_state=12)
    # Limpieza en features relevantes 
    df_reduced = df_reduced[["photo_id", "label"]]
    # Comprobacion de que las rutas se han creado correctamente 
    print(f"El tamaño del nuevo DataFrame es: {len(df_reduced)}")
    return df_reduced

# 6. Funcion que homogeiniza las muestras de un dataframe a un numero concreto target_count
def homogenize_dataset(data: pd.DataFrame, target_count:int, image_dir:str ='photos') ->  pd.DataFrame :
    # Instancia de aumentador de imagenes (desenfoque, corte y girado)
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),   # Voltear horizontalmente el 50% de las imágenes
        iaa.Crop(percent=(0, 0.1)),   # Recortar entre 0% a 10% de cada lado de la imagen
        iaa.GaussianBlur(sigma=(0, 3.0)) # Aplicar desenfoque gaussiano con sigma entre 0 y 3.0
       ])   
     # Contar el número de imágenes por categoría
    category_counts = data['label'].value_counts()
    # Lista para almacenar las imágenes aumentadas
    augmented_images = []
    
    # 4. Función para aumentar imágenes: Establece el numero de aumentos por categoria y aplica la instancia de aumento seq([])
    def augment_images(image_paths: list, label: str, num_augmentations_needed: int) -> list:
        augmented = []
        for image_path in image_paths:
            image = imageio.imread(image_path)
            for i in range(num_augmentations_needed // len(image_paths) + 1):
                image_aug = seq(image=image)
                augmented_image_id = f"aug_{os.path.basename(image_path).split('.')[0]}_{i}"
                augmented_image_path = os.path.join(image_dir, f"{augmented_image_id}.jpg")
                imageio.imwrite(augmented_image_path, image_aug)
                augmented.append({'photo_id': augmented_image_id, 'label': label})
                if len(augmented) >= num_augmentations_needed:
                    break
            if len(augmented) >= num_augmentations_needed:
                break
        return augmented
    # Procesar cada categoría
    for category, count in tqdm(category_counts.items(), desc='Processing Categories'):
        category_images = data[data['label'] == category]

        if count > target_count:
            # Seleccionar aleatoriamente target_count imágenes
            category_images = category_images.sample(n=target_count, random_state=42)
        elif count < target_count:
            # Aumentar imágenes para alcanzar target_count
            image_paths = [os.path.join(image_dir, f"{row['photo_id']}.jpg") for index, row in category_images.iterrows()]
            num_augmentations_needed = target_count - count
            augmented = augment_images(image_paths, category, num_augmentations_needed)
            augmented_images.extend(augmented)
    # Crear un nuevo dataframe con las imágenes aumentadas
    df_augmented = pd.DataFrame(augmented_images)
    # Combinar el dataframe original (filtrado si se redujo) con el dataframe de imágenes aumentadas
    df_balanced = pd.concat([data[data['label'] != 'food'], df_augmented], ignore_index=True)
    # Seleccionar aleatoriamente target_count imágenes de la categoría 'food' si es necesario
    food_images = data[data['label'] == 'food']
    if len(food_images) > target_count:
        food_images = food_images.sample(n=target_count, random_state=42)
    df_balanced = pd.concat([df_balanced, food_images], ignore_index=True)

    return df_balanced

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
      