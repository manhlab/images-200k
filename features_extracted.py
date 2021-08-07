import os
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
# tf.enable_eager_execution()
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import datetime
from tqdm import tqdm
from matplotlib import pyplot as plt
import augly.image as imaugs
import augly.utils as utils
import pickle
PART = 7 # 10 parts
DEBUG = False
temp_dir = "/content/data/data_origin"
REQUIRED_OUTPUT = 'logits_sup'
hub_path = 'gs://simclr-checkpoints/simclrv2/finetuned_100pct/r50_1x_sk0/hub/'
model = hub.Module(hub_path, trainable=False)
aug1_compose = imaugs.Compose(
    [
        imaugs.PerspectiveTransform(sigma=20),
        imaugs.OverlayEmoji()
    ]
)
# aug1_compose = imaugs.Compose(
#     [ 
#         imaugs.MemeFormat(caption_height=75, meme_bg_color=(0, 0, 0), text_color=(255, 255, 255),p=0.2),
#         imaugs.PerspectiveTransform(sigma=20),
#         imaugs.OverlayEmoji()
#     ]
# )
def aug1_function(input_img):
    return aug1_compose(input_img)
def get_embeddings(image_root_dir: str):
    def get_id(image_path: Path):
        return str(image_path.name).split(".")[0]
    
    def get_embedding_single(image_path: Path) -> np.ndarray:

        image_input = Image.open(str(image_path))
        image_aug1 = aug1_function(image_input)
        #image_aug2 = aug2_function(image_input)

        def get_emb(img):
            image_data = np.array(img.convert('RGB'))
            image_tensor = tf.convert_to_tensor(image_data)
            image_tensor = tf.image.convert_image_dtype(image_tensor, dtype=tf.float32)
            image_tensor = tf.expand_dims(image_tensor, axis=0)
            output = model(image_tensor, signature="default", as_dict=True)[REQUIRED_OUTPUT][:,:]
            output = tf.reshape(output, [-1])
            return output
        image_id = get_id(image_path)

        return [image_id, get_emb(image_input), get_emb(image_aug1)]

    image_paths = [p for p in Path(image_root_dir).rglob('*.jpg')]
    print(len(image_paths))
    
    embeddings = [get_embedding_single(image_path) 
                  for i, image_path in tqdm(enumerate(image_paths))]

    return embeddings

if True:
    print("{date:%Y%m%d-%H%M%S}".format(date=datetime.datetime.now()))
    embeddings = get_embeddings(temp_dir)
    print(len(embeddings))
    data_features = {"embeddings": embeddings}
    file_to_store = open(f"data_features_{PART}.pickle", "wb")
    pickle.dump(data_features, file_to_store)
    file_to_store.close()
    print("{date:%Y%m%d-%H%M%S}".format(date=datetime.datetime.now()))
