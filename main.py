import os 
import cv2
from PIL import Image
from ModelBuilding.RnnModel import Model
from predict import Predict, Visualize_predictions
from chunking import ChunkImage
from ModelBuilding.TrainEvaluate import TransformList
from torchvision.transforms import transforms
import torch 

ALPHABET = os.environ['russianALPHABET']
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = Model(256, len(ALPHABET) + 1)
model.to(DEVICE)
model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'CNNLstm.pt'), weights_only=False, map_location=DEVICE))

image = cv2.imread('/Users/talgatbaktybekov/Desktop/pet projects/telegram-russian-ocr/test2.jpeg')

chunks = ChunkImage(image)
tensor_chunks = []
transforms = transforms.Compose(TransformList().transform_list)

for chunk in chunks:
    
    chunk = Image.fromarray(chunk)

    chunk = transforms(chunk)

    tensor_chunks.append(chunk)

tensor_chunks = torch.stack(tensor_chunks)

Visualize_predictions(model, tensor_chunks)

# output = ''

# for prediction in predictions:
#     output += prediction 

# print(output)



