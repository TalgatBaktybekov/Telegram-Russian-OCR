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
print(os.environ)
PROMPT = os.environ['PROMPT']

model = Model(256, len(ALPHABET) + 1)
model.to(DEVICE)
model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'CNNLstm.pt'), weights_only=False, map_location=DEVICE))

image = cv2.imread('/Users/talgatbaktybekov/Desktop/OCRRussian/test2.jpeg')

chunked_rows = ChunkImage(image)

tensored_chunked_rows = []

text = ''

transforms = transforms.Compose(TransformList().transform_list)

for row in chunked_rows:

    tensor_chunks = []

    for chunk in row:
        
        chunk = Image.fromarray(chunk)

        chunk = transforms(chunk)

        tensor_chunks.append(chunk)

    tensor_chunks = torch.stack(tensor_chunks)
    tensored_chunked_rows.append(tensor_chunks)

    predictions = Predict(model, tensor_chunks)

    text += ' '.join(predictions) + '\n'

llm = 

print(llm.invoke(PROMPT+text))



