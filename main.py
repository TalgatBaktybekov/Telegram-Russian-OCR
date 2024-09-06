import os 
import cv2
from PIL import Image
from ModelBuilding.RnnModel import Model
from predict import Predict, VisualizePredict
from chunking import ChunkImage
from ModelBuilding.TrainEvaluate import TransformList
from torchvision.transforms import transforms
import torch 


ALPHABET = os.environ['russianALPHABET']
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

PROMPT="""I have a text passage that may contain errors typical of handwritten word recognition, such as incorrect character substitutions, omissions, insertions, incorrect word boundaries, misinterpretations of handwriting variations, homophone confusion, contextual errors, poorly handwritten words, and issues with slant and orientation.

Your task is to:

Correct Character Substitutions: Fix any incorrect characters that were substituted for similar-looking ones.
Address Character Omissions: Add any missing characters to complete the words.
Fix Character Insertions: Remove any extra characters that do not belong in the words.
Correct Word Boundaries: Properly split or merge words where necessary to accurately reflect the intended text.
Adjust Handwriting Variations: Interpret and correct words that may have been misread due to handwriting style.
Resolve Homophone Confusion: Replace incorrect words with the proper ones when homophones are confused.
Fix Contextual Errors: Ensure that words fit the context of the sentence or passage.
Improve Readability: Correct any issues caused by poorly handwritten words or irregularities.

Please put the corrected text between @@ symbols like @'corrected text'@ 

input:
"""

model = Model(256, len(ALPHABET) + 1)
model.to(DEVICE)
model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'CNNLstm.pt'), weights_only=False, map_location=DEVICE))

image = cv2.imread('/Users/talgatbaktybekov/Desktop/pet projects/telegram-russian-ocr/test.png')

# Split the image into chunks (hopefully clear, clean splittings of words)
chunked_rows = ChunkImage(image)

# tensors of the rows to pass to VisualisePredict()
tensored_chunked_rows = []

text = ''

transforms = transforms.Compose(TransformList().transform_list)

# Note the difference between input types of Predict and VisualisePredict 
#- Predict takes in one chunked row, while VisualisePredict takes in list of chunked rows when test=False, data loader when test=True
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

print(text)

VisualizePredict(model, tensored_chunked_rows, test=False)

# llm = 'Some llm'

# print(llm.invoke(PROMPT+text).content.split('@')[0])


