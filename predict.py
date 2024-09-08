import os
import torch 
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision.transforms import transforms
from ModelBuilding.DataLoad import LabelCoder
from ModelBuilding.RnnModel import Model
from chunking import ChunkImage
from ModelBuilding.TrainEvaluate import TransformList


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ALPHABET = os.environ['russianALPHABET']
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


def Predict(model, img):

    coder = LabelCoder(ALPHABET)

    logits = model(img.to(DEVICE))
    logits = logits

    T, B, H = logits.size()
    pred_sizes = torch.LongTensor([T for i in range(B)])

    probs, pos = logits.max(2)
    pos = pos.transpose(1, 0).contiguous().view(-1)

    sim_preds = coder.decode(pos.data, pred_sizes.data, raw=False)

    return sim_preds

def VisualizePredict(model, data):

    # data: list of chunked rows (type: Tensors)

    all_data = []
    all_pred_labels = []
    
    for chunked_row in data:
        pred_label = Predict(model, chunked_row)
        if type(pred_label)==list:
            pred_label[-1] += '\n'
        elif type(pred_label)==str:
            pred_label = [pred_label + '\n']

        all_data.append(chunked_row)
        all_pred_labels.append(pred_label)

    all_data = torch.cat(all_data, dim=0) 
    all_pred_labels = sum(all_pred_labels, []) 

    num_images = all_data.size(0)
    columns = 2 
    rows = num_images//columns +1

    fig = plt.figure(figsize=(7, rows))
    
    for i in range(num_images):
    
        ax = fig.add_subplot(rows, columns, i + 1)
        ax.imshow(all_data[i].permute(1, 2, 0)) 
        ax.set_xlabel(all_pred_labels[i], fontsize=7)  
        ax.yaxis.set_visible(False) 
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False) 

    plt.subplots_adjust(wspace=0.7, hspace=0.3)
    plt.show()

    return ' '.join(all_pred_labels).replace('\n ', '\n')

def TranscribeImage(image, visualise=False):
    
    model = Model(256, len(ALPHABET) + 1)
    model.to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'CNNLstm.pt'), weights_only=False, map_location=DEVICE))

    # Split the image into chunks (hopefully clear, clean splittings of words)
    chunked_rows = ChunkImage(image)

    # tensors of the rows to pass to VisualisePredict()
    tensored_chunked_rows = []

    text = ''

    transformator = transforms.Compose(TransformList().transform_list)

    # Note the difference between input types of Predict and VisualisePredict 
    #- Predict takes in one chunked row, while VisualisePredict takes in list of chunked rows when test=False, data loader when test=True
    for row in chunked_rows:

        tensor_chunks = []
        for chunk in row:
            
            chunk = Image.fromarray(chunk)
            chunk = transformator(chunk)
            tensor_chunks.append(chunk)

        tensor_chunks = torch.stack(tensor_chunks)
        tensored_chunked_rows.append(tensor_chunks)

        if not visualise:
            predictions = Predict(model, tensor_chunks)
            text += ' '.join(predictions) + '\n'

    if visualise:
       text = VisualizePredict(model, tensored_chunked_rows)
        
    # llm = 'Some llm'

    # text = llm.invoke(PROMPT+text).content.split('@')[0]
    return text, chunked_rows

