import os
import torch 
from ModelBuilding.DataLoad import LabelCoder
import matplotlib.pyplot as plt

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ALPHABET = os.environ['russianALPHABET']

def Predict(model, img):

    coder = LabelCoder(ALPHABET)

    logits = model(img.to(DEVICE))
    logits = logits.contiguous().cpu()

    T, B, H = logits.size()
    pred_sizes = torch.LongTensor([T for i in range(B)])

    probs, pos = logits.max(2)
    pos = pos.transpose(1, 0).contiguous().view(-1)

    sim_preds = coder.decode(pos.data, pred_sizes.data, raw=False)

    return sim_preds

def Visualize_predictions(model, data, test=False):

    predictions = []
    idx = 0
    if test:
        for batch in data:

            img, true_label = batch['img'], batch['label']

            pred_label = Predict(model, img)
            predictions.append([img, true_label, pred_label])

            idx += 1
            if idx == 8:
                break

        fig = plt.figure(figsize=(10, 10))
        rows = 2
        columns = 2

        for j, exp in enumerate(predictions):

            fig.add_subplot(rows, columns, j + 1)

            plt.imshow(exp[0][0].permute(2, 1, 0).permute(1, 0, 2))
            plt.title('true:' + exp[1][0] + '\npred:' + exp[2][0], loc = 'left')
    else:
        all_data = []
        all_pred_labels = []
        
        for chunked_row in data:
            pred_label = Predict(model, chunked_row) 
            all_data.append(chunked_row)
            all_pred_labels.append(pred_label)

        all_data = torch.cat(all_data, dim=0) 
        all_pred_labels = sum(all_pred_labels, []) 

        num_images = all_data.size(0)
        columns = 2 
        rows = (num_images + 1) // columns  

        fig = plt.figure(figsize=(20, 20))

        for i in range(num_images):
        
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(all_data[i].permute(2, 1, 0).permute(1, 0, 2))  
            plt.title('pred: ' + all_pred_labels[i], loc='left')
            plt.axis('off')  

        plt.tight_layout()  
        plt.show()