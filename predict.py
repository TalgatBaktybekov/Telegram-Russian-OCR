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
        pred_label = Predict(model, data)
        predictions = [data, pred_label]

        fig = plt.figure(figsize=(10, 10))
        rows = int(data.size()[0]/2)
        columns = 2

        for i in range(data.size()[0]):

            fig.add_subplot(rows, columns, i + 1)

            plt.imshow(predictions[0][i].permute(2, 1, 0).permute(1, 0, 2))
            plt.title('pred:' + predictions[1][i], loc = 'left')