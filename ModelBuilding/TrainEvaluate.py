import time
import math
import torch
import os
from torch.nn.utils.clip_grad import clip_grad_norm_
from textdistance import levenshtein as lev
from torchvision.transforms import transforms
from ModelBuilding.DataLoad import OCRdataset, Collator, LabelCoder
from ModelBuilding.RnnModel import Model

ALPHABET = os.environ['russianALPHABET']
PATH_TO_TRAIN_IMGDIR = os.environ["PATH_TO_TRAIN_IMGDIR"]
PATH_TO_TRAIN_LABELS = os.environ["PATH_TO_TRAIN_LABELS"]
PATH_TO_TEST_IMGDIR = os.environ["PATH_TO_TEST_IMGDIR"]
PATH_TO_TEST_LABELS = os.environ["PATH_TO_TRAIN_LABELS"]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 8

class CustomCTCLoss(torch.nn.Module):

    # T x B x H => Softmax on dimension 2
    def __init__(self, dim=2):

        super().__init__()

        self.dim = dim
        self.ctc_loss = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)

    def forward(self, logits, labels, prediction_sizes, target_sizes):

        loss = self.ctc_loss(logits, labels, prediction_sizes, target_sizes)
#       self.debug(loss, logits, labels, prediction_sizes, target_sizes)
        loss = self.sanitize(loss)

        return self.debug(loss, logits, labels, prediction_sizes, target_sizes)
    
    def sanitize(self, loss):

        if abs(loss.item()) > 99999:
            return torch.zeros_like(loss, requires_grad = True)
        
        if math.isnan(loss.item()):
            return torch.zeros_like(loss, requires_grad = True)
        
        return loss

    def debug(self, loss, logits, labels, prediction_sizes, target_sizes):

        if math.isnan(loss.item()):

            print("Loss:", loss)
            print("logits:", logits)
            print("logits nan:", torch.sum(torch.isnan(logits)))
            print("labels:", labels)
            print("prediction_sizes:", prediction_sizes)
            print("target_sizes:", target_sizes)
            raise Exception("NaN loss obtained.")
        
        return loss

    
def print_epoch_data(epoch, mean_loss, char_error, word_error, time_elapsed, zero_out_losses):

    if epoch == 0:
        print('epoch | mean loss | mean cer | mean wer | time elapsed | warnings')

    epoch_str = str(epoch)

    zero_out_losses_str = str(zero_out_losses)

    if len(epoch_str) < 2:
        epoch_str = '0' + epoch_str
    if len(zero_out_losses_str) < 2:
        zero_out_losses_str = '0' + zero_out_losses_str

    report_line = epoch_str + ' '*7 + "%.3f" % mean_loss + ' '*7 + "%.3f" % char_error + ' '*7 + \
             "%.3f" % word_error + ' '*7 +  "%.1f" % float(time_elapsed)
    
    if zero_out_losses != 0:
        report_line += f'       {zero_out_losses} batch losses skipped due to nan value'

    print(report_line)
    
    
def fit(model, optimizer, loss_fn, loader, epochs = 64):

    report = []
    coder = LabelCoder(ALPHABET)

    for epoch in range(epochs):

        zero_out_losses = 0
        start_time = time.time()
        model.train()
        outputs = []
        
        for batch in loader:

            optimizer.zero_grad()

            input_, targets = batch['img'], batch['label']

            targets, lengths = coder.encode(targets)
            logits = model(input_.to(DEVICE))
            logits = logits.contiguous().cpu()

            T, B, H = logits.size()
            pred_sizes = torch.LongTensor([T for i in range(B)])
            targets = targets.view(-1).contiguous()
            loss = loss_fn(logits, targets, pred_sizes, lengths)

            if (torch.zeros(loss.size()) == loss).all():
                zero_out_losses += 1
                continue

            probs, preds = logits.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = coder.decode(preds.data, pred_sizes.data, raw=False)

            try:

                char_error = sum([lev(batch['label'][i], sim_preds[i])/max(len(batch['label'][i]), len(sim_preds[i])) for i in range(len(batch['label']))])/len(batch['label'])
                word_error = 1 - sum([batch['label'][i] == sim_preds[i] for i in range(len(batch['label']))])/len(batch['label'])

            except:

                types1, types2 = dict(), dict()

                for i in range(len(batch['label'])):

                    if isinstance(batch['label'][i], float):

                        print(batch['label'][i], batch['img'][i])
                        if i >= 1:
                            print(batch['label'][i-1], batch['img'][i-1])
                        else:
                            print(batch['label'][i+1], batch['img'][i+1])

                    if type(batch['label'][i]) in types1.keys():
                        types1[type(batch['label'][i])] += 1
                    else:
                        types1[type(batch['label'][i])] = 1

                    if type(batch['label']) in types1.keys():
                        types2[type(sim_preds[i])] += 1
                    else:
                        types2[type(sim_preds[i])] = 1

                print(f"""types1 {types1}, 
                types2 {types2}""")

                return
            
            loss.backward()

            clip_grad_norm_(model.parameters(), 0.05)
            optimizer.step()

            output = {'loss': abs(loss.item()),'cer': char_error,'wer': word_error}
            outputs.append(output)

        if len(outputs) == 0:
            print('ERROR: bad loss, try to decrease learning rate and batch size')
            return None
        
        end_time = time.time()

        mean_loss = sum([outputs[i]['loss'] for i in range(len(outputs))])/len(outputs)
        char_error = sum([outputs[i]['cer'] for i in range(len(outputs))])/len(outputs)
        word_error = sum([outputs[i]['wer'] for i in range(len(outputs))])/len(outputs)

        report.append({'mean_loss' : mean_loss, 'mean_cer' : char_error, 'mean_wer' : word_error})

        print(f"""The epoch started at : {time.localtime(start_time).tm_hour-4}:{time.localtime(start_time).tm_min}:{time.localtime(start_time).tm_sec},
              ended at: {time.localtime(end_time).tm_hour-4}:{time.localtime(end_time).tm_min}:{time.localtime(end_time).tm_sec}""")
        
        print_epoch_data(epoch, mean_loss, char_error, word_error, end_time - start_time, zero_out_losses)

        if epoch%3 == 0:
            torch.save(model.state_dict(), str(epoch) + 'epoch.pt')

    torch.save(model.state_dict(), str(epoch) + 'epoch.pt')

    return report 

def evaluate(model, loader):

    coder = LabelCoder(ALPHABET)
    labels, predictions = [], []

    for batch in loader:

        input_, targets = batch['img'].to(DEVICE), batch['label']

        labels.extend(targets)
        targets, _ = coder.encode(targets)

        logits = model(input_)
        logits = logits.contiguous().cpu()

        T, B, H = logits.size()
        pred_sizes = torch.LongTensor([T for i in range(B)])
        probs, pos = logits.max(2)
        pos = pos.transpose(1, 0).contiguous().view(-1)

        sim_preds = coder.decode(pos.data, pred_sizes.data, raw=False)
        predictions.extend(sim_preds)

    char_error = sum([lev(labels[i], predictions[i])/max(len(labels[i]), len(predictions[i])) for i in range(len(labels))])/len(labels)
    word_error = 1 - sum([labels[i] == predictions[i] for i in range(len(labels))])/len(labels)

    return {'char_error' : char_error, 'word_error' : word_error}

class TransformList():

    def __init__(self, transform_list=None):

        if transform_list is None:
            self.transform_list = [
            transforms.Grayscale(1),
            transforms.Resize((64, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]

if __name__ == 'main' :

    transform_list = TransformList().transform_list

    dataset = OCRdataset(PATH_TO_TRAIN_IMGDIR, PATH_TO_TRAIN_LABELS, transform_list = transform_list)
    collator = Collator()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = 8, collate_fn = collator, shuffle = True)

    model = Model(256, len(ALPHABET)+1)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    loss_fn = CustomCTCLoss()

    report = fit(model=model, optimizer=optimizer, loss_fn=loss_fn, loader=train_loader, epochs=32)

    print("The results of the training are: \n", report)

    test_dataset = OCRdataset(PATH_TO_TEST_IMGDIR, PATH_TO_TEST_LABELS, transform_list=transform_list)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 2, collate_fn = collator)

    print("The results of the testing are: \n", evaluate(model, test_loader))