import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class LabelCoder(object):

    def __init__(self, alphabet):

        self.alphabet = alphabet
        self.char2idx = {}

        for i, char in enumerate(alphabet):
            self.char2idx[char] = i + 1
        self.char2idx[''] = 0

    def encode(self, text: str):

        length = []
        result = []

        for item in text:

            item = str(item)
            length.append(len(item))

            for char in item:
                if char in self.char2idx:
                    index = self.char2idx[char]
                else:
                    index = 0

                result.append(index)

        text = result

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):

        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

        
class OCRdataset(Dataset):

    def __init__(self, path_to_imgdir: str, path_to_labels: str, transform_list = None):

        super(OCRdataset, self).__init__()

        self.imgdir = path_to_imgdir
        df = pd.read_csv(path_to_labels, sep = '\t', names = ['image_name', 'label'])
        df = df.dropna()

        self.image2label = [(self.imgdir + image, label) for image, label in zip(df['image_name'], df['label'])]

        self.transform = transforms.Compose(transform_list)
        self.collate_fn = Collator()

    def __len__(self):

        return len(self.image2label)

    def __getitem__(self, index):

        assert index <= len(self), 'index range error'
        image_path, label = self.image2label[index]
        img = Image.open(image_path)

        if self.transform is not None:
            img = self.transform(img)

        item = {'idx' : index, 'img': img, 'label': label}
        return item


class Collator(object):
    
    def __call__(self, batch):

        width = [item['img'].shape[2] for item in batch]
        indexes = [item['idx'] for item in batch]
        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], 
                           max(width)], dtype=torch.float32)
        
        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print(imgs.shape)

        item = {'img': imgs, 'idx':indexes}
        if 'label' in batch[0].keys():
            labels = [item['label'] for item in batch]
            item['label'] = labels
            
        return item
    
