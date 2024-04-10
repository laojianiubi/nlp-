import pickle
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import os
from tqdm import tqdm


def read_data(file):
    with open(file, encoding="utf-8") as f:
        lines = f.read().split("\n")
        words = []
        labels = []
        for line in lines:
            data = line.strip().split("\t")
            if len(data) != 2:
                continue
            word, label = data
            words.append(word)
            labels.append(label)
    return words, labels


class MyDataSet(Dataset):
    def __init__(self, all_data, all_label):
        super().__init__()

        self.all_data = all_data
        self.all_label = all_label

    def __getitem__(self, idx):
        text = self.all_label[idx][:max_len]
        label = self.all_label[idx]
        label = label_2_index[label]
        text_idx = [word_2_index[i] for i in text]
        text_idx = text_idx + [0] * (max_len - len(text_idx))
        return torch.tensor(text_idx), torch.tensor(label)

    def __len__(self):
        assert len(self.all_data) == len(self.all_label)
        return len(self.all_data)


def word_2_index(all_data):
    word_2_index = {"PAD": 0, "UNK": 1}
    for data in all_data:
        for word in data:
            word_2_index[word] = word_2_index.get(word, len(word_2_index))
    return word_2_index, list(word_2_index)


def label_2_index(all_label):
    label_2_index = {}
    for label in all_label:
        label_2_index[label] = label_2_index.get(label, len(label_2_index))

    return label_2_index, list(label_2_index)
class MyRnn(nn.Module):
    def __init__(self,input_size,hidden_size,batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.W = nn.Linear(self.input_size,self.hidden_size)
        self.U = nn.Linear(self.hidden_size,self.hidden_size)
        self.tanh = nn.Tanh()

    def forward(self,x):   #x:batch_size * sq_len * embed

        batch_,len_,embed_ = x.shape
        t = torch.zeros(batch_, self.hidden_size)
        out_put1 = torch.zeros(batch_,len_,self.hidden_size)
        out_put2 = torch.zeros(batch_,1,self.hidden_size)
        for i in range(len_):
            h1 = self.W.forward(x[:,i])     #h1:batch_size * hidden_size
            h2 = h1 + t                    #h2:batch_size * hidden_size
            h3 = self.tanh(h2)              #h3:batch_size * hidden_size
            t = self.U.forward(h3)
            out_put1[:,i] = h3
            if i == len_-1:
                out_put2[:,0] = h3
        return out_put1,out_put2



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(crop_num, embeddeing_num)
        self.rnn = MyRnn(input_size=embeddeing_num, hidden_size=hidden_size)
        self.rule = nn.ReLU()
        self.dupout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, class_num)
        self.loss_funtiong = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        x = self.embedding(x)
        x, y = self.rnn(x)
        x = self.rule(x)
        x = self.dupout(x)
        pre = self.linear(x)
        pre = pre[:, 0]
        if label is not None:
            return self.loss_funtiong(pre, label)
        else:
            return torch.argmax(pre, dim=-1)


if __name__ == '__main__':
    all_data, all_label = read_data(os.path.join("data", "train.txt"))
    dev_all_data, dev_all_label = read_data(os.path.join("data", "test.txt"))
    word_2_index, index_2_word = word_2_index(all_data)
    label_2_index, _ = label_2_index(all_label)

    max_len = 30
    lr = 0.001
    crop_num = len(word_2_index)
    embeddeing_num = 200
    hidden_size = 100
    dropout_rate = 0.2
    class_num = len(label_2_index)
    batch_size = 10
    epoch = 100
    hidden_size = 300

    train_dataset = MyDataSet(all_data, all_label)
    train_daloder = DataLoader(train_dataset, batch_size=batch_size)
    dev_dataset = MyDataSet(dev_all_data, dev_all_label)
    dev_daloder = DataLoader(dev_dataset, batch_size=batch_size)

    model = Model()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        model.train()
        for batch_idx, batch_labels in tqdm(train_daloder):
            loss = model(batch_idx, batch_labels)
            loss.backward()
            optim.step()
            optim.zero_grad()
            # break
        print(loss)
        right_num = 0
        model.eval()
        for batch_idx, batch_labels in tqdm(dev_daloder):
            pre = model(batch_idx)
            right_num += int(torch.sum(pre == batch_labels))

        print(f"acc:{right_num / len(dev_dataset)}")
        break

    with open("data/rnn_classiton.pkl", "wb") as f:
        pickle.dump(model, f)
    print("")


