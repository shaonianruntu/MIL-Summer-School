# coding:utf-8
import random
import torch.nn as nn
import torch.optim as optim
import dataHandler  # 数据处理
from model import PoetryModel
from utils import *
import pickle as p  # 序列化对象并保存到磁盘中，并在需要的时候读取出来

# 数据读取
data = dataHandler.parseRawData()  # All if author=None

for s in data:
    print(s)
# 定义字典
word_to_ix = {}

for sent in data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

word_to_ix['<EOP>'] = len(word_to_ix)
word_to_ix['<START>'] = len(word_to_ix)

VOCAB_SIZE = len(word_to_ix)

print("VOCAB_SIZE:", VOCAB_SIZE)  # VOCAB_SIZE: 8537
print("data_size", len(data))  # data_size 57577

for i in range(len(data)):
    data[i] = toList(data[i])
    data[i].append("<EOP>")

# save the word dic for sample method
# 保存字典
fw = open("word_to_ix", "wb")
p.dump("wordDic", fw)

model = PoetryModel(len(word_to_ix), 256, 256);
optimizer = optim.RMSprop(model.parameters(), lr=0.01, weight_decay=0.0001)
criterion = nn.NLLLoss()

one_hot_var_target = {}
for w in word_to_ix:
    one_hot_var_target.setdefault(w, make_one_hot_vec_target(w, word_to_ix))

epochNum = 1
TRAINSIZE = len(data)-50000
print(TRAINSIZE)
batch = 100

def test():
    v = int(TRAINSIZE / batch)
    loss = 0
    counts = 0
    for case in range(v * batch, min((v + 1) * batch, TRAINSIZE)):
        s = data[case]
        hidden = model.initHidden()
        t, o = makeForOneCase(s, one_hot_var_target)
        output, hidden = model(t, hidden)
        loss += criterion(output, o)
        counts += 1
    loss = loss / counts
    print("=====",loss.item())

print("start training")
for epoch in range(epochNum):
    for batchIndex in range(int(TRAINSIZE / batch)):
        model.zero_grad()
        loss = 0
        counts = 0
        for case in range(batchIndex * batch, min((batchIndex + 1) * batch, TRAINSIZE)):
            s = data[case]
            hidden = model.initHidden()
            t, o = makeForOneCase(s, one_hot_var_target)
            output, hidden = model(t, hidden)
            loss += criterion(output, o)
            counts += 1
        loss = loss / counts
        loss.backward()
        print(epoch, loss.item())
        optimizer.step()
    test()
torch.save(model, 'poetry-gen.pt')