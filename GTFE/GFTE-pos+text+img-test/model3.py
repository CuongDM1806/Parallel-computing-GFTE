import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
import time
import csv
import os

class TbNet(torch.nn.Module):
    def __init__(self, num_node_features, vocab_size, num_text_features, num_classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(TbNet, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(num_text_features, 64)
        self.conv4 = GCNConv(64, 64)

        self.embeds = nn.Embedding(vocab_size, num_text_features)
        self.rnn = nn.GRU(num_text_features, 64, bidirectional=False, batch_first=True)

        self.lin1 = torch.nn.Linear(64*2, 64)
        self.lin_img = torch.nn.Linear(64*2, 64)
        self.lin_text = torch.nn.Linear(64*2, 64)
        self.lin_final = torch.nn.Linear(64*3, num_classes)

        ks = [3, 3, 3]
        ps = [1, 1, 1]
        ss = [1, 1, 1]
        nm = [64, 64, 64]
        cnn = nn.Sequential()
        def convRelu(i, batchNormalization=False):
            nIn = 1 if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module(f'conv{i}', nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(nOut))
            cnn.add_module(f'relu{i}', nn.ReLU(True))
        convRelu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(2, 2))
        convRelu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(2, 2))
        convRelu(2, True)
        cnn.add_module('pooling2', nn.MaxPool2d(2, 2))
        self.cnn = cnn

        self.time_log = {}
        self.csv_path = "forward_times.csv"

        # Tạo CSV nếu chưa có và ghi tiêu đề
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['total', 'text', 'img', 'nodenum', 'concat'])

    def sample_box_feature(self, cnnout, nodenum, pos):
        cnt = 0
        for i in range(nodenum.size()[0]):
            imgpos = pos[cnt:cnt + nodenum[i], :]
            imgpos = imgpos.unsqueeze(0).unsqueeze(0)
            cnnin = cnnout[i].unsqueeze(0)
            sout = F.grid_sample(cnnin, imgpos, mode='bilinear', padding_mode='border')
            cnt += nodenum[i]
            sout = sout.squeeze(0).squeeze(1).permute(1, 0)
            out = sout if i == 0 else torch.cat((out, sout), 0)
        return out

    def forward(self, data):
        # Đưa về GPU
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        xtext = data.xtext.to(self.device)
        img = data.img.to(self.device)
        nodenum = data.nodenum.to(self.device)
        imgpos = data.imgpos.to(self.device)


        # Reset đồng hồ
        self.time_log = {'text': 0, 'img': 0, 'nodenum': 0, 'concat': 0}
        start_total = time.time()

        # Node features
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x1 = x[edge_index[0]]
        x2 = x[edge_index[1]]
        xpair = torch.cat((x1, x2), dim=1)
        xpair = F.relu(self.lin1(xpair))

        # TEXT
        start = time.time()
        xtext = self.embeds(xtext)
        textout, _ = self.rnn(xtext)
        textout = textout[:, -1, :]
        textout = self.conv3(textout, edge_index)
        textout = F.relu(textout)
        textout = self.conv4(textout, edge_index)
        textout = F.relu(textout)
        x1text = textout[edge_index[0]]
        x2text = textout[edge_index[1]]
        xpairtext = torch.cat((x1text, x2text), dim=1)
        xpairtext = F.relu(self.lin_text(xpairtext))
        self.time_log['text'] = time.time() - start

        # IMG
        start = time.time()
        imgconv = self.cnn(img)
        self.time_log['img'] = time.time() - start

        # nodenum + imgpos
        start = time.time()
        ximg = self.sample_box_feature(imgconv, nodenum, imgpos)
        x1img = ximg[edge_index[0]]
        x2img = ximg[edge_index[1]]
        ximgpair = torch.cat((x1img, x2img), dim=1)
        ximgpair = F.relu(self.lin_img(ximgpair))
        self.time_log['nodenum'] = time.time() - start

        # Concat + final layer
        start = time.time()
        xfin = torch.cat((xpair, xpairtext, ximgpair), dim=1)
        xfin = self.lin_final(xfin)
        output = F.log_softmax(xfin, dim=1)
        self.time_log['concat'] = time.time() - start

        total_time = time.time() - start_total
        # print("⏱️ Total forward time:", round(total_time, 4), "s")
        # for k, v in self.time_log.items():
        #     print(f"  ├─ {k:8s}: {round(v, 4)} s")

        # Ghi log vào CSV
        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                round(total_time, 4),
                round(self.time_log['text'], 4),
                round(self.time_log['img'], 4),
                round(self.time_log['nodenum'], 4),
                round(self.time_log['concat'], 4)
            ])

        return output
