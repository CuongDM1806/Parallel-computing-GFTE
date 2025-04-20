from __future__ import print_function
import os
import csv
import time
import math
import json
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torch_geometric.data import DataLoader
from model3 import TbNet
from dataset3 import ScitsrDataset
import utils

def parse_args():
    parser = argparse.ArgumentParser(description="Train TbNet on SciTSR dataset")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batchSize', type=int, default=32)
    parser.add_argument('--imgH', type=int, default=32)
    parser.add_argument('--imgW', type=int, default=32)
    parser.add_argument('--nh', type=int, default=256)
    parser.add_argument('--niter', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--ngpu', type=int, default=0)
    parser.add_argument('--crnn', default='', help="path to pretrained model")
    parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz\'')
    parser.add_argument('--experiment', default='expr')
    parser.add_argument('--displayInterval', type=int, default=20)
    parser.add_argument('--valInterval', type=int, default=1)
    parser.add_argument('--saveInterval', type=int, default=10)
    parser.add_argument('--adam', action='store_true')
    parser.add_argument('--adadelta', action='store_true')
    parser.add_argument('--keep_ratio', action='store_true')
    parser.add_argument('--random_sample', action='store_true')
    return parser.parse_args()

def weights_init(m):
    """Initialize weights for Conv and BatchNorm layers."""
    classname = m.__class__.__name__
    if 'Conv' in classname:
        if hasattr(m, 'weight'):
            m.weight.data.normal_(0.0, 0.02)
    elif 'BatchNorm' in classname:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def validate(model, dataset, criterion, device, max_iter=100):
    """Validation function for the model."""
    print('Start validation...')
    model.eval()
    data_loader = DataLoader(dataset, batch_size=4)
    correct = total = 0
    loss_avg = utils.averager()
    max_iter = min(max_iter, len(data_loader))

    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            if idx >= max_iter:
                break
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            loss_avg.add(loss)

            preds = output.argmax(dim=1).cpu().numpy()
            labels = data.y.cpu().numpy()
            correct += (preds == labels).sum()
            total += labels.shape[0]

    accuracy = correct / float(total)
    print(f'Validation Loss: {loss_avg.val():.4f}, Accuracy: {accuracy:.4f}')

import os
import csv
import time

def train_batch(train_iter, model, criterion, optimizer, device, epoch, log_path="train_log.csv"):
    """Train a single batch."""
    data = next(train_iter).to(device)

    start_time = time.time()
    output = model(data)
    loss = criterion(output, data.y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    end_time = time.time()

    # Logging
    log_dir = os.path.dirname(log_path)
    if log_dir:  # Chỉ tạo thư mục nếu log_path có chỉ định thư mục
        os.makedirs(log_dir, exist_ok=True)

    with open(log_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["duration_seconds", "epoch"])
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow({
            "duration_seconds": round(end_time - start_time, 4),
            "epoch": epoch
        })

    return loss


def main():
    opt = parse_args()

    os.makedirs(opt.experiment, exist_ok=True)

    # Reproducibility
    manual_seed = random.randint(1, 10000)
    print("Random Seed:", manual_seed)
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dataset = ScitsrDataset(r"D:\TTSS\New folder (2)\data\SciTSR\train")
    subset_size = len(train_dataset) // 20
    train_dataset, _ = torch.utils.data.random_split(train_dataset, [subset_size, len(train_dataset) - subset_size])
    test_dataset = ScitsrDataset(r"D:\TTSS\New folder (2)\data\SciTSR\test")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)

    # Model setup
    nclass = 2
    input_num = 8
    vocab_size = 39
    num_text_features = 64
    model = TbNet(input_num, vocab_size, num_text_features, nclass).to(device)
    model.apply(weights_init)

    # Criterion and Optimizer
    criterion = torch.nn.NLLLoss().to(device)
    if opt.adam:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    elif opt.adadelta:
        optimizer = optim.Adadelta(model.parameters(), lr=opt.lr)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=opt.lr)

    # Load pretrained model
    if opt.crnn:
        print(f'Loading pretrained model from {opt.crnn}')
        model.load_state_dict(torch.load(opt.crnn), strict=False)

    # Training loop
    for epoch in range(opt.niter):
        model.train()
        train_iter = iter(train_loader)
        loss_avg = utils.averager()
        print(f"Epoch {epoch} | Training samples: {len(train_loader)}")

        for i in range(len(train_loader)):
            loss = train_batch(train_iter, model, criterion, optimizer, device, epoch)
            loss_avg.add(loss)

            if (i + 1) % opt.displayInterval == 0:
                print(f"[{epoch}/{opt.niter}] Batch {i+1}/{len(train_loader)} | Loss: {loss_avg.val():.4f}")
                loss_avg.reset()

        # Save checkpoint
        if (epoch + 1) % opt.saveInterval == 0:
            ckpt_path = os.path.join(opt.experiment, f'net_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        # # Validation
        # if (epoch + 1) % opt.valInterval == 0:
        #     validate(model, test_dataset, criterion, device)

if __name__ == '__main__':
    main()
