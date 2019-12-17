import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from dense_mask_generator import DenseMaskGenerator
from dataset import *
from result_save_visualization import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

data_dict = {'epoch': [], 'time': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

# パラメータ利用, 全結合パラメータの凍結
new_net = parameter_use('./result/CIFAR10_original_train_epoch150.pkl')
for param in new_net.features.parameters():
    param.requires_grad = False

optimizer = optim.SGD(new_net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# 全結合層のリスト
dense_list = [new_net.classifier[i] for i in range(len(new_net.classifier)) if
              isinstance(new_net.classifier[i], nn.Linear)]
dense_count = len(dense_list)

# マスクのオブジェクト
de_mask = [DenseMaskGenerator() for _ in dense_list]
inv_prune_ratio = 50

# weight_pruning
for count in range(1, inv_prune_ratio):
    print(f'\nweight pruning: {count}')
    # 全結合層を可視化
    # if count == 1 or count == 10 or count == 18:
    #     param_list = [torch.t(new_net.fc1.weight.data).cpu().numpy(), torch.t(new_net.fc2.weight.data).cpu().numpy()]
    #     dense_vis(param_list)

    # 重みを１次元ベクトル化
    weight_vector = [np.reshape(torch.abs(dense.weight.data.clone()).cpu().detach().numpy(),
                                (1, dense_list[i].in_features * dense_list[i].out_features)).squeeze()
                     for i, dense in enumerate(dense_list)]

    # 昇順にソート
    for i in range(len(weight_vector)):
        weight_vector[i].sort()

    # 刈る基準の閾値を格納
    threshold = [weight_vector[i][int(dense_list[i].in_features * dense_list[i].out_features / inv_prune_ratio * count) - 1]
                 for i in range(dense_count)]

    # 枝刈り本体
    with torch.no_grad():
        for i, dense in enumerate(dense_list):
            save_mask = de_mask[i].generate_mask(dense.weight.data.clone(), threshold[i])
            dense.weight.data *= torch.tensor(save_mask, device=device, dtype=dtype)

    # パラメータの割合
    weight_ratio = [np.count_nonzero(dense.weight.cpu().detach().numpy()) / np.size(dense.weight.cpu().detach().numpy())
                    for dense in dense_list]

    # 枝刈り後のニューロン数
    neuron_num_new = [dense_list[i].in_features - de_mask[i].neuron_number(torch.t(dense.weight)) for i, dense in enumerate(dense_list)]

    for i in range(len(dense_list)):
        print(f'dense{i+1}_param: {weight_ratio[i]:.4f}', end=", " if i != dense_count - 1 else "\n"if i != dense_count - 1 else "\n")
    for i in range(len(dense_list)):
        print(f'neuron_number{i+1}: {neuron_num_new[i]}', end=", " if i != dense_count - 1 else "\n"if i != dense_count - 1 else "\n")

    f_num_epochs = 10
    # finetune
    start = time.time()
    for epoch in range(f_num_epochs):
        # train
        new_net.train()
        train_loss, train_acc = 0, 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = new_net(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            train_acc += (outputs.max(1)[1] == labels).sum().item()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for j, dense in enumerate(dense_list):
                    dense.weight.data *= torch.tensor(de_mask[j].mask, device=device, dtype=dtype)
        avg_train_loss, avg_train_acc = train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset)

        # val
        new_net.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = new_net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_val_loss, avg_val_acc = val_loss / len(test_loader.dataset), val_acc / len(test_loader.dataset)

        process_time = time.time() - start

        print(f'epoch [{epoch + 1}/{f_num_epochs}], train_loss: {avg_train_loss:.4f}, train_acc: {avg_train_acc:.4f}, '
              f'val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')

        # 結果の保存
        input_data = [epoch + 1, process_time, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc]
        result_save('./result/de_prune_result_cifar10.csv', data_dict, input_data)

# パラメータの保存
parameter_save('./result/CIFAR10_dense_prune.pkl', new_net)
