import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator_test as tg
import torch.nn.functional as F
import math
import argparse
import scipy as sp
import scipy.stats
import os
import time

parser = argparse.ArgumentParser(description="Few shot learning")
parser.add_argument("-hid","--hidden_dim",type = int, default = 256)
parser.add_argument("-r","--relation_dim",type = int, default = 1)
parser.add_argument("-Tw","--train_way",type = int, default = 5)
parser.add_argument("-Ts","--train_shot",type = int, default = 5)
parser.add_argument("-Tq","--train_query",type = int, default = 15)
parser.add_argument("-Vw","--val_way",type = int, default = 5)
parser.add_argument("-Vs","--val_shot",type = int, default = 5)
parser.add_argument("-Vq","--val_query",type = int, default = 15)
parser.add_argument("-s","--size",type = int, default = 100)
parser.add_argument("-e","--episode",type = int, default= 100000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("--train", type = bool, default = True)
parser.add_argument("--exp", type = str, default = 'train-5-5-15-test-5-5-15')
args = parser.parse_args()

# Hyper Parameters
Hidden_dim = args.hidden_dim
Relation_dim = args.relation_dim
Train_way = args.train_way
Train_shot = args.train_shot
Train_query = args.train_query
Val_way = args.val_way
Val_shot = args.val_shot
Val_query = args.val_query
Size = args.size
Episode = args.episode
Test_episode = args.test_episode
Learning_rate = args.learning_rate

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = out.view(out.size(0),-1)
        return out  # 64

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, size, hidden_dim, relation_dim):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(size * size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, int(hidden_dim / 1))
        self.fc4 = nn.Linear(int(hidden_dim / 1), relation_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x),inplace=True)
        out = F.relu(self.fc2(out),inplace=True)
        out = F.relu(self.fc3(out),inplace=True)
        out = self.fc4(out)
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_folders,metatest_folders = tg.mini_imagenet_folders(trainval=False)

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(Size,Hidden_dim,Relation_dim)

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    feature_encoder = feature_encoder.cuda()
    relation_network = relation_network.cuda()

    # feature_encoder.cuda()
    # relation_network.cuda()

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=Learning_rate)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=20000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=Learning_rate)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=20000,gamma=0.5)

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0

    train_counter = 0
    train_total_rewards = 0

    for episode in range(Episode):

        feature_encoder.train()
        relation_network.train()

        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        # init train dataset
        task = tg.MiniImagenetTask(metatrain_folders,Train_way,Train_shot,Train_query)
        support_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=Train_shot,split="train",shuffle=False,train=True)
        query_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=Train_query,split="test",shuffle=True,train=True)

        # sample datas
        samples,sample_labels = support_dataloader.__iter__().next()
        batches,batch_labels = query_dataloader.__iter__().next()

        if Train_shot > 1:
            sample_labels = sample_labels.view(Train_way, Train_shot)
            sample_labels = torch.mean(sample_labels.float(), 1).long()

        for _ in range(1):
            # calculate support features and query features
            support_features = feature_encoder(Variable(samples).cuda())
            query_features = feature_encoder(Variable(batches).cuda())
            _, Channel, Height, Width = support_features.size()

            if Train_shot > 1:
                support_features = support_features.view(Train_way, Train_shot, Channel, Height, Width)
                support_features = torch.mean(support_features, 1).squeeze(1)

            # calculate features
            support_features = support_features.view(support_features.size(0), Channel, Height * Width).transpose(1, 2)
            query_features = query_features.view(query_features.size(0), Channel, Height * Width).transpose(1, 2)

            # calculate mean
            support_mean = support_features.mean(2, keepdim=True)
            query_mean = query_features.mean(2, keepdim=True)

            # centered features
            support_centered = support_features - support_mean
            query_centered = query_features - query_mean

            relation_matrix = (1.0 / (Height * Width - 1)) * torch.matmul(query_centered.unsqueeze(1),support_centered.transpose(1, 2)).view(query_centered.size(0) * Train_way, -1)
            relations = relation_network(relation_matrix).view(query_features.size(0), Train_way)
            cre = nn.CrossEntropyLoss().cuda()
            loss = cre(relations, Variable(batch_labels.cuda()))

            # training
            feature_encoder.zero_grad()
            relation_network.zero_grad()

            loss.backward()

            feature_encoder_optim.step()
            relation_network_optim.step()

        _, train_predict_labels = torch.max(relations.data, 1)

        rewards = [1 if train_predict_labels.cpu()[j] == batch_labels[j] else 0 for j in range(query_features.size(0))]

        train_total_rewards += np.sum(rewards)
        train_counter += query_features.size(0)

        if (episode+1)%100 == 0:
            train_accuracy = train_total_rewards / 1.0 / train_counter
            train_total_rewards = 0
            train_counter = 0
            print("episode:",episode+1,"loss",loss.data[0],'accuracy aver 100 episode', train_accuracy)

        if episode%500 == 0:
            # test
            print("Testing...")
            accuracies = []
            for i in range(Test_episode):
                feature_encoder.eval()
                relation_network.eval()
                total_rewards = 0
                counter = 0
                task = tg.MiniImagenetTask(metatest_folders,Val_way,Val_shot,Val_query)
                support_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=Val_shot,split="train",shuffle=False,train=False)
                test_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=Val_query,split="test",shuffle=True,train=False)

                sample_images,sample_labels = support_dataloader.__iter__().next()
                # calculate support features
                support_features = feature_encoder(Variable(sample_images).cuda())
                _, Channel, Height, Width = support_features.size()
                if Val_shot > 1:
                    support_features = support_features.view(Val_way, Val_shot, Channel, Height, Width)
                    support_features = torch.mean(support_features, 1).squeeze(1)

                # calculate centered support features
                support_features = support_features.view(support_features.size(0), Channel, Height * Width).transpose(1, 2)
                support_mean = support_features.mean(2, keepdim=True)
                support_centered = support_features - support_mean

                for test_images,test_labels in test_dataloader:
                    batch_size = test_labels.shape[0]
                    # calculate centered test features
                    test_features = feature_encoder(Variable(test_images).cuda())

                    test_features = test_features.view(test_features.size(0), Channel, Height * Width).transpose(1, 2)
                    test_mean = test_features.mean(2, keepdim=True)
                    test_centered = test_features - test_mean

                    # calculate relation matrix
                    relation_matrix = (1.0 / (Height * Width - 1)) * torch.matmul(test_centered.unsqueeze(1),support_centered.transpose(1,2)).view(test_centered.size(0) * Val_way, -1)
                    relations = relation_network(relation_matrix).view(batch_size, Val_way)

                    _,predict_labels = torch.max(relations.data,1)

                    rewards = [1 if predict_labels.cpu()[j]==test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)
                    counter += batch_size
                accuracy = total_rewards/1.0/counter
                accuracies.append(accuracy)

            test_accuracy,h = mean_confidence_interval(accuracies)
            print("test accuracy:",test_accuracy,"h:",h)

            if test_accuracy > last_accuracy:
                # create exp directory
                if not os.path.exists(str("./models/" + args.exp)):
                    os.makedirs(str("./models/" + args.exp))

                # save networks
                torch.save(feature_encoder.state_dict(), str("./models/" + args.exp + "/miniimagenet_feature_encoder_" + str(Val_way) +"way_" + str(Val_shot) +"shot.pkl"))
                torch.save(relation_network.state_dict(),str("./models/" + args.exp + "/miniimagenet_relation_network_" + str(Val_way) +"way_" + str(Val_shot) +"shot.pkl"))

                print("save networks for episode:",episode)

                last_accuracy = test_accuracy

def test_one(metatest_folders, feature_encoder, relation_network):

    accuracies = []
    for i in range(Test_episode):
        feature_encoder.eval()
        relation_network.eval()
        total_rewards = 0
        counter = 0
        task = tg.MiniImagenetTask(metatest_folders, Val_way, Val_shot, Val_query)
        support_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=Val_shot, split="train",shuffle=False,train=False)
        test_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=Val_query, split="test", shuffle=True,train=False)

        sample_images, sample_labels = support_dataloader.__iter__().next()
        # calculate support features
        support_features = feature_encoder(Variable(sample_images).cuda())
        _, Channel, Height, Width = support_features.size()
        if Val_shot > 1:
            support_features = support_features.view(Val_way, Val_shot, Channel, Height, Width)
            support_features = torch.mean(support_features, 1).squeeze(1)

        # calculate centered support features
        support_features = support_features.view(support_features.size(0), Channel, Height * Width).transpose(1, 2)
        support_mean = support_features.mean(2, keepdim=True)
        support_centered = support_features - support_mean

        for test_images, test_labels in test_dataloader:
            batch_size = test_labels.shape[0]
            # calculate features
            test_features = feature_encoder(Variable(test_images).cuda())
            test_features = test_features.view(test_features.size(0), Channel, Height * Width).transpose(1, 2)
            test_mean = test_features.mean(2, keepdim=True)
            test_centered = test_features - test_mean

            # calculate relation matrix
            relation_matrix = (1.0 / (Height * Width - 1)) * torch.matmul(test_centered.unsqueeze(1),support_centered.transpose(1, 2)).view(test_centered.size(0) * Val_way, -1)
            relations = relation_network(relation_matrix).view(batch_size, Val_way)

            _, predict_labels = torch.max(relations.data, 1)

            rewards = [1 if predict_labels.cpu()[j] == test_labels[j] else 0 for j in range(batch_size)]

            total_rewards += np.sum(rewards)
            counter += batch_size
        accuracy = total_rewards / 1.0 / counter
        accuracies.append(accuracy)

    test_accuracy,_ = mean_confidence_interval(accuracies)
    return test_accuracy

def test_ten():
    print("Testing...")
    # init dataset
    _, metatest_folders = tg.mini_imagenet_folders()

    # init network
    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(Size, Hidden_dim, Relation_dim)

    feature_encoder = feature_encoder.cuda()
    relation_network = relation_network.cuda()

    if not os.path.exists(str("./models/" + args.exp)):
        feature_encoder.apply(weights_init)
        relation_network.apply(weights_init)
        print("Testing with random initialization ...")

    if os.path.exists(str("./models/" + args.exp + "/miniimagenet_feature_encoder_" + str(Val_way) + "way_" + str(Val_shot) + "shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/" + args.exp + "/miniimagenet_feature_encoder_" + str(Val_way) + "way_" + str(Val_shot) + "shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str("./models/" + args.exp + "/miniimagenet_relation_network_" + str(Val_way) + "way_" + str(Val_shot) + "shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/" + args.exp + "/miniimagenet_relation_network_" + str(Val_way) + "way_" + str(Val_shot) + "shot.pkl")))
        print("load relation network success")
    accuracy = []
    for _ in range(10):
        accuracy.append(test_one(metatest_folders, feature_encoder, relation_network))
    print('accuracies', accuracy)
    print('mean', np.mean(accuracy), 'std', np.std(accuracy), 'max', np.max(accuracy), 'min',np.min(accuracy))

if __name__ == '__main__':
    print("Starting at: ", time.ctime())
    if args.train:
        main()
        test_ten()
    else:
        test_ten()
    print("Ending at: ", time.ctime())