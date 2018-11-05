import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        if torch.cuda.is_available():
            self.resnet = self.resnet.cuda()
        # freeze all model parameters
        for param in self.resnet.parameters():
            param.requires_grad = False
        num_ftrs = self.resnet.fc.out_features
        self.fc = nn.Linear(num_ftrs, 4)

    def forward(self, x1, x2):
        output1 = self.resnet(x1)
        output1 = self.fc(output1)
        output2 = self.resnet(x2)
        output2 = self.fc(output2)
        return output1, output2

# class EmbeddingNet(nn.Module):
#     def __init__(self):
#         super(EmbeddingNet, self).__init__()
#         self.convnet = nn.Sequential(nn.Conv2d(1, 16, 5), nn.ReLU(),
#                                      nn.MaxPool2d(2, stride=2),
#                                      nn.Conv2d(16, 32, 5), nn.ReLU(),
#                                      nn.MaxPool2d(2, stride=2))

#         self.fc = nn.Sequential(nn.Linear(32 * 64 * 64, 512),
#                                 nn.ReLU(),
#                                 nn.Linear(512, 512),
#                                 nn.ReLU(),
#                                 nn.Linear(256, 8)
#                                 )

#     def forward(self, x):
#         output = self.convnet(x)
#         output = output.view(output.size()[0], -1)
#         output = self.fc(output)
#         return output

#     def get_embedding(self, x):
#         return self.forward(x)


# class EmbeddingNetL2(EmbeddingNet):
#     def __init__(self):
#         super(EmbeddingNetL2, self).__init__()

#     def forward(self, x):
#         output = super(EmbeddingNetL2, self).forward(x)
#         output /= output.pow(2).sum(1, keepdim=True).sqrt()
#         return output

#     def get_embedding(self, x):
#         return self.forward(x)


# class ClassificationNet(nn.Module):
#     def __init__(self, embedding_net, n_classes):
#         super(ClassificationNet, self).__init__()
#         self.embedding_net = embedding_net
#         self.n_classes = n_classes
#         self.non_linear = nn.ReLU()
#         self.fc1 = nn.Linear(2, n_classes)

#     def forward(self, x):
#         output = self.embedding_net(x)
#         output = self.non_linear(output)
#         scores = F.log_softmax(self.fc1(output), dim=1)
#         return scores

#     def get_embedding(self, x):
#         return self.non_linear(self.embedding_net(x))


# class SiameseNet(nn.Module):
#     def __init__(self, embedding_net):
#         super(SiameseNet, self).__init__()
#         self.embedding_net = embedding_net

#     def forward(self, x1, x2):
#         output1 = self.embedding_net(x1)
#         output2 = self.embedding_net(x2)
#         return output1, output2

#     def get_embedding(self, x):
#         return self.embedding_net(x)