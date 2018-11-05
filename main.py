import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets, models
import torch.nn.functional as F
from sklearn.datasets import fetch_olivetti_faces

from datasets import OlivettiDataset
from networks import SiameseNet
from losses import ContrastiveLoss

import numpy as np
import matplotlib.pyplot as plt

use_gpu = torch.cuda.is_available()

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
ds_trans = transforms.Compose([transforms.Resize(224),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               normalize])

olivetti_train_dataset = OlivettiDataset(olivetti_dataset=fetch_olivetti_faces(data_home='data/olivetti'), train=1, transform=ds_trans)
olivetti_valid_dataset = OlivettiDataset(olivetti_dataset=fetch_olivetti_faces(data_home='data/olivetti'), train=0, transform=ds_trans)

train_dataloader = DataLoader(olivetti_train_dataset, batch_size=4)
valid_dataloader = DataLoader(olivetti_valid_dataset, batch_size=4)

def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs=1):
    use_gpu = torch.cuda.is_available()
    dataset_sizes = {'train': len(dataloaders['train'].dataset),
                     'valid': len(dataloaders['valid'].dataset)}
    for epoch in range(num_epochs):
        showed_images = False
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)
            running_loss = 0.0

            for inputs1, inputs2, targets in dataloaders[phase]:
                if use_gpu:
                    inputs1, inputs2, targets = Variable(inputs1.cuda()), Variable(inputs2.cuda()), Variable(targets.cuda())
                else:
                    inputs1, inputs2, targets = Variable(inputs1), Variable(inputs2), Variable(targets)

                optimizer.zero_grad()

                outputs1, outputs2 = model(inputs1, inputs2)
                distances = F.pairwise_distance(outputs1, outputs2)
                loss = criterion(outputs1, outputs2, targets)
                # print(targets, loss.item())

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                if epoch % 10 == 9:
                    if phase == 'valid' and showed_images == False:
                        if 1 in targets and 0 in targets:
                            show_images(inputs1, inputs2, targets, distances)
                            showed_images = True

                running_loss += loss.data.item()

            if phase == 'train':
                train_epoch_loss = running_loss / dataset_sizes[phase]
            else:
                valid_epoch_loss = running_loss / dataset_sizes[phase]

        print('Epoch [{}/{}] train loss: {:.4f} ' 
              'valid loss: {:.4f}'.format(
                epoch, num_epochs - 1,
                train_epoch_loss, 
                valid_epoch_loss))
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_img(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp
    # plt.imshow(inp)
    # if title is not None:
    #     plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated
    # plt.show()

def show_images(inputs1, inputs2, targets, distances):
    fig, ax = plt.subplots(nrows=len(targets), ncols=2)
    for row_num in range(len(targets)):
        img1 = ax[row_num, 0].imshow(get_img(inputs1.cpu().data[row_num]))
        img2 = ax[row_num, 1].imshow(get_img(inputs2.cpu().data[row_num]))
        ax[row_num, 0].xaxis.set_visible(False)
        ax[row_num, 0].yaxis.set_visible(False)
        ax[row_num, 1].xaxis.set_visible(False)
        ax[row_num, 1].yaxis.set_visible(False)
        t = ax[row_num, 0].set_title("Target: {}\nDistance = {}".format(targets.cpu().data[row_num].item(), distances[row_num]))
    plt.tight_layout()
    plt.show()

def visualize_model(model, num_samples=3):
    # was_training = model.training
    was_training = False
    model.eval()
    images_so_far = 0

    with torch.no_grad():
        for i, (inputs1, inputs2, target) in enumerate(dloaders['valid']):
            if i == 0:
                inputs1 = inputs1.to(device)
                inputs2 = inputs2.to(device)
                target = target.to(device)

                outputs1, outputs2 = model(inputs1, inputs2)

                fig, ax = plt.subplots(nrows=num_samples, ncols=2)

                for row_num in range(num_samples):
                    img1 = ax[row_num, 0].imshow(get_img(inputs1.cpu().data[row_num]))
                    img2 = ax[row_num, 1].imshow(get_img(inputs2.cpu().data[row_num]))
                    ax[row_num, 0].xaxis.set_visible(False)
                    ax[row_num, 0].yaxis.set_visible(False)
                    ax[row_num, 1].xaxis.set_visible(False)
                    ax[row_num, 1].yaxis.set_visible(False)
                    ax[row_num, 0].set_title("Target: {}".format(target.cpu().data[row_num].item()))

                # for j in range(inputs1.size()[0]):
                #     images_so_far += 1
                #     ax = plt.subplot(num_samples, 2, images_so_far)
                #     ax.axis('off')
                #     ax.set_title('target: {}'.format(target[j]))
                #     imshow(inputs1.cpu().data[j])

                    if images_so_far == num_samples:
                        model.train(mode=was_training)
                        return
            plt.show()
        model.train(mode=was_training)

dloaders = {'train':train_dataloader, 'valid':valid_dataloader}
siameseNet = SiameseNet().cuda()
criterion = ContrastiveLoss(margin=10)
optimizer = optim.Adam(siameseNet.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model_ft = train_model(dloaders, siameseNet, criterion, optimizer, exp_lr_scheduler, num_epochs=50)

# visualize_model(model_ft)