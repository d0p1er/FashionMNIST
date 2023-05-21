import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import FashionMNIST

from matplotlib import pyplot as plt
import imageio
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import os


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out,
                              kernel_size=(3, 3), stride=stride)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)
        return x


class NeuralNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        layer_config = ((64, 2), (64, 1), (128, 2), (128, 1))

        ch_in = 1
        block_list = []
        for ch_out, stride in layer_config:
            block = ConvBlock(ch_in, ch_out, stride)
            block_list.append(block)
            ch_in = ch_out

        self.backbone = nn.Sequential(*block_list)
        bottleneck_channel = 2
        self.bottleneck = nn.Linear(layer_config[-1][0], bottleneck_channel)
        self.head = nn.Linear(bottleneck_channel, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, input):
        featuremap = self.backbone(input)
        squashed = F.adaptive_avg_pool2d(featuremap, output_size=(1, 1))
        squeezed = squashed.view(squashed.shape[0], -1)
        self.bottleneck_out = self.bottleneck(squeezed)
        pred = self.head(self.bottleneck_out)

        self.softmax_vals = self.softmax(pred)
        return pred

    @classmethod
    def loss(cls, pred, gt):
        return F.cross_entropy(pred, gt)


class Trainer:
    def __init__(self):

        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=(28, 28), scale=(0.7, 1.1)),
            transforms.ToTensor(),
        ])
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = FashionMNIST("./data", train=True,
                                     transform=self.train_transform,
                                     download=True)
        self.val_dataset = FashionMNIST("./data", train=False,
                                   transform=self.val_transform,
                                   download=True)

        self.samples = {}
        [self.samples.setdefault(i, []) for i in range(10)]
        for sample, label in self.val_dataset:
            if len(self.samples[label]) < 10:
                self.samples[label].append(sample)

        batch_size = 1024
        self.train_loader = data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True, num_workers=4)
        self.val_loader = data.DataLoader(self.val_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=4)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        print(self.device)

        self.net = NeuralNet()
        self.net.to(self.device)

        self.logger = SummaryWriter()
        self.i_batch = 0

    def train(self):

        num_epochs = 100

        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

        for i_epoch in range(num_epochs):
            self.net.train()

            for feature_batch, gt_batch in self.train_loader:
                feature_batch = feature_batch.to(self.device)
                gt_batch = gt_batch.to(self.device)

                pred_batch = self.net(feature_batch)

                loss = NeuralNet.loss(pred_batch, gt_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.logger.add_scalar("train/loss", loss.item(), self.i_batch)

                if self.i_batch % 100 == 0:
                    print(f"batch={self.i_batch} loss={loss.item():.6f}")

                self.i_batch += 1

            self.validate()
            self.savefig(i_epoch)

            torch.save(self.net, "model.pth")

    def validate(self):
        self.net.eval()

        loss_all = []
        pred_all = []
        gt_all = []
        for feature_batch, gt_batch in self.val_loader:
            feature_batch = feature_batch.to(self.device)
            gt_batch = gt_batch.to(self.device)

            with torch.no_grad():
                pred_batch = self.net(feature_batch)
                loss = NeuralNet.loss(pred_batch, gt_batch)

            loss_all.append(loss.item())
            pred_all.append(pred_batch.cpu().numpy())
            gt_all.append(gt_batch.cpu().numpy())

        loss_mean = np.mean(np.array(loss_all))
        pred_all = np.argmax(np.concatenate(pred_all, axis=0), axis=1)
        gt_all = np.concatenate(np.array(gt_all))

        accuracy = np.sum(np.equal(pred_all, gt_all)) / len(pred_all)

        self.logger.add_scalar("val/loss", loss_mean, self.i_batch)
        self.logger.add_scalar("val/accuracy", accuracy, self.i_batch)

        print(f"Val_loss={loss_mean} val_accu={accuracy:.6f}")

    def savefig(self, i_epoch):
        self.net.eval()
        samples_batch = np.empty(shape=(100, self.samples[0][0].shape[0], self.samples[0][0].shape[1], self.samples[0][0].shape[2]), dtype=np.float32)
        current_sample = 0
        cmap = []

        for key, samples in self.samples.items():
            for sample in samples:
                cmap.append(key)
                samples_batch[current_sample] = sample
                current_sample += 1

        samples_batch = torch.from_numpy(samples_batch).to(self.device)

        with torch.no_grad():
            pred_batch = self.net(samples_batch)

        x = self.net.bottleneck_out[:, 0].cpu()
        y = self.net.bottleneck_out[:, 1].cpu()

        plt.title("epoch {}".format(i_epoch))

        plt.xlim(-100, 100)
        plt.ylim(-100, 100)

        colors = np.array(["black", "green", "red", "purple", "blue", "yellow", "orange", "pink", "grey", "brown"])
        plt.scatter(x, y, c=colors[cmap])

        plt.savefig("images/train/train_{}.png".format(i_epoch))
        plt.close()

    def miss_classificator(self):
        pred_all = []
        gt_all = []
        softmax_all = []

        for feature_batch, gt_batch in self.val_loader:
            feature_batch = feature_batch.to(self.device)
            gt_batch = gt_batch.to(self.device)

            with torch.no_grad():
                pred_batch = self.net(feature_batch)
                loss = NeuralNet.loss(pred_batch, gt_batch)
                softmax = self.net.softmax_vals

            pred_all.append(pred_batch.cpu().numpy()) 
            gt_all.append(gt_batch.cpu().numpy())
            softmax_all.append(softmax.cpu().numpy())

        pred_all = np.argmax(np.concatenate(pred_all, axis=0), axis=1)
        gt_all = np.concatenate(np.array(gt_all))
        softmax_all = np.max(np.concatenate(softmax_all, axis=0), axis=1)

        validation_pictures = []
        for sample, label in self.val_dataset:
            validation_pictures.append(sample)

        softmax_all, gt_all, pred_all, validation_pictures = (list(t) for t in zip(*sorted(zip(softmax_all, gt_all, pred_all, validation_pictures), key=lambda x: x[0])))

        classes = ["T-Shirt", "Trousers", "Pullover", "Dress", "Coat", "Sandals", "Shirt", "Sneakers", "Bag", "Ankle Boots"]

        for class_num in range(10):
            idx = 0
            mis_idx = 0
            print("Correct class:", classes[class_num])
            os.mkdir('images/classes/{}'.format(''.join(classes[class_num].split())))

            while (mis_idx < 5):
                if (gt_all[idx] != class_num):
                    idx += 1
                elif (gt_all[idx] == pred_all[idx]):
                    idx += 1
                else:
                    plt.imshow(validation_pictures[idx][0])
                    print("Miss class:", classes[pred_all[idx]])
                    plt.savefig("images/classes/{}/{}_{}.png".format(''.join(classes[class_num].split()), ''.join(classes[pred_all[idx]].split()), mis_idx))
                    # plt.show()
                    mis_idx += 1
                    idx += 1

        plt.close()

        # confusion matrix
        confusion = confusion_matrix(gt_all, pred_all)
        show = ConfusionMatrixDisplay(confusion)

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.xaxis.set_ticklabels(classes)
        ax.yaxis.set_ticklabels(classes)
        fig.suptitle('Confusion matrix')
        show.plot(ax=ax)

        plt.savefig("images/ConfusionMatrix.png")
        plt.show()


def main():
    print('what to do?')
    print('1) train: train')
    print('2) information bottleneck: ib')
    print('3) compute mis classifications: mis')
    todo = input()
    if (todo == 'train'):
        trainer = Trainer()
        trainer.train()
    elif (todo == 'ib'):
        images = []
        for i in range(100):
            image = imageio.imread('images/train/train_{}.png'.format(i))
            images.append(image)
        imageio.mimsave('images/train/information_bottleneck.gif', images, duration=0.2)
    elif (todo == 'mis'):
        trainer = Trainer()
        trainer.net = NeuralNet()
        trainer.net.load_state_dict(torch.load("model.pth").state_dict())
        trainer.net.eval()

        trainer.miss_classificator()

    print("Done!")


if __name__ == "__main__":
    main()
