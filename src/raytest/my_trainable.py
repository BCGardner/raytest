from pathlib import Path

from typing import Dict, Optional, Union
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from ray.air import session
from ray import tune

from .model import ConvNet
from .utils import train, test


class Trainable(tune.Trainable):
    def setup(self, config):
        # Data Setup
        mnist_transforms = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))])

        self.train_loader = DataLoader(
            datasets.MNIST("~/data", train=True, download=True, transform=mnist_transforms),
            batch_size=64,
            shuffle=True)
        self.test_loader = DataLoader(
            datasets.MNIST("~/data", train=False, transform=mnist_transforms),
            batch_size=64,
            shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ConvNet()
        self.model.to(device)

        self.optimizer = optim.SGD(
            self.model.parameters(), lr=config["lr"], momentum=config["momentum"]
        )

    def step(self):  # This is called iteratively.
        train(self.model, self.optimizer, self.train_loader)
        acc = test(self.model, self.test_loader)
        return {"mean_accuracy": acc}

    def save_checkpoint(self, checkpoint_dir: str) -> str | Dict | None:
        checkpoint = str((Path(checkpoint_dir) / "model.pth").resolve())
        torch.save(self.model.state_dict(), checkpoint)
        return checkpoint

    def load_checkpoint(self, checkpoint: Dict | str):
        self.model.load_state_dict(checkpoint)


def train_mnist(config):
    # Data Setup
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    train_loader = DataLoader(
        datasets.MNIST("~/data", train=True, download=True, transform=mnist_transforms),
        batch_size=64,
        shuffle=True)
    test_loader = DataLoader(
        datasets.MNIST("~/data", train=False, transform=mnist_transforms),
        batch_size=64,
        shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNet()
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"])
    for i in range(10):
        train(model, optimizer, train_loader)
        acc = test(model, test_loader)

        # Send the current training result back to Tune
        session.report({"mean_accuracy": acc})

        if i % 5 == 0:
            # This saves the model to the trial directory
            torch.save(model.state_dict(), "./model.pth")
