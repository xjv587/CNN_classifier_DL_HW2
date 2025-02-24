import torch
import torch.optim as optim
import torch.utils.tensorboard as tb
import tempfile
import sys
from os import path
from torchvision import transforms
from torch.utils.data import DataLoader
from .models import CNNClassifier, save_model
from .utils import SuperTuxDataset, accuracy, load_data


log_dir = tempfile.mkdtemp()
train_logger = tb.SummaryWriter(log_dir + '/model/train')
valid_logger = tb.SummaryWriter(log_dir + '/model/valid')

tdata_path = "data/train"
vdata_path = "data/valid"

batch_size = 32
learning_rate = 0.01
num_epochs = 10

def train(args):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train_dataset = load_data(tdata_path, batch_size=batch_size)
    valid_dataset = load_data(vdata_path, batch_size=batch_size)

    model = CNNClassifier()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, data in enumerate(train_dataset, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            global_step = epoch * len(train_dataset) + i
            train_logger.add_scalar('loss', loss.item(), global_step=global_step)

        train_accuracy = 100 * correct_train / total_train
        train_logger.add_scalar('accuracy', train_accuracy, global_step=epoch)

        model.eval()
        correct_valid = 0
        total_valid = 0

        with torch.no_grad():
            for data in valid_dataset:
                inputs, labels = data
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()

        valid_accuracy = 100 * correct_valid / total_valid
        valid_logger.add_scalar('accuracy', valid_accuracy, global_step=epoch)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataset):.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, Valid Acc: {valid_accuracy:.2f}%')

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
