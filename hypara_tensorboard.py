# %%
import torch
import torchvision
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 

# tensorboardのエラー回避
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# %%
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
# Hyperparameters
in_channels = 1
num_classes = 10
num_epochs = 1

# %%
Load Data
train_dataset = datasets.MNIST(
    root="mnist_dataset/", train=True, transform=transforms.ToTensor(), download=True
)

# %%
# 検索したいパラメータをlist形式で入れる
batch_sizes = [8, 64, 128, 256]
learning_rates = [0.1, 0.01, 0.001, 0.0001]
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# %%
# runファイルの保存パスは変えたほうがいい
for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        step = 0
        # Initialize network
        model = CNN(in_channels=in_channels, num_classes=num_classes)
        model.to(device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
        writer = SummaryWriter(
            f"test_runs/MNIST/MiniBatchSize {batch_size} LR {learning_rate}"
        )

        # Visualize model in TensorBoard
        images, _ = next(iter(train_loader))
        writer.add_graph(model, images.to(device))
        writer.close()

        for epoch in range(num_epochs):
            losses = []
            accuracies = []

            for batch_idx, (data, targets) in enumerate(train_loader):
                # Get data to cuda if possible
                data = data.to(device=device)
                targets = targets.to(device=device)

                # forward
                scores = model(data)
                loss = criterion(scores, targets)
                losses.append(loss.item())

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate 'running' training accuracy
                features = data.reshape(data.shape[0], -1)
                img_grid = torchvision.utils.make_grid(data)
                _, predictions = scores.max(1)
                num_correct = (predictions == targets).sum()
                running_train_acc = float(num_correct) / float(data.shape[0])
                accuracies.append(running_train_acc)

                # Plot things to tensorboard
                class_labels = [classes[label] for label in predictions]
                writer.add_image("mnist_images", img_grid)
                writer.add_histogram("fc1", model.fc1.weight)
                writer.add_scalar("Training loss", loss, global_step=step)
                writer.add_scalar(
                    "Training Accuracy", running_train_acc, global_step=step
                )

                if batch_idx == 230:
                    writer.add_embedding(
                        features,
                        metadata=class_labels,
                        label_img=data,
                        global_step=batch_idx,
                    )
                step += 1

            writer.add_hparams(
                {"lr": learning_rate, "bsize": batch_size},
                {
                    "accuracy": sum(accuracies) / len(accuracies),
                    "loss": sum(losses) / len(losses),
                },
            )
            
