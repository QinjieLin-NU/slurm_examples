import argparse
from filelock import FileLock
import os

import torch
from torch import nn, optim
from torchvision import datasets, transforms

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
parser.add_argument('--data_dir', type=str, default='~/.pytorch/MNIST_data/', help='Directory for the MNIST data')
parser.add_argument('--model_dir', type=str, default='./model', help='Directory to save the model')
args = parser.parse_args()

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Lock to download files
lock_path = f"{os.path.expanduser(args.data_dir)}/mnist.lock"
os.makedirs(os.path.expanduser(args.data_dir), exist_ok=True)
with FileLock(lock_path):
    trainset = datasets.MNIST(args.data_dir, download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

# Move the model and data to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr)

# Training loop
for e in range(args.epochs):
    running_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

# Save the model
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
print(f"saving model to {args.model_dir}.")
torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))