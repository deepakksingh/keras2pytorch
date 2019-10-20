
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch.optim as optim
import torch
import torchvision

batch_size = 128
num_epochs = 12


# Note that Torch has channels first by default (M, C, W, H) whereas Keras has
# channels last (M, W, H, C)
class mnist_cnn(nn.Module):
    def __init__(self):
        super(mnist_cnn, self).__init__()
        # input shape = (1,28,28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        # output shape = (32,26,26) where 26 = (28-3+1)/1
        # Params = ((3*3)*1 +  1)*32 = 320
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        # output shape = (64,24,24) where 24 = (26-3+1)/1
        # Params = ((3*3)*32 + 1)*64 = 18496
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        # output shape = (64,12,12), Params = 0
        self.dropout1 = nn.Dropout(p=0.25)
        # output shape = (64,12,12), Params = 0
        self.flatten = nn.Flatten()
        # output shape = (9216,), Params = 0
        self.fc1 = nn.Linear(in_features=9216, out_features=128)
        # output shape = (128,)
        # Params = (9216+1)*128 = 1179776
        self.dropout2 = nn.Dropout(p=0.5)
        # output shape = (128,), Params = 0
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        # output shape = (10,)
        # Params = (128+1)*10 = 1290

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def train(model, train_dataloader, optimizer, device, epoch, writer):
    model.train() # sets to training mode
    num_batches = len(train_dataloader.dataset)/batch_size+1
    for batch_id, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        optimizer.zero_grad()
        loss = F.cross_entropy(input=output, target=target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        print("[Train] Epoch: {} [{}/{}]    Loss: {:.6f}   Batch Acc: {:.2f}".format(
              epoch, batch_id*batch_size, len(train_dataloader.dataset),
              loss.item(), correct/batch_size*100))
        writer.add_scalar('Loss/Train', loss.item(), num_batches*epoch+batch_id)
        writer.add_scalar('Accuracy/Train', correct/batch_size*100, num_batches*epoch+batch_id)


def test(model, test_dataloader, device, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_dataloader):
                grid = torchvision.utils.make_grid(data)
                writer.add_image('images', grid, 0)
                writer.add_graph(model, data)
                writer.close()

                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_dataloader.dataset)

        print("[Test] Epoch: {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
              epoch, test_loss, correct, len(test_dataloader.dataset),
              100.*correct/len(test_dataloader.dataset))
             )
        writer.add_scalar('Loss/Eval', test_loss, epoch)
        writer.add_scalar('Accuracy/Eval', 100.*correct/len(test_dataloader.dataset), epoch)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mnist_cnn()
    model.to(device)
    print(model)

    transform = transforms.Compose([transforms.ToTensor()]) # converts to 0-1 and makes it (M, 1, H, W)
    mnist_train = datasets.MNIST('/tmp/', train=True, download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

    mnist_test = datasets.MNIST('/tmp/', train=False, download=True, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    writer = SummaryWriter()
    for epoch in range(num_epochs):
        train(model, train_dataloader, optimizer, device, epoch, writer)
        test(model, test_dataloader, device, epoch, writer)


if __name__=="__main__":
    main()
