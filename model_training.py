import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from progress_bar import *
import models.resnet as resnet
torch.cuda.empty_cache()
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False



parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--batch', default=256, type=int, help='batch size')
parser.add_argument('--shuffle', default=True, type=bool, help='shuffle the training dataset')
parser.add_argument('--model', type=str, required=True, help='---Model type: conv, googlenet, resnet34, resnet50---')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=600, type=int, help='epoch number')
args, unparsed = parser.parse_known_args()



device = 'cuda:1' if torch.cuda.is_available() else 'cpu'


model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
best_loss = sys.maxsize
train_accuracy = []
val_accuracy = []
train_loss = []
val_loss = []


def build_model():
    if args.model == 'resnet18':
        return resnet.__dict__[model_names[0]]()
    elif args.model == 'resnet34':
        return resnet.__dict__[model_names[1]]()
    elif args.model == 'resnet50':
        return resnet.__dict__[model_names[2]]()
    elif args.model == 'resnet101':
        return resnet.__dict__[model_names[3]]()
    elif args.model == 'resnet152':
        return resnet.__dict__[model_names[4]]()


print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    #transforms.Grayscale(),
    transforms.ToTensor()])
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    #transforms.Grayscale(),
    transforms.ToTensor()])
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


trainset = torchvision.datasets.CIFAR10(
    root='./cifar10', train=True, download=True, transform=transform_train)

trainLoader = DataLoader(
    trainset, batch_size=args.batch, shuffle=args.shuffle, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./cifar10', train=False, download=True, transform=transform_test)

testLoader = DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Building model..')

models = build_model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(models.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def train(model, loader, optimizer):
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        progress_bar(batch_idx, len(trainLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return correct, train_loss


def validate(model, loader):
    model.eval()

    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            val_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            progress_bar(batch_idx, len(testLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return correct, val_loss


for epoch in range(args.epoch):
    print('\nEpoch: %d' % (epoch+1))
    train_correct, training_loss = train(models, trainLoader, optimizer)
    val_correct, validating_loss = validate(models, testLoader)
    scheduler.step()
    train_accuracy.append(train_correct)
    train_loss.append(training_loss)

    val_accuracy.append(val_correct)
    val_loss.append(validating_loss)
    torch.cuda.empty_cache()
    if epoch >= 0 and (validating_loss - best_loss) < 0:
        best_loss = validating_loss
        torch.save(models.state_dict(), f'./base_model_saved/{args.model}_base.pth')

train_accuracy_np = np.asarray(train_accuracy)
train_loss_np = np.asarray(train_loss)

val_accuracy_np = np.asarray(val_accuracy)
val_loss_np = np.asarray(val_loss)

np.save(f'./numpy_outputs/train_accuracy_{args.model}', train_accuracy_np)
np.save(f'./numpy_outputs/train_loss_{args.model}', train_loss_np)

np.save(f'./numpy_outputs/val_accuracy_{args.model}', val_accuracy_np)
np.save(f'./numpy_outputs/val_loss_{args.model}', val_loss_np)

