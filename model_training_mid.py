import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from progress_bar import progress_bar
import models.resnet_last_down_extract as resnet_down
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
torch.cuda.empty_cache()
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training with KD')
parser.add_argument('--batch', default=256, type=int, help='batch size')
parser.add_argument('--shuffle', default=True, type=bool, help='shuffle the training dataset')
parser.add_argument('--model', type=str, required=True, help='---Model type: resnet18, resnet34, resnet50---')
parser.add_argument('--model_kd', type=str, required=True, help='---Model type: resnet18, resnet34, resnet50---')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--pair_keys', type=int, required=True, help='---Indicate pair of keys unique for teacher and student---')
parser.add_argument('--alpha', type=float, default=0.3, help='---Distillation weight (alpha) (default: 0.3)---')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=300, type=int, help='epoch number')
args, unparsed = parser.parse_known_args()

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
best_acc = 0
train_accuracy = []
val_accuracy = []
train_loss = []
val_loss = []
train_accuracy_kd = []
val_accuracy_kd = []
train_loss_kd = []
val_loss_kd = []


def build_model():
    if args.model == 'resnet18':
        return resnet_down.__dict__[model_names[0]]()
    elif args.model == 'resnet34':
        return resnet_down.__dict__[model_names[1]]()
    elif args.model == 'resnet50':
        return resnet_down.__dict__[model_names[2]]()
    elif args.model == 'resnet101':
        return resnet_down.__dict__[model_names[3]]()
    elif args.model == 'resnet152':
        return resnet_down.__dict__[model_names[4]]()


def build_model_kd():
    if args.model_kd == 'resnet18':
        return resnet_down.__dict__[model_names[0]]()
    elif args.model_kd == 'resnet34':
        return resnet_down.__dict__[model_names[1]]()
    elif args.model_kd == 'resnet50':
        return resnet_down.__dict__[model_names[2]]()
    elif args.model_kd == 'resnet101':
        return resnet_down.__dict__[model_names[3]]()
    elif args.model_kd == 'resnet152':
        return resnet_down.__dict__[model_names[4]]()

print('Teacher model type: ', args.model)
print('Student model type: ', args.model_kd)

print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()])
# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    transforms.ToTensor()])
# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

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


def train(model, loader, optimizer):
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output1, output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        progress_bar(batch_idx, len(trainLoader), 'Teacher: Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return correct, train_loss


def train_distil(model, distil_model, loader, optimizer, distil_weights):
    model.eval()
    distil_model.train()

    train_loss_kd = 0
    correct_kd = 0
    total_kd = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output1_t, output_t = model(data)
        output1_s, output_s = distil_model(data)
        kd_loss = F.mse_loss(output1_s, output1_t.detach()) * distil_weights
        kd_loss_cls = criterion(output_s, target)
        loss_kd = kd_loss + kd_loss_cls
        loss_kd.backward()
        optimizer.step()

        train_loss_kd += loss_kd.item()
        _, predicted = output_s.max(1)
        total_kd += target.size(0)
        correct_kd += predicted.eq(target).sum().item()
        progress_bar(batch_idx, len(trainLoader), 'Student: Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss_kd / (batch_idx + 1), 100. * correct_kd / total_kd, correct_kd, total_kd))
    return correct_kd, train_loss_kd


def validate(model, loader):
    model.eval()

    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            output_1, output = model(data)
            loss = criterion(output, target)

            val_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            progress_bar(batch_idx, len(testLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return correct, val_loss


models_teacher = build_model().to(device)
distil_models = build_model_kd().to(device)
criterion = nn.CrossEntropyLoss()
optimizer_teacher = optim.SGD(models_teacher.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
scheduler_teacher = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_teacher, T_max=200)

optimizer_student = optim.SGD(distil_models.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
scheduler_student = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_student, T_max=200)

distil_weight = args.alpha
best_loss = sys.maxsize
best_loss_kd = sys.maxsize

print("Training Teacher first... =====>")
for epoch in range(args.epoch):
    print('\nTeacher Epoch: %d' % epoch)
    train_correct, training_loss = train(models_teacher, trainLoader, optimizer_teacher)
    val_correct, validating_loss = validate(models_teacher, testLoader)
    scheduler_teacher.step()
    train_accuracy.append(train_correct)
    train_loss.append(training_loss)

    val_accuracy.append(val_correct)
    val_loss.append(validating_loss)
    torch.cuda.empty_cache()
    if epoch >= 0 and (validating_loss - best_loss) < 0:
        best_loss = validating_loss
        torch.save(models_teacher.state_dict(), f'./vanilla_kd_model_saved_base/{args.model}_teacher.pth')
print('\n')

print("Training Student second... =====>")
for epoch in range(args.epoch):
    print('\nStudent Epoch: %d' % epoch)
    train_correct_kd, training_loss_kd = train_distil(models_teacher, distil_models, trainLoader, optimizer_student,
                                                      distil_weight)
    val_correct_kd, validating_loss_kd = validate(distil_models, testLoader)
    scheduler_student.step()

    train_accuracy_kd.append(train_correct_kd)
    train_loss_kd.append(training_loss_kd)

    val_accuracy_kd.append(val_correct_kd)
    val_loss_kd.append(validating_loss_kd)
    torch.cuda.empty_cache()
    if epoch >= 0 and (validating_loss_kd - best_loss_kd) < 0:
        best_loss_kd = validating_loss_kd
        torch.save(distil_models.state_dict(), f'./vanilla_kd_model_saved_base/{args.model}_student.pth')

train_accuracy_np = np.asarray(train_accuracy)
train_loss_np = np.asarray(train_loss)

val_accuracy_np = np.asarray(val_accuracy)
val_loss_np = np.asarray(val_loss)

np.save('./numpy_outputs/train_accuracy_teacher', train_accuracy_np)
np.save('./numpy_outputs/train_loss_teacher', train_loss_np)

np.save('./numpy_outputs/val_accuracy_teacher', val_accuracy_np)
np.save('./numpy_outputs/val_loss_teacher', val_loss_np)

train_accuracy_np_kd = np.asarray(train_accuracy_kd)
train_loss_np_kd = np.asarray(train_loss_kd)

val_accuracy_np_kd = np.asarray(val_accuracy_kd)
val_loss_np_kd = np.asarray(val_loss_kd)

np.save('./numpy_outputs/train_accuracy_student', train_accuracy_np_kd)
np.save('./numpy_outputs/train_loss_student', train_loss_np_kd)

np.save('./numpy_outputs/val_accuracy_student', val_accuracy_np_kd)
np.save('./numpy_outputs/val_loss_student', val_loss_np_kd)
