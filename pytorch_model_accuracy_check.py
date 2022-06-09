import torchvision
import torchvision.transforms as transforms
import argparse
from progress_bar import progress_bar
import models.resnet_check as resnet
import models.resnet_last_down_extract as resnet_down
import models.resnet_teacher as teacher
import models.resnet_student as student
import models.resnet_teacher_all_fm as teacher_fm
import models.resnet_student_all_fm as student_fm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchsummary import summary

parser = argparse.ArgumentParser(description='Checking the PyTorch model accuracy')
parser.add_argument('--model', type=str, required=True, help='---Model type: resnet18, resnet34, resnet50---')
parser.add_argument('--pair_keys', type=int, required=True,
                    help='---Indicate pair of keys unique for teacher and student---')
parser.add_argument('--type', type=str, default='base', help='---Choose the model either teacher or student---')
parser.add_argument('--base', type=str, required=True, help='---Choose whether the convertable model is base or vanilla---')
args, unparsed = parser.parse_known_args()

model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
#device = 'cuda:1' if torch.cuda.is_available() else 'cpu'


def build_model_base():
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
def build_model_kd_te():
    if args.model == 'resnet18':
        return teacher_fm.__dict__[model_names[0]]()
    elif args.model == 'resnet34':
        return teacher_fm.__dict__[model_names[1]]()
    elif args.model == 'resnet50':
        return teacher_fm.__dict__[model_names[2]]()
    elif args.model == 'resnet101':
        return teacher_fm.__dict__[model_names[3]]()
    elif args.model == 'resnet152':
        return teacher_fm.__dict__[model_names[4]]()
def build_model_kd_st():
    if args.model == 'resnet18':
        return student.__dict__[model_names[0]]()
    elif args.model == 'resnet34':
        return student.__dict__[model_names[1]]()
    elif args.model == 'resnet50':
        return student.__dict__[model_names[2]]()
    elif args.model == 'resnet101':
        return student.__dict__[model_names[3]]()
    elif args.model == 'resnet152':
        return student.__dict__[model_names[4]]()



array = torch.randn((32, 256, 8, 8))
transform_test = transforms.Compose([
    #transforms.Grayscale(),
    transforms.ToTensor()])

testset = torchvision.datasets.CIFAR10(
    root='./cifar10', train=False, download=True, transform=transform_test)

testLoader = DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

#criterion = nn.CrossEntropyLoss()
#number_of_classes = 10
#confusion_matrix = torch.zeros(number_of_classes, number_of_classes)


if args.base == 'yes':
    net = build_model_base()#.to(device)
    summary(net, (3, 32, 32))
    net.load_state_dict(torch.load(f'./base_model_saved/{args.model}_base.pth',
                                   map_location=torch.device('cpu')))
    with torch.no_grad():
        val_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(testLoader):
            output = net(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            progress_bar(batch_idx, len(testLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss / (batch_idx + 1), 100. * correct / total, correct, total))


elif args.base == 'no':
    if args.type == 'teacher':
        net = build_model_kd_te()  # .to(device)

        summary(net, (3, 32, 32))
        net.load_state_dict(torch.load(f'./vanilla_kd_model_saved_base/{args.model}_{args.type}_{args.pair_keys}.pth',
                                       map_location=torch.device('cpu')))
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(testLoader):
                # data, target = data.to(device), target.to(device)
                output_1, output, output_array = net(data)
                #output1_t, output, output_array1, output_array2, output_array3 = net(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                progress_bar(batch_idx, len(testLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (val_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    elif args.type == 'student':
        net = build_model_kd_st()  # .to(device)

        # summary(net, (3, 32, 32))
        net.load_state_dict(torch.load(f'./vanilla_kd_model_saved_base/{args.model}_{args.type}_{args.pair_keys}.pth',
                                       map_location=torch.device('cpu')))
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(testLoader):
                # data, target = data.to(device), target.to(device)
                #output1_s, output, out_array1_s, out_array2_s, out_array3_s = net(data, array, array, array, state=False, impact=0)
                output_1, output, output_array = net(data, array, state=False, impact=0)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                progress_bar(batch_idx, len(testLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss / (batch_idx + 1), 100. * correct / total, correct, total))
