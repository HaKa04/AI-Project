import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.functional import F
import matplotlib.pyplot as plt
import time
import os
import pickle
want_plot = False
cuda = torch.cuda.is_available()
# cuda = False
# Laden des CIFAR-10-Datensatzes
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# Einfaches CNN-Modell
class SimpleCNN(nn.Module):
    def __init__(self, first_pannel_number, second_pannel_number, third_pannel_number,kernel_size, first_conneceted_layer_number, second_conneceted_layer_number,dropout):
        super(SimpleCNN, self).__init__()
        self.third_pannel_number = third_pannel_number
        padding = 1 if kernel_size == 3 else 2
        self.conv1 = nn.Conv2d(3, first_pannel_number, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(first_pannel_number)  # BatchNorm after first conv layer
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(first_pannel_number, second_pannel_number, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(second_pannel_number)  # BatchNorm after second conv layer
        self.conv3 = nn.Conv2d(second_pannel_number, third_pannel_number, kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(third_pannel_number)
        self.fc1 = nn.Linear(third_pannel_number * 8 * 8, first_conneceted_layer_number)
        self.fc2 = nn.Linear(first_conneceted_layer_number, second_conneceted_layer_number)
        self.fc3 = nn.Linear(second_conneceted_layer_number, 10)
        self.dropout = nn.Dropout(dropout)  # 50% Dropout


    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        X = self.dropout(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(-1, self.third_pannel_number * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

list_of_model_constructions = [[[32,64,128],5,[256,128],0.0004,'Adam', 64,0.5]] # replace with your list of model constructions
start = time.perf_counter()
for model_construction_numbers in list_of_model_constructions:

    conv_part = '_'.join(str(num) for num in model_construction_numbers[0])
    fc_part = '_'.join(str(num) for num in model_construction_numbers[2])
    learning_rate = model_construction_numbers[3]
    kernel_size = model_construction_numbers[1]
    optimizer_type = model_construction_numbers[4]
    batch_size = model_construction_numbers[5]
    dropout = model_construction_numbers[6]

    model_construction = f'conv_{conv_part}_ks_{kernel_size}_fc_{fc_part}_lr_{learning_rate}_opt_{optimizer_type}_bs_{batch_size}_do_{dropout}'

    #model_construction = '16_32-8-8_128'
    first_pannel_number, second_pannel_number, third_pannel_number = model_construction_numbers[0][0], model_construction_numbers[0][1],model_construction_numbers[0][2]
    first_conneceted_layer_number, second_conneceted_layer_number= model_construction_numbers[2][0], model_construction_numbers[2][1]
    # Modell, Verlustfunktion und Optimierer erstellen
    model = SimpleCNN(first_pannel_number, second_pannel_number, third_pannel_number, kernel_size, first_conneceted_layer_number, second_conneceted_layer_number, dropout)
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    

    if os.path.exists(os.path.join('models','neural_net', f'final_model_{model_construction}.pth')):
        model.load_state_dict(torch.load(os.path.join('models','neural_net', f'final_model_{model_construction}.pth')))
    if os.path.exists(os.path.join('models','optimizer', f'final_optimizer_{model_construction}.pth')):
        optimizer.load_state_dict(torch.load(os.path.join('models','optimizer', f'final_optimizer_{model_construction}.pth')))

    if cuda:
        model.cuda()
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    criterion = nn.CrossEntropyLoss()
    # Listen zum Speichern der Genauigkeiten
    if os.path.exists(os.path.join('models','train_accuracy', f'final_train_{model_construction}.pkl')):
        with open(os.path.join('models','train_accuracy', f'final_train_{model_construction}.pkl'), 'rb') as f:
            train_accuracies = pickle.load(f)
    else:  
        train_accuracies = []
    if os.path.exists(os.path.join('models','test_accuracy', f'final_test_{model_construction}.pkl')):
        with open(os.path.join('models','test_accuracy', f'final_test_{model_construction}.pkl'), 'rb') as f:
            test_accuracies = pickle.load(f)
    else:
        test_accuracies = []

    os.makedirs(os.path.join('models','neural_net', f'running_{model_construction}'), exist_ok=True)
    os.makedirs(os.path.join('models','optimizer', f'running_{model_construction}'), exist_ok=True)
    os.makedirs(os.path.join('models','train_accuracy', f'running_{model_construction}'), exist_ok=True)
    os.makedirs(os.path.join('models','test_accuracy', f'running_{model_construction}'), exist_ok=True)

    # Training
    try:
        for epoch in range(101):  # Anzahl der Epochen anpassen
            if len(train_accuracies) % 5 == 0:
                print('train_iteration: ', len(train_accuracies))
            correct_train = 0
            total_train = 0
            for images, labels in train_loader:
                if cuda:
                    images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted_train = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()

            # Testen
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    if cuda:
                        images, labels = images.cuda(), labels.cuda()
                    outputs = model(images)
                    _, predicted_test = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted_test == labels).sum().item()

            train_accuracies.append(100 * correct_train / total_train)
            test_accuracies.append(100 * correct_test / total_test)

            if len(train_accuracies) % 10 == 0:
                torch.save(model.state_dict(), os.path.join('models','neural_net', f'running_{model_construction}', f'running_model_epoch({len(test_accuracies)})_{model_construction}.pth'))
                torch.save(optimizer.state_dict(), os.path.join('models','optimizer', f'running_{model_construction}', f'running_optimizer_epoch({len(test_accuracies)})_{model_construction}.pth'))
                with open(os.path.join('models','train_accuracy', f'running_{model_construction}', f'running_train_epoch({len(test_accuracies)})_{model_construction}.pkl'), 'wb') as f:
                    pickle.dump(train_accuracies, f)
                with open(os.path.join('models','test_accuracy', f'running_{model_construction}', f'running_test_epoch({len(test_accuracies)})_{model_construction}.pkl'), 'wb') as f:
                    pickle.dump(test_accuracies, f)
    except KeyboardInterrupt:
        pass

    # Genauigkeiten plotten
    if want_plot:
        plt.plot(train_accuracies, label='Trainingsgenauigkeit')
        plt.plot(test_accuracies, label='Testgenauigkeit')
        plt.xlabel('Epoche')
        plt.ylabel('Genauigkeit')
        plt.legend()
        plt.show()

    # Modell speichern
    model.cpu()
    if cuda:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()

    torch.save(model.state_dict(), os.path.join('models','neural_net', f'final_model_{model_construction}.pth'))
    # Optimizer speichern
    torch.save(optimizer.state_dict(), os.path.join('models','optimizer', f'final_optimizer_{model_construction}.pth'))

    # Speichern der Listen
    with open(os.path.join('models','train_accuracy', f'final_train_{model_construction}.pkl'), 'wb') as f:
        pickle.dump(train_accuracies, f)

    with open(os.path.join('models','test_accuracy', f'final_test_{model_construction}.pkl'), 'wb') as f:
        pickle.dump(test_accuracies, f)

end = time.perf_counter()
print('Dauer: ', end - start)