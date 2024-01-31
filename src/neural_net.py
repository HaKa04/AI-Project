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
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

# Modell, Verlustfunktion und Optimierer erstellen
model = SimpleCNN()
model_construction = '16_32-8-8_128'
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

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
start = time.perf_counter()
try:
    for epoch in range(100):  # Anzahl der Epochen anpassen
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

        train_accuracies.append(100 * correct_train / total_train)

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

        test_accuracies.append(100 * correct_test / total_test)

        if len(train_accuracies) % 3 == 0:
            torch.save(model.state_dict(), os.path.join('models','neural_net', f'running_{model_construction}', f'running_model_epoch({len(test_accuracies)})_{model_construction}.pth'))
            torch.save(optimizer.state_dict(), os.path.join('models','optimizer', f'running_{model_construction}', f'running_optimizer_epoch({len(test_accuracies)})_{model_construction}.pth'))
            with open(os.path.join('models','train_accuracy', f'running_{model_construction}', f'running_train_epoch({len(test_accuracies)})_{model_construction}.pkl'), 'wb') as f:
                pickle.dump(train_accuracies, f)
            with open(os.path.join('models','test_accuracy', f'running_{model_construction}', f'running_test_epoch({len(test_accuracies)})_{model_construction}.pkl'), 'wb') as f:
                pickle.dump(test_accuracies, f)
except KeyboardInterrupt:
    pass
end = time.perf_counter()
print('Dauer: ', end - start)

# Genauigkeiten plotten
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
