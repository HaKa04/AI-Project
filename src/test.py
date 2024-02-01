import torch
import os
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.functional import F
import itertools

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)  # BatchNorm after first conv layer
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)  # BatchNorm after second conv layer
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x
    

model_construction = 'conv_32_64_128_ks_5_fc_256_128_lr_0.0004_opt_Adam_bs_64_do_0'
model = SimpleCNN()
model.load_state_dict(torch.load(os.path.join('models','optimal_net', 'running_model_epoch(60)_conv_32_64_128_ks_5_fc_256_128_lr_0.0004_opt_Adam_bs_64_do_0.pth')))
model.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

test_loader = DataLoader(test_dataset, shuffle=False)
'''
# Testen des Modells
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # Index des maximalen Elements
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Genauigkeit des Netzes auf den 10000 Testbildern: %d %%' % (
        100 * correct / total))
'''

# Testen des Modells auf einzelnen Bildern
import matplotlib.pyplot as plt
import numpy as np
import random
# Wählen Sie ein Bild aus dem Testdatensatz
image_number = random.randint(0, len(test_dataset))
image, label = test_dataset[image_number]  # Wählen Sie das erste Bild
image = image.unsqueeze(0)  # Fügen Sie eine zusätzliche Dimension hinzu, um eine Batch-Größe von 1 zu erstellen

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Machen Sie eine Vorhersage mit dem Modell
model.eval()  # Setzen Sie das Modell in den Evaluierungsmodus
with torch.no_grad():
    output = model(image)
predicted_class = torch.argmax(output).item()

predicted_label = class_names[predicted_class]
true_label = class_names[label]

# Zeigen Sie das Bild an
image = image.squeeze(0)  # Entfernen Sie die zusätzliche Dimension
image = image / 2 + 0.5  # Denormalisieren Sie das Bild
npimg = image.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.title(f'Bild {image_number}, Predicted Label: {predicted_label}, True Label: {true_label}')
plt.show()

# Initialisieren Sie die Konfusionsmatrix
confusion_matrix = torch.zeros(10, 10)

# Sammeln Sie alle Vorhersagen und wahren Labels
for images, labels in test_loader:
    with torch.no_grad():
        outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    for t, p in zip(labels.view(-1), predicted.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

# Zeigen Sie die Konfusionsmatrix an
plt.figure(figsize=(10, 10))
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# Fügen Sie die Zahlen in die Zellen ein
for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
    plt.text(j, i, format(confusion_matrix[i, j], '.0f'),
             horizontalalignment="center",
             color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
