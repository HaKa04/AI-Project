import torch
import os
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.functional import F
import itertools
from PIL import Image
from pdf2image import convert_from_path

def convert_png_to_jpg(path):
    jpg_path = path.replace('.png', '.jpg')
    img = Image.open(path)  # Öffnen des PNG-Bildes
    rgb_img = img.convert('RGB')  # Konvertieren des PNG-Bildes in RGB-Format
    rgb_img.save(jpg_path, 'JPEG')  # Speichern des Bildes im JPEG-Format
    os.remove(path)  # Löschen des PNG-Bildes

def convert_tiff_to_jpg(path):
    jpg_path = path.replace('.tiff', '.jpg')  # Ersetzen der .tiff-Erweiterung durch .jpg
    img = Image.open(path)  # Öffnen des TIFF-Bildes
    rgb_img = img.convert('RGB')  # Konvertieren des TIFF-Bildes in RGB-Format
    rgb_img.save(jpg_path, 'JPEG')  # Speichern des Bildes im JPEG-Format
    os.remove(path)  # Löschen des TIFF-Bildes

def convert_pdf_to_jpg(path):
    jpg_path = path.replace('.pdf', '.jpg')  # Ersetzen der .pdf-Erweiterung durch .jpg
    images = convert_from_path(path)  # Konvertieren des PDFs in eine Liste von Bildern
    for i, image in enumerate(images):
        image.save(jpg_path if i == 0 else f"{jpg_path[:-4]}_{i}.jpg", 'JPEG')  # Speichern der Bilder im JPEG-Format
    os.remove(path)  # Löschen des PDFs

# Klasse für das CNN-Modell
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

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

from PIL import Image
import matplotlib.pyplot as plt

# Liste aller Bilder im Ordner 'images'
image_folder = 'images'
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Iterieren Sie über alle Bilder
for image_file in image_files:
    # Laden Sie das Bild
    if image_file.endswith('.png'):
        convert_png_to_jpg(os.path.join(image_folder, image_file))
        image_file = image_file.replace('.png', '.jpg')
    elif image_file.endswith('.tiff'):
        convert_tiff_to_jpg(os.path.join(image_folder, image_file))
        image_file = image_file.replace('.tiff', '.jpg')
    elif image_file.endswith('.pdf'):
        convert_pdf_to_jpg(os.path.join(image_folder, image_file))
        image_file = image_file.replace('.pdf', '.jpg')
    if not image_file.lower().endswith(('.jpg', '.jpeg')):
        print(f"Unsupported file format: {image_file}")
    image_path = os.path.join(image_folder, image_file)
    original_image = Image.open(image_path)

    # Skalieren Sie das Bild auf 32x32
    scaled_image = original_image.resize((32, 32))

    # Wandeln Sie das Bild in ein PyTorch Tensor um und normalisieren Sie es
    image = transform(scaled_image)

    # Fügen Sie eine zusätzliche Dimension hinzu, um eine Batch-Größe von 1 zu erstellen
    image = image.unsqueeze(0)

    # Machen Sie eine Vorhersage mit dem Modell
    model.eval()
    with torch.no_grad():
        output = model(image)
    predicted_class = torch.argmax(output).item()

    
    probabilities = output[0].numpy()
    for i, probability in enumerate(probabilities):
        print(f"Label: {class_names[i]}, Wahrscheinlichkeit: {probability}")


    # Zeigen Sie das skalierte Bild und das vorhergesagte Label an
    plt.figure(figsize=(6, 6))
    plt.imshow(scaled_image)
    plt.title('Scaled Image - Predicted class: ' + class_names[predicted_class])
    plt.show()