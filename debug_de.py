import torch
from torchvision import datasets, transforms
from models.target_model import MNISTNet
from attacks.differential_evolution_attack import de_attack

device = 'cpu'
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.ImageFolder('./data/RoadSigns/test', transform=transform)
image, label = test_dataset[0]

model = MNISTNet(in_channels=3, num_classes=4).to(device)
try:
    model.load_state_dict(torch.load('saved_models/base_model.pth', map_location=device))
except Exception as e:
    print(e)
model.eval()

print("Running DE attack...")
res = de_attack(model, image, label, epsilon=0.3, maxiter=2, popsize=2, device=device)
print("Success:", res['success'])
