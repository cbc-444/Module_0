import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from transformers import AutoImageProcessor, AutoModelForImageClassification

Model_name="microsoft/resnet-50"
Batch_size=64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained(Model_name)
model = AutoModelForImageClassification.from_pretrained(Model_name)
model.to(device)
model.eval()


mnist_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3)
])
dataset = MNIST(root = "data",train = False, download = True,transform = mnist_transform)

def collate(batch):
    images = [image.convert("RGB") for image,_ in batch]
    labels = torch.tensor([label for _,label in batch])
    return images,labels

loader = DataLoader(dataset,batch_size = Batch_size,shuffle = False,collate_fn=collate)

correct = 0
total = 0

with torch.inference_mode():
    for images,labels in loader:
        inputs = processor(images = images,return_tensors = "pt")
        inputs = {name: value.to(device) for name,value in inputs.items()}
        logits = model(**inputs).logits
        predictions = logits.argmax(dim = -1).cpu()
        correct += (predictions == labels).sum().item()
        total += labels.numel()

accuracy = correct/total
print(f"Accuracy: {accuracy:.3f}")