import torch
import torch.nn as nn
import torchvision.models as models
import re
from torchvision import transforms
from PIL import Image
import json


with open("./models_places365/categories_places365.txt") as f:
    categories = [line.strip().split()[0].split('/')[-1] for line in f]
cat_2_idx = {cat : i for i, cat in enumerate(categories)}

device = 'cpu'

with open('./finetuned_models/resnet50_20251209_111405/config.json') as f:
    config = json.load(f)

def loadModel(model_name, num_classes=365):
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model_path = "./models_places365/resnet18_places365.pth.tar"
    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model_path = "./models_places365/resnet50_places365.pth.tar"
    elif model_name == 'densenet':
        model = models.densenet161(weights=None)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
        model_path = "./models_places365/densenet161_places365.pth.tar"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        
        # Fix DenseNet naming: convert 'norm.1' -> 'norm1', 'conv.2' -> 'conv2', etc.
        # This handles the old checkpoint format vs new torchvision format
        if model_name == 'densenet':
            name = re.sub(r'\.(\d+)\.', lambda m: m.group(1) + '.', name)
        
        new_state_dict[name] = v

    # Load weights
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print(f"Successfully loaded {model_name} weights")
    except RuntimeError as e:
        print(f"Warning: Could not load with strict=True, trying strict=False")
        print(f"Error: {e}")
        model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()
    return model


def load_finetuned_model(checkpoint_path):
    """
    Load a fine-tuned model from checkpoint
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: 'cuda' or 'cpu'
    
    Returns:
        model: Loaded model in eval mode
        checkpoint: Full checkpoint dict (contains config, class mappings, etc.)

    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config
    config = checkpoint['config']
    model_name = config['model_name']
    dropout_rate = config.get('dropout_rate', 0.5)
    
    # Create model architecture
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 365)  # 365 Places365 classes
        )
    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 365)
        )
    elif model_name == 'densenet':
        model = models.densenet161(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(2208, 365)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    print(f"✓ Loaded {model_name} from epoch {checkpoint['epoch']}")
    print(f"  Val Accuracy: {checkpoint['val_acc']:.2f}%")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    
    return model




img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class Ensemble:
    def __init__(self, fine_tuned_epoch, alpha, threshold=0.3, k=1, device='cpu'):
        self.pretrained = loadModel('resnet50').to(device)
        self.finetuned = load_finetuned_model(checkpoint_path=f'./finetuned_models/resnet50_20251209_111405/best_loss_model_{fine_tuned_epoch}.pth')
        self.finetuned = self.finetuned.to(device)
        self.alpha = alpha
        self.threshold = threshold
        self.k = k
        self.device = device
        print('Model loaded in ', device)
    def __call__(self, image):
        return self.alpha * self.pretrained(image) + (1 - self.alpha) * self.finetuned(image)
    
    def eval(self):
        self.pretrained.eval()
        self.finetuned.eval()

    def predict(self, image):
        logits = self.__call__(image)
        probs = torch.nn.functional.softmax(logits, dim=1)
        p, i = torch.topk(probs, self.k)
        print(p.shape, i.shape, 'u')
        return p.unsqueeze(0), i.unsqueeze(0)