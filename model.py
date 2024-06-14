import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import pickle as pkl
from sklearn.preprocessing import LabelEncoder


class Food101Model:
    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder
        self.transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_tensor):
        self.model.eval()
        logits = self.model(image_tensor)
        predict = int(torch.argmax(F.softmax(logits, dim=1), dim=1))
        predict_classname = self.encoder.inverse_transform([predict])

        return predict_classname


def get_food101_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50()
    model.fc = torch.nn.Linear(2048, 101)
    model.load_state_dict(torch.load(r'./resnet50_weights.pth', map_location=device))
    with open(r'./encoder.pkl', 'rb') as f:
        encoder = pkl.load(f)
    return Food101Model(model, encoder)