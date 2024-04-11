from dataset import BirdSoundDataset
from torch.utils.data import DataLoader,Dataset
import torchaudio
from torchvision.transforms import ToTensor
import torch

class AudioImageDataset(Dataset):
    def __init__(self,audiodataset:BirdSoundDataset,image_transormations=None,device='cpu'):
        self.audiodataset = audiodataset
        self.transormations = image_transormations if image_transormations else None
        self.device = device
    def __len__(self):
        return len(self.audiodataset)
    
    def __getitem__(self,idx):
        spectrogram, label = self.audiodataset[idx]

        # Convert to decibel (dB) scale
        spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)

        # Normalize to range [0, 1]
        spectrogram_db_normalized = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min())

        # Convert to numpy array
        spectrogram_db_normalized_np = spectrogram_db_normalized.cpu().numpy()
#         spectrogram_image = Image.fromarray((spectrogram_db_normalized_np[0] * 255).astype('uint8'))

        # Convert spectrogram_image to tensor
        t1 = ToTensor()
        image = torch.tensor(spectrogram_db_normalized_np)
#         print(image)
        # Repeat the single channel image along the channel dimension to create an RGB image
        image = torch.cat((image, image, image), dim=0)

        # Apply transformations if needed
        if self.transormations:
            image = self.transormations(image)

        return image, label
