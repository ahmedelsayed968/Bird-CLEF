from typing import List
import audiomentations
import torch
from torch import nn
from torch.utils.data import DataLoader
from audiomentations import AddGaussianNoise,TimeStretch,Compose,PitchShift
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torchaudio
import torch 
from torch import nn
from torch.utils.data import DataLoader,Dataset
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import timm 
import librosa
import os
import pandas as pd
import numpy as np

class BirdSoundDataset(Dataset):
    
    def __init__(self,
                 base_dir,
                 annotation_dir,
                 val_dir,
                 device,
                 split='train',
                 transormations=None,
                 target_sampling_rate=32000*2,
                 num_samples=32000,
                with_augmentation=True):
        self.BASE_DIR = base_dir
        self.ANNOTATIONS_DIR  = annotation_dir
        self.split = split
        self.with_augmentation = with_augmentation
#         self.annotations = pd.read_csv(os.path.join(self.ANNOTATIONS_DIR,self.split+'.csv'))
        self.VAL_DIR = val_dir
        self._get_annotations()
        self.device=device
        self.transormations = None if not transormations else transormations.to(self.device)
        self.target_sampling_rate = target_sampling_rate
        self.num_samples = num_samples
        self.labels:List[str] = list(self.annotations['primary_label'].unique())
        self.resampler= torchaudio.transforms.Resample(
                                      self.target_sampling_rate,
                                    ).to(self.device)

        self.augmenter = Compose([
                                audiomentations.AddGaussianNoise(p=0.5),
                                audiomentations.TimeStretch(p=1),
                                audiomentations.PitchShift(p=0.6),
                                #     audiomentations.Shift(p=0.5),
                                #     audiomentations.Normalize(p=0.5)
                            ])
        self.c = len(self.labels)
        
    def __len__(self):
        return self.annotations.shape[0]
    def _get_annotations(self):
        if self.split == 'train':
            if self.with_augmentation:
                path_to_load = os.path.join(self.ANNOTATIONS_DIR,self.split+'-aug.csv')
        elif self.split == 'val':
            path_to_load = os.path.join(self.VAL_DIR ,self.split+'.csv')
        else:
                path_to_load = os.path.join(self.ANNOTATIONS_DIR,self.split+'.csv')
        self.annotations = pd.read_csv(path_to_load)


    def __getitem__(self,idx):
        "get the sound file and the corresponding label"
        sound = self._get_sound(idx)
        label = self._get_label(idx)
        return sound, torch.tensor(self.labels.index(label))
    
    def _get_sound(self,idx):
        sound_file_name = self.annotations.iloc[idx,-2]
        # use torchaudio to load it from the harddisk
        sound_path = os.path.join(self.BASE_DIR,
                                 sound_file_name)
        sound,sample_rate = torchaudio.load(sound_path)
#         sound,sample_rate = self._spectral_subtraction(sound_path)
        
#         sound = sound[np.newaxis,:]
#         sound = torch.tensor(sound)
        # Do aumgnetation before sending the audio to GPU
        sound = sound.to(self.device)
        # CPU Operation
#         print(sound.shape)

#         if np.random.random() >= 0.5 and self.with_augmentation and self.split =='train':
#             #             sound = self.augmenter(sound,sample_rate)
#             sound = self._apply_augmentations(sound,sample_rate)
#             #         sound = sound.reshape(-1,1)
#             #         sound = torch.tensor(sound)

#         sound = sound.abs()
        #Preprocessing Sound
        sound = self._mix_down_if_necessary(sound) #CPU
        sound = self._cut_if_necessary(sound) # CPU
        sound = self._pad_if_necessary(sound) # CPU
        
#         sound = sound.to(self.device)
        sound = self._resample_if_necessary(sound,sample_rate) #GPU
        if self.transormations:
            sound = self.transormations(sound) #GPU

        return sound
    def _apply_augmentations(self,waveform,sample_rate):
        # Apply random scaling (time stretch)
        import random
        scale_factor = random.uniform(0.8, 1.21)
        t1 = torchaudio.transforms.TimeStretch(n_freq=1).to(self.device)
#         print(waveform.shape)
        waveform = t1(waveform,1/scale_factor)

#         # Apply pitch shift
#         pitch_shift = random.uniform(-4, 4)  # Shift by -4 to +4 semitones
#         t2 = torchaudio.transforms.Vol(-pitch_shift,).to(self.device)
#         waveform = t2(waveform)
        del t1
        try:
            torch.cuda.empty_cache()
        except:
            pass
        return waveform
    def _cut_if_necessary(self,sound):
        if sound.shape[1] > self.num_samples:
            sound = sound[:,:self.num_samples]
        return sound
    def _pad_if_necessary(self,sound):
        _len = sound.shape[1]
        if _len < self.num_samples:
            diff = self.num_samples - _len
            sound = torch.nn.functional.pad(sound,pad=(0,diff))
        return sound
    
    def _resample_if_necessary(self,sound,sample_rate):
        if sample_rate == self.target_sampling_rate:
            return sound
        self.resampler.orig_freq = sample_rate
        return self.resampler(sound)
        
    def _mix_down_if_necessary(self,sound):
        if sound.shape[0]==1:
            return sound
        return torch.mean(sound,dim=0,keepdim=True)
    
    def _spectral_subtraction(self,file_name):
        output_file = os.path.basename(file_name)
        # read_audio file
        sound,sample_rate= librosa.load(file_name,sr=None)
        # Do STFT
        freq_domain = np.abs(librosa.stft(sound))
        # Avg Noise
        noise_spc   = np.mean(freq_domain,axis=1,keepdims=True)
        beta = 20
        clean_spectram = np.maximum(freq_domain-beta*noise_spc,0)
        istft_singal = librosa.istft(clean_spectram)
    #     librosa.output.write_ogg(file_name)
        return istft_singal,sample_rate
    
    def _get_label(self,idx):
        return self.annotations.iloc[idx,-1]
