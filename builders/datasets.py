import os
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from sklearn.model_selection import train_test_split

def normalize(img):
    img = img/np.pi/2 + 0.5
    return img

def create_dataset(args, mode='train'):

    # Ensure the data root directory exists
    if not os.path.exists(args.dataRootDir):
        raise ValueError(f"Data root directory does not exist: {args.dataRootDir}")

    if mode == 'test':
        interf_dir = os.path.join(args.dataRootDir)
        phase_dir = None  
    else:
        interf_dir = os.path.join(args.dataRootDir, 'wrapped.mat')  
        phase_dir = os.path.join(args.dataRootDir, 'absolute.mat') 

    if not os.path.exists(interf_dir):
        raise ValueError(f"Interferogram directory does not exist: {interf_dir}")
    
    if phase_dir and not os.path.exists(phase_dir):
        raise ValueError(f"Ground truth directory does not exist: {phase_dir}")

    if mode == 'test':
        test_files = [f for f in os.listdir(interf_dir) if f.endswith('.mat')]
        file_list = test_files
    else:
        split_file = os.path.join(args.dataRootDir, 'split.txt')
        if not os.path.exists(split_file):
            print("Split file not found, automatically splitting dataset...")
            all_files = [f for f in os.listdir(interf_dir) if f.endswith('.mat')]
            train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
            with open(split_file, 'w') as f:
                f.write("train:\n")
                for file in train_files:
                    f.write(f"{file}\n")
                f.write("val:\n")
                for file in val_files:
                    f.write(f"{file}\n")
        else:
            # Read the split file
            train_files = []
            val_files = []
            current_section = None

            with open(split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line == 'train:':
                        current_section = 'train'
                    elif line == 'val:':
                        current_section = 'val'
                    elif line and current_section:
                        if current_section == 'train':
                            train_files.append(line)
                        elif current_section == 'val':
                            val_files.append(line)

        # Select file list based on mode
        if mode == 'train':
            file_list = train_files
        elif mode == 'val':
            file_list = val_files

    # Create dataset instance
    dataset = PhaseUnwrappingDataset(
        interf_dir=interf_dir,
        phase_dir=phase_dir,
        file_list=file_list,
        input_size=args.input_size,
        random_mirror=args.random_mirror if mode == 'train' else False,
        is_test=(mode == 'test')
    )

    return dataset


class PhaseUnwrappingDataset(Dataset):
    """Phase Unwrapping Dataset Class"""

    def __init__(self, interf_dir, phase_dir, file_list, input_size, random_mirror=False, is_test=False):
 
        self.interf_dir = interf_dir
        self.phase_dir = phase_dir
        self.file_list = file_list
        self.input_size = input_size
        self.random_mirror = random_mirror
        self.is_test = is_test

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Get filename
        filename = self.file_list[idx]

        # Load interferogram
        interf_path = os.path.join(self.interf_dir, filename)
        interf_data = sio.loadmat(interf_path)
        first_var_name = [key for key in interf_data.keys() if not key.startswith('__')][0]
        interf = interf_data[first_var_name].astype(np.float32)

        if self.is_test:
            # if self.input_size and (interf.shape[0] != self.input_size[0] or interf.shape[1] != self.input_size[1]):
            #     from skimage.transform import resize
            #     interf = resize(interf, self.input_size, order=1, preserve_range=True)
            
            interf = torch.from_numpy(interf.copy()).unsqueeze(0)  #  [1, H, W]
            interf = normalize(interf)
            
            return interf, filename

        phase_path = os.path.join(self.phase_dir, filename)
        phase_data = sio.loadmat(phase_path)
        first_var_name = [key for key in phase_data.keys() if not key.startswith('__')][0]
        phase = phase_data[first_var_name].astype(np.float32)

        if interf.shape != phase.shape:
            raise ValueError(f"Interferogram and ground truth dimensions do not match: {interf.shape} vs {phase.shape}")

        # if self.input_size and (interf.shape[0] != self.input_size[0] or interf.shape[1] != self.input_size[1]):
        #     from skimage.transform import resize
        #     interf = resize(interf, self.input_size, order=1, preserve_range=True)
        #     phase = resize(phase, self.input_size, order=1, preserve_range=True)

        if self.random_mirror:
            if np.random.rand() > 0.5:
                interf = np.fliplr(interf)
                phase = np.fliplr(phase)
            if np.random.rand() > 0.5:
                interf = np.flipud(interf)
                phase = np.flipud(phase)

        interf = torch.from_numpy(interf.copy()).unsqueeze(0)  #  [1, H, W]
        phase = torch.from_numpy(phase.copy()).unsqueeze(0)  #  [1, H, W]
        
        interf = normalize(interf)
        return interf, phase