import os
import glob
import time
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.io import savemat, loadmat
from builders.models import creat_model
from builders.datasets import create_dataset
from PIL import Image
import torch.nn.functional as F
from matplotlib import image, pyplot

def load_model_for_testing(model, checkpoint_path, device='cpu'):
    """Load model weights for testing"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    if not isinstance(model, torch.nn.DataParallel) and list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()  
    return model

def save_result(result, save_path, original_filename, output_format='mat'):
    """Save prediction results"""
    os.makedirs(save_path, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    
    if output_format == 'npy':
        np.save(os.path.join(save_path, f"{base_name}.npy"), result)
    elif output_format == 'mat':
        savemat(os.path.join(save_path, f"{base_name}.mat"), {'data': np.squeeze(result)})
        print(f"Saved .mat file to {save_path}")
    elif output_format == 'png':
        # result_normalized = (result - result.min()) / (result.max() - result.min()) * 255
        # Image.fromarray(result_normalized.astype(np.uint8)).save(os.path.join(save_path, f"{base_name}.png"))
        image.imsave(os.path.join(save_path, f"{base_name}.png"), result, pyplot.jet())
        print(f"Saved .png to {save_path} ")
    else:
        raise ValueError(f"不支持的输出格式: {output_format}")

def test_model(args):
    """Test model and save results"""
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = creat_model(args.model)
    model = model.to(device)
    
    # Load trained model weights
    if args.checkpoint:
        print(f"Loading model weights: {args.checkpoint}")
        model = load_model_for_testing(model, args.checkpoint, device)
    else:
        raise ValueError("Checkpoint path must be specified")
    
    test_dataset = create_dataset(args, 'test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved to: {args.output_dir}")
    
    # Testing loop
    with torch.no_grad():
        for i, (data, filenames) in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            
            # Convert output to numpy array
            if isinstance(output, tuple) or isinstance(output, list):
                output = output[0]
            output_np = output.squeeze().cpu().numpy()
            
            original_filename = filenames[0] if isinstance(filenames, list) and len(filenames) > 0 else f"test_{i:04d}"
            save_result(output_np, args.output_dir, original_filename, args.output_format)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} images")
    
    print(f"Testing completed! Results saved to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test model')

    parser.add_argument('--model', type=str, default="WaveUNet", help="model name")
    parser.add_argument('--dataset', type=str, default="InSAR-DLPU", help="dataset")
    parser.add_argument('--dataRootDir', type=str, default=r"./data/InSAR_Funnel_Deformation/interferogram", help="dataset dir")
    parser.add_argument('--input_size', type=str, default="256,256", help="input size")
    parser.add_argument('--num_workers', type=int, default=1, help="the number of parallel threads")
    parser.add_argument('--num_channels', type=int, default=1, help="the number of input channels")

    parser.add_argument('--checkpoint', type=str, default=r"./checkpoint/InSAR_Funnel_Deformation.pth", help="checkpoint")
    parser.add_argument('--output_dir', type=str, default=r"./data/InSAR_Funnel_Deformation/interferogram_WaveUNet", help="output dir")
    parser.add_argument('--output_format', type=str, default='mat', choices=['npy', 'mat', 'png'],
                       help="Output file format")
    parser.add_argument('--cuda', type=bool, default=True, help="run on CPU or GPU")
    
    args = parser.parse_args()
    if isinstance(args.input_size, str):
        args.input_size = tuple(map(int, args.input_size.split(',')))

    print('start')
    start_time = time.time()
    test_model(args)
    time_taken = time.time() - start_time
    print('time: %.4f' % (time_taken))
    print('all over')
