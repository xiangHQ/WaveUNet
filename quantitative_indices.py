import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import os
import numpy as np
import scipy.io
import networkx as nx
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import scipy.io
from skimage.metrics import structural_similarity as sk_ssim

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, data_range=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    if data_range is None:
        data_range = (img2.max() - img2.min()).item()

    if data_range < 1e-6:
        data_range = 1.0

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, data_range=None, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, data_range)


def calculate_rmse(predicted, actual):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def calculate_mea(predicted, actual):
    return np.mean(np.abs(predicted - actual))


def calculate_mstd(predicted, actual):
    std_pred = np.std(predicted)
    std_actual = np.std(actual)
    return (std_pred + std_actual) / 2


def calculate_usr(predicted, actual):
    difference = np.abs(predicted - actual)
    success_pixels = np.sum(difference <= np.pi)
    usr = (success_pixels / predicted.size) * 100
    return usr


def find_reference_point_by_coherence(coherence_map, top_n=1):
    """
    Select a reference point based on coherence map

    Parameters:
    - coherence_map: Coherence map, higher values indicate better quality
    - top_n: Select the top N points with highest coherence, then take average position

    Returns:
    - reference_point: Reference point coordinates (row, col)
    """
    flat_coherence = coherence_map.flatten()
    top_indices = np.argpartition(flat_coherence, -top_n)[-top_n:]
    rows, cols = np.unravel_index(top_indices, coherence_map.shape)
    ref_row = int(np.mean(rows))
    ref_col = int(np.mean(cols))
    avg_coherence = np.mean(flat_coherence[top_indices])
    print(f"Selected reference point at ({ref_row}, {ref_col}) with coherence {avg_coherence:.4f}")
    return (ref_row, ref_col)


def apply_reference_point_shift(predicted, actual, reference_point):
    """
    Apply shift alignment using reference point

    Parameters:
    - predicted: Predicted phase map
    - actual: Ground truth phase map
    - reference_point: Reference point coordinates (row, col)

    Returns:
    - shifted: Shifted predicted phase map
    """
    ref_row, ref_col = reference_point
    ref_row = min(max(0, ref_row), predicted.shape[0] - 1)
    ref_col = min(max(0, ref_col), predicted.shape[1] - 1)
    ref_diff = predicted[ref_row, ref_col] - actual[ref_row, ref_col]
    k = np.round(ref_diff / (2 * np.pi))
    shifted = predicted - k * 2 * np.pi
    print(f"Applied {k}*2π shift using reference point ({ref_row}, {ref_col})")
    return shifted


def process_mat_files_in_folder(folder_path, real_data_folder, coherence_folder=None):
    """
    Process .MAT files, select reference point using coherence or image center
    Parameters:
    - folder_path: Path to folder containing prediction data
    - real_data_folder: Path to folder containing ground truth data
    - coherence_folder: Path to folder containing coherence files, if None then don't use coherence
    """
    mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
    total_rmse = 0
    total_usr = 0
    total_ssim = 0
    total_mea = 0
    total_mstd = 0
    count = 0
    results_txt_file = os.path.join(folder_path, 'results_with_shift.txt')
    with open(results_txt_file, 'w') as f:
        f.write("Phase Unwrapping Results with Coherence-based Reference Point\n")
        f.write("=============================================================\n")
        if coherence_folder:
            f.write(f"Coherence folder: {coherence_folder}\n")
        else:
            f.write("Using image center as reference point\n")
        f.write("\n")

    for mat_file in mat_files:
        mat_file_path = os.path.join(folder_path, mat_file)
        real_file_path = os.path.join(real_data_folder, mat_file.replace('_unwrapped.mat', ''))

        if not os.path.exists(real_file_path):
            print(f"Warning: Real data file {real_file_path} not found, skipping {mat_file}")
            continue

        # Load predicted and actual data
        try:
            mat_data = scipy.io.loadmat(mat_file_path)
            real_data = scipy.io.loadmat(real_file_path)
        except Exception as e:
            print(f"Error loading {mat_file}: {e}")
            continue

        first_var_name = [key for key in mat_data.keys() if not key.startswith('__')][0]
        predicted = np.squeeze(mat_data[first_var_name])
        first_var_name = [key for key in real_data.keys() if not key.startswith('__')][0]
        actual = np.squeeze(real_data[first_var_name])

        if predicted.shape != actual.shape:
            print(f"Warning: Shape mismatch for {mat_file}, predicted {predicted.shape}, actual {actual.shape}")
            min_shape = (min(predicted.shape[0], actual.shape[0]),
                         min(predicted.shape[1], actual.shape[1]))
            predicted = predicted[:min_shape[0], :min_shape[1]]
            actual = actual[:min_shape[0], :min_shape[1]]

        # Determine reference point
        if coherence_folder:
            coherence_file_path = os.path.join(coherence_folder, mat_file)
            if os.path.exists(coherence_file_path):
                try:
                    coherence_data = scipy.io.loadmat(coherence_file_path)
                    first_var_name = [key for key in coherence_data.keys() if not key.startswith('__')][0]
                    coherence_map = np.squeeze(coherence_data[first_var_name])
                    if coherence_map.shape == predicted.shape:
                        reference_point = find_reference_point_by_coherence(coherence_map)
                    else:
                        print(
                            f"Warning: Coherence map shape {coherence_map.shape} doesn't match phase map {predicted.shape}")
                        h, w = predicted.shape
                        reference_point = (h // 2, w // 2)
                except Exception as e:
                    print(f"Error loading coherence file {coherence_file_path}: {e}")
                    h, w = predicted.shape
                    reference_point = (h // 2, w // 2)
            else:
                print(f"Warning: Coherence file {coherence_file_path} not found")
                h, w = predicted.shape
                reference_point = (h // 2, w // 2)
        else:
            h, w = predicted.shape
            reference_point = (h // 2, w // 2)

        # Apply reference point shift
        shifted_predicted = apply_reference_point_shift(predicted, actual, reference_point)

        # Compute metrics
        rmse = calculate_rmse(shifted_predicted, actual)
        usr = calculate_usr(shifted_predicted, actual)
        mea = calculate_mea(shifted_predicted, actual)
        mstd = calculate_mstd(shifted_predicted, actual)

        # Convert to PyTorch tensors for SSIM
        y1 = torch.tensor(actual).float()
        x1 = torch.tensor(shifted_predicted).float()
        y1 = y1.unsqueeze(0).unsqueeze(0)
        x1 = x1.unsqueeze(0).unsqueeze(0)

        SSIM = ssim(x1, y1).item()
        # SSIM = sk_ssim(shifted_predicted, actual, data_range=actual.max() - actual.min())

        # Accumulate results
        total_rmse += rmse
        total_usr += usr
        total_ssim += SSIM
        total_mea += mea
        total_mstd += mstd
        count += 1

        # Write per-file results
        with open(results_txt_file, 'a') as f:
            f.write(
                f"File: {mat_file}\n"
                f"  Reference point: {reference_point}\n"
                f"  RMSE: {rmse:.4f}, USR: {usr:.4f}%, "
                f"SSIM: {SSIM:.4f}, MEA: {mea:.4f}, MSTD: {mstd:.4f}\n\n"
            )

        print(f"Processed {mat_file}")
        print(f"  Reference point: {reference_point}")
        print(f"  RMSE: {rmse:.4f}, USR: {usr:.4f}%, SSIM: {SSIM:.4f}, MEA: {mea:.4f}, MSTD: {mstd:.4f}")

    # Compute averages
    if count > 0:
        average_rmse = total_rmse / count
        average_usr = total_usr / count
        average_ssim = total_ssim / count
        average_mea = total_mea / count
        average_mstd = total_mstd / count

        print("\n--- Final Results ---")
        print(f"Average RMSE: {average_rmse:.4f}")
        print(f"Average USR: {average_usr:.4f}%")
        print(f"Average SSIM: {average_ssim:.4f}")
        print(f"Average MEA: {average_mea:.4f}")
        print(f"Average MSTD: {average_mstd:.4f}")

        with open(results_txt_file, 'a') as f:
            f.write("\n--- Final Results ---\n")
            f.write(f"Average RMSE: {average_rmse:.4f}\n")
            f.write(f"Average USR: {average_usr:.4f}%\n")
            f.write(f"Average SSIM: {average_ssim:.4f}\n")
            f.write(f"Average MEA: {average_mea:.4f}\n")
            f.write(f"Average MSTD: {average_mstd:.4f}\n")
    else:
        print("No valid files found to process.")


if __name__ == '__main__':
    folder_path = r"F:\WaveUNet_data\Wave_all_data\train\data\InSAR_Topography\interferogram\wa228"
    real_path = r"F:\WaveUNet_data\Wave_all_data\train\data\InSAR_Topography\true_phase"
    coherence_path = r"F:\WaveUNet_data\Wave_all_data\train\data\InSAR_Topography\coherence"
    process_mat_files_in_folder(folder_path, real_path, coherence_folder=coherence_path)

