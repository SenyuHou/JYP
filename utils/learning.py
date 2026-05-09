import random
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import torch.utils.data as data
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def set_random_seed(seed):
    print(f"\n* Set seed {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def adjust_learning_rate(optimizer, epoch, warmup_epochs=40, n_epochs=1000, lr_input=0.001):
    """
    Decay the learning rate with a half-cycle cosine after warmup.

    Parameters:
    - optimizer: The optimizer to adjust the learning rate for.
    - epoch: The current epoch number.
    - warmup_epochs: The number of warmup epochs.
    - n_epochs: The total number of epochs.
    - lr_input: The initial learning rate.

    Returns:
    - lr: The adjusted learning rate.
    """
    if epoch < warmup_epochs:
        lr = lr_input * epoch / warmup_epochs
    else:
        lr = 0.0 + lr_input * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (n_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def cast_label_to_one_hot_and_prototype(y_labels_batch, n_class, return_prototype=False):
    """
    Convert labels to one-hot encoding and optionally return the prototype.

    Parameters:
    - y_labels_batch: A vector of length batch_size.
    - n_class: The number of classes.
    - return_prototype: Whether to return the prototype.

    Returns:
    - y_one_hot_batch: The one-hot encoded labels.
    - y_logits_batch (optional): The prototype logits if return_prototype is True.
    """
    y_one_hot_batch = nn.functional.one_hot(y_labels_batch, num_classes=n_class).float()
    if return_prototype:
        label_min, label_max = [0.001, 0.999]
        y_logits_batch = torch.logit(nn.functional.normalize(
            torch.clip(y_one_hot_batch, min=label_min, max=label_max), p=1.0, dim=1))
        return y_one_hot_batch, y_logits_batch
    else:
        return y_one_hot_batch


def init_fn(worker_id):
    """
    Initialize the random seed for data loader workers.

    Parameters:
    - worker_id: The worker ID.
    """
    np.random.seed(77 + worker_id)

def prepare_2_fp_x(args, fp_encoder, dataset, save_dir=None, device='cpu', fp_dim=768, batch_size=400, seed=None):
    """
    Prepare feature embeddings for weak and strong augmentations.

    Parameters:
    - fp_encoder: The feature extractor.
    - dataset: The dataset to extract features from.
    - save_dir: The directory to save the embeddings.
    - device: The device to perform computation on.
    - fp_dim: The dimension of the feature embeddings.
    - batch_size: The batch size for data loading.

    Returns:
    - fp_embed_all_weak: The weakly augmented feature embeddings.
    - fp_embed_all_strong: The strongly augmented feature embeddings.
    """
    # Set random seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Generate unique file names based on seed
    if save_dir is not None:
        file_weak = save_dir + f'_weak_seed:{seed}.npy' if seed is not None else save_dir + '_weak.npy'
        file_strong = save_dir + f'_strong_seed:{seed}.npy' if seed is not None else save_dir + '_strong.npy'

        # Check if precomputed features already exist
        if os.path.exists(file_weak) and os.path.exists(file_strong):
            fp_embed_all_weak = torch.tensor(np.load(file_weak))
            fp_embed_all_strong = torch.tensor(np.load(file_strong))
            print(f'Embeddings were computed before, loaded from: {save_dir}')
            return fp_embed_all_weak.cpu(), fp_embed_all_strong.cpu()

    # Initialize two sets of feature spaces for weak and strong augmentations
    fp_embed_all_weak = torch.zeros([len(dataset), fp_dim], device=device)
    fp_embed_all_strong = torch.zeros([len(dataset), fp_dim], device=device)
    # Store features and labels for training the classifier
    features = []
    labels = []

    with torch.no_grad():
        data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)
        with tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Computing train embeddings fp(x)', ncols=100) as pbar:
            for i, data_batch in pbar:
                [x_batch_weak, x_batch_strong, y_batch, data_indices] = data_batch[:4]
                if args.fp_encoder == 'ResNet':
                    temp_weak, _ = fp_encoder(x_batch_weak.to(device))
                    temp_strong, _ = fp_encoder(x_batch_strong.to(device))
                else:
                    x_batch_weak = x_batch_weak.to(device)
                    x_batch_strong = x_batch_strong.to(device)
                    temp_weak = fp_encoder(x_batch_weak)
                    temp_strong = fp_encoder(x_batch_strong)
                data_indices = data_indices.to(device)
                fp_embed_all_weak[data_indices] = temp_weak
                fp_embed_all_strong[data_indices] = temp_strong

                # Collect features and labels
                features.append(temp_weak.cpu().numpy())  # Add weak features to the list
                labels.append(y_batch.cpu().numpy())  # Add true labels

        # Concatenate all features and labels
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

    # Save the computed features with unique file names (if save directory is specified)
    if save_dir is not None:
        np.save(file_weak, fp_embed_all_weak.cpu().numpy())
        np.save(file_strong, fp_embed_all_strong.cpu().numpy())

    return fp_embed_all_weak.cpu(), fp_embed_all_strong.cpu()


def prepare_fp_x(args, fp_encoder, dataset, save_dir=None, device='cpu', fp_dim=768, batch_size=400, seed=None):
    """
    Prepare feature embeddings for the dataset.

    Parameters:
    - fp_encoder: The feature extractor.
    - dataset: The dataset to extract features from.
    - save_dir: The directory to save the embeddings.
    - device: The device to perform computation on.
    - fp_dim: The dimension of the feature embeddings.
    - batch_size: The batch size for data loading.

    Returns:
    - fp_embed_all: The feature embeddings.
    """
    if save_dir is not None:
        file_test = save_dir + f'_seed:{seed}.npy' if seed is not None else save_dir + '.npy'
        if os.path.exists(file_test):
            fp_embed_all = torch.tensor(np.load(file_test))
            print(f'Embeddings were computed before, loaded from: {save_dir}')
            return fp_embed_all.cpu()

    with torch.no_grad():
        data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)
        fp_embed_all = torch.zeros([len(dataset), fp_dim]).to(device)
        with tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Computing test embeddings fp(x)', ncols=100) as pbar:
            for i, data_batch in pbar:
                [x_batch, _, data_indices] = data_batch[:3]
                temp = fp_encoder(x_batch.to(device))
                fp_embed_all[data_indices, :] = temp

        if save_dir is not None:
            np.save(file_test, fp_embed_all.cpu())

    return fp_embed_all.cpu()


def cnt_agree(output, target, topk=(1,), softmax=False):
    """
    Compute the accuracy over the k top predictions for the specified values of k.

    - If topk has only one value (e.g., (1,)), return an int
    - If topk has multiple values(e.g., (1,5)), return a dict {k: correct_count}
    """
    maxk = min(max(topk), output.size()[1])

    # softmax for denoising generating
    if softmax:
        output = torch.softmax(-(output - 1) ** 2, dim=-1)

    # top-k predictions
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    # Return a scalar when only one top-k value is requested.
    if len(topk) == 1:
        k = topk[0]
        return correct[:k].reshape(-1).float().sum().item()

    # Return a dict when multiple top-k values are requested.
    results = {}
    for k in topk:
        results[k] = correct[:k].reshape(-1).float().sum().item()

    return results


def adjust_distance_by_t(distance, t_values, T):
    """
    Adjusts the distance values based on the time step t using a scaling factor
    that depends on how close t is to T.

    Parameters:
        distance (Tensor): The distance values to adjust (MSE between prediction and label).
        t_values (Tensor): The time step t values associated with each sample.
        T (float): The maximum time step, representing the end of the diffusion process.
        epsilon (float): A small value to avoid division by zero.

    Returns:
        Tensor: The adjusted distance values.
    """
    # Compute the scaling factor based on the time step t and maximum time step T
    weights = (T - t_values) / T  # Ensure no division by zero

    # Apply the weights to scale the distances
    adjusted_distance = distance * weights
    
    return adjusted_distance

def js_loss(p, q, eps=1e-10):
    """
    Compute the Jensen-Shannon divergence between two probability distributions.
    
    Parameters:
    - p: The first probability distribution (tensor).
    - q: The second probability distribution (tensor).
    - eps: Small value added to avoid log(0) (default 1e-10).
    
    Returns:
    - The Jensen-Shannon divergence between the two distributions.
    """
    # M = 0.5 * (p + q)
    m = 0.5 * (p + q)
    
    # KL divergence between p and m: D_KL(p || m)
    kl_p_m = torch.sum(p * torch.log((p + eps) / (m + eps)), dim=1)
    
    # KL divergence between q and m: D_KL(q || m)
    kl_q_m = torch.sum(q * torch.log((q + eps) / (m + eps)), dim=1)
    
    # JS divergence is the average of the two KL divergences
    js_div = 0.5 * (kl_p_m + kl_q_m)
    
    return js_div.mean()

def gce_loss(p_w, p_s, y_true, q=0.7):
    
    y_true = y_true.argmax(dim=1) 

    p_w_true = p_w.gather(1, y_true.view(-1, 1))  # 
    p_s_true = p_s.gather(1, y_true.view(-1, 1))  # 
    
    loss_w = (1 - p_w_true ** q) / q
    loss_s = (1 - p_s_true ** q) / q
    
    loss = loss_w + loss_s
    return loss.squeeze()  # 


def sharpen_labels(labels, T=0.5):
    """
    Perform label sharpening by increasing the probability differences between classes.
    Args:
        labels (Tensor): Softmax probabilities or label distribution.
        T (float): Temperature for sharpening. A higher T means more sharp.
    Returns:
        Tensor: Sharpened labels.
    """
    # Apply temperature scaling
    labels = labels ** (1.0 / T)
    
    # Normalize to ensure the sum of probabilities is 1 (after scaling)
    labels = F.normalize(labels, p=1, dim=1)
    
    return labels
