import numpy as np
import random
import torch
import torch.nn.functional as F
from scipy import stats
from math import inf
from PIL import Image
from tqdm import tqdm


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))


def multiclass_noisify(y, P, seed=None):
    if seed is not None:
        set_random_seed(seed)

    y = np.array(y.cpu())
    y_noisy = y.copy()
    n = len(y)

    for i in range(n):
        y_noisy[i] = np.random.choice(len(P), p=P[y[i]])

    return y_noisy


def build_symmetric_P(num_classes, noise_ratio):
    assert 0.0 <= noise_ratio <= 1.0
    P = np.ones((num_classes, num_classes)) * (noise_ratio / (num_classes - 1))
    np.fill_diagonal(P, 1.0 - noise_ratio)
    return P


def build_pairflip_P(num_classes, noise_ratio):
    assert 0.0 <= noise_ratio <= 1.0
    P = np.eye(num_classes)
    for i in range(num_classes - 1):
        P[i, i] = 1.0 - noise_ratio
        P[i, i + 1] = noise_ratio
    P[num_classes - 1, num_classes - 1] = 1.0 - noise_ratio
    P[num_classes - 1, 0] = noise_ratio
    return P


def build_asymmetric_P_cifar10(noise_ratio):
    P = np.eye(10)
    n = noise_ratio
    P[9, 9], P[9, 1] = 1. - n, n      # truck -> automobile
    P[2, 2], P[2, 0] = 1. - n, n      # bird -> airplane
    P[3, 3], P[3, 5] = 1. - n, n      # cat -> dog
    P[5, 5], P[5, 3] = 1. - n, n      # dog -> cat
    P[4, 4], P[4, 7] = 1. - n, n      # deer -> horse
    return P


def build_asymmetric_P_cifar100(noise_ratio):
    P = np.eye(100)
    nb_superclasses = 20
    nb_subclasses = 5
    for i in range(nb_superclasses):
        start = i * nb_subclasses
        end = (i + 1) * nb_subclasses
        for j in range(start, end - 1):
            P[j, j] = 1. - noise_ratio
            P[j, j + 1] = noise_ratio
        P[end - 1, end - 1] = 1. - noise_ratio
        P[end - 1, start] = noise_ratio
    return P


def add_noise(targets, noise_ratio, num_classes, noise_type='symmetric', seed=None):
    """noise_type: 'symmetric' | 'asymmetric' | 'pairflip'"""

    if seed is not None:
        set_random_seed(seed)

    if noise_type == 'sym':
        P = build_symmetric_P(num_classes, noise_ratio)
    elif noise_type == 'pair':
        P = build_pairflip_P(num_classes, noise_ratio)
    elif noise_type == 'asym':
        if num_classes == 10:
            P = build_asymmetric_P_cifar10(noise_ratio)
        elif num_classes == 100:
            P = build_asymmetric_P_cifar100(noise_ratio)
        else:
            raise ValueError(f"Asymmetric noise for num_classes={num_classes} is not supported.")
    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")

    noisy_targets = multiclass_noisify(targets, P, seed)
    return noisy_targets


def generate_instance_dependent_noise(train_labels, noise_rate=0.2, topk=3, seed=None, pool_multiplier=1.5):
    if seed is not None:
        np.random.seed(seed)

    label = np.array(train_labels)
    num_classes = len(np.unique(label))

    if num_classes == 10:
        softmax_path = './noise_label_saved/noise_label_gen_idn/cifar10/softmax_out_avg.npy'
    elif num_classes == 100:
        softmax_path = './noise_label_saved/noise_label_gen_idn/cifar100/softmax_out_avg.npy'
    else:
        raise ValueError(f"Unsupported number of classes: {num_classes}. Only 10 or 100 are supported.")

    softmax_out_avg = np.load(softmax_path)

    label_noisy_cand = np.full_like(label, fill_value=-1)
    label_noisy_prob = np.full_like(label, fill_value=-1.0, dtype=np.float32)

    for i in range(len(label)):
        pred = softmax_out_avg[i].copy()
        pred[label[i]] = -1
        topk_indices = pred.argsort()[-topk:]
        label_noisy_cand[i] = np.random.choice(topk_indices)
        label_noisy_prob[i] = np.max(pred)

    label_noisy = label.copy()
    num_noisy = int(noise_rate * len(label))
    pool_size = min(int(pool_multiplier * num_noisy), len(label))
    candidate_pool = np.argsort(label_noisy_prob)[-pool_size:]
    noisy_indices = np.random.choice(candidate_pool, size=num_noisy, replace=False)
    label_noisy[noisy_indices] = label_noisy_cand[noisy_indices]

    return label_noisy


def generate_instance_noise_labels(
    data,
    targets,
    transform,
    num_classes,
    tau=0.2,
    std=0.1,
    feature_size=3 * 32 * 32,
    seed=42,
    device='cuda'
):
    num_samples = len(targets)
    min_target = min(targets)
    max_target = max(targets)

    if seed is not None:
        set_random_seed(seed)

    flip_distribution = stats.truncnorm((0 - tau) / std, (1 - tau) / std, loc=tau, scale=std)
    q = flip_distribution.rvs(num_samples)

    W = torch.tensor(np.random.randn(num_classes, feature_size, num_classes)).float().to(device)

    P = []
    for i in tqdm(range(num_samples), desc="Generating instance noise"):
        x = transform(Image.fromarray(data[i])).to(device)
        y = targets[i]

        p = x.reshape(1, -1).mm(W[y]).squeeze(0)
        p[y] = -inf
        p = q[i] * F.softmax(p, dim=0)
        p[y] += 1 - q[i]
        P.append(p)

    P = torch.stack(P, 0).cpu().numpy()
    label_space = [i for i in range(min_target, max_target + 1)]
    new_labels = [np.random.choice(label_space, p=P[i]) for i in range(num_samples)]

    return new_labels
