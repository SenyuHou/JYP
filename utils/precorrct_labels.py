import torch
import numpy as np
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import scipy.stats as stats

def _as_long_tensor(indices, device):
    if isinstance(indices, torch.Tensor):
        return indices.to(device=device, dtype=torch.long).flatten()
    if indices is None:
        return torch.empty(0, dtype=torch.long, device=device)
    if isinstance(indices, (int, np.integer)):
        return torch.tensor([int(indices)], dtype=torch.long, device=device)
    return torch.as_tensor(indices, dtype=torch.long, device=device).flatten()

def knn_cos(query, data, k=50, use_cosine_similarity=False):
    """
    Perform k-Nearest Neighbors using either cosine similarity or Euclidean distance.

    Parameters:
    - query: Query tensor.
    - data: Data tensor.
    - k: Number of neighbors to consider (default is 50).
    - use_cosine_similarity: Whether to use cosine similarity (default is False, uses Euclidean distance).

    Returns:
    - v: Similarity or distance values for the k nearest neighbors.
    - ind: Indices of the k nearest neighbors.
    """
    assert data.shape[1] == query.shape[1]

    if use_cosine_similarity:
        # Normalize feature vectors for cosine similarity calculation
        query_norm = query / query.norm(dim=1)[:, None]
        data_norm = data / data.norm(dim=1)[:, None]
        # Calculate cosine similarity
        sim = torch.mm(query_norm, data_norm.t())
        # Select top k highest similarities
        v, ind = sim.topk(k, largest=True)
    else:
        # Calculate Euclidean distance
        M = torch.cdist(query, data)
        # Select top k smallest distances
        v, ind = M.topk(k, largest=False)

    return v, ind[:, 0:min(k, data.shape[0])].to(torch.long)

def knn_label_distribution_excluding_self(query_embd, y_query, prior_embd, labels, k=50, n_class=10, use_cosine_similarity=True):
    """
    Estimate labels with the empirical label distribution of k nearest neighbors.
    If the query sample is present in the prior set, its self-neighbor is removed.
    """
    n_sample = query_embd.shape[0]
    n_prior = prior_embd.shape[0]
    device = query_embd.device
    labels = labels.to(device)
    y_query = y_query.to(device)

    search_k = min(k + 1, n_prior)
    neighbour_v, neighbour_ind = knn_cos(
        query_embd,
        prior_embd,
        k=search_k,
        use_cosine_similarity=use_cosine_similarity
    )

    label_estimation = torch.zeros((n_sample, n_class), device=device)
    for i in range(n_sample):
        row_values = neighbour_v[i]
        row_indices = neighbour_ind[i]

        if use_cosine_similarity:
            self_pos = torch.nonzero(row_values >= 1.0 - 1e-6, as_tuple=False).flatten()
        else:
            self_pos = torch.nonzero(row_values <= 1e-6, as_tuple=False).flatten()

        if self_pos.numel() > 0:
            keep_mask = torch.ones(row_indices.shape[0], dtype=torch.bool, device=device)
            keep_mask[self_pos[0]] = False
            row_indices = row_indices[keep_mask]

        row_indices = row_indices[:min(k, row_indices.shape[0])]
        if row_indices.numel() == 0:
            label_estimation[i] = F.one_hot(y_query[i], num_classes=n_class).float()
            continue

        neighbour_labels = labels[row_indices]
        label_estimation[i] = F.one_hot(neighbour_labels, num_classes=n_class).float().mean(dim=0)

    return label_estimation

def get_loss_weights(query_embd, y_query, prior_embd, labels, y_label_batch, k=10, n_class=10, use_cosine_similarity=True):
    """
    Compute loss weights based on the frequency of the estimated labels in the nearest neighbors.

    Parameters:
    - query_embd: Embeddings of the query set.
    - y_query: Labels of the query set.
    - prior_embd: Embeddings of the prior set.
    - labels: Labels of the prior set.
    - y_label_batch: Estimated labels for each sample (either from weak or strong estimations).
    - k: Number of nearest neighbors to consider (default is 10).
    - n_class: Number of classes (default is 10).

    Returns:
    - weights: Computed loss weights for each sample.
    """
    n_sample = query_embd.shape[0]
    _, neighbour_ind = knn_cos(query_embd, prior_embd, k=k, use_cosine_similarity=use_cosine_similarity)

    # Compute the labels of the nearest neighbors
    neighbour_label_distribution = labels[neighbour_ind]

    # Append the label of the query (no need to add query label here, as it will be handled by y_label_batch)
    neighbour_label_distribution = torch.cat((neighbour_label_distribution, y_query[:, None]), 1)

    # Use y_label_batch (w or s) as the reference for estimated label
    estimated_labels = torch.argmax(y_label_batch, dim=1)

    # Convert labels to bincount (row wise)
    y_one_hot_batch = F.one_hot(neighbour_label_distribution, num_classes=n_class).float()

    # Compute the frequency of the estimated labels in the neighbour set
    # For each sample, count how many times the estimated label appears in the nearest neighbors
    neighbour_freq = torch.sum(y_one_hot_batch, dim=1)[torch.tensor([range(n_sample)]), estimated_labels]

    # Normalize max count as weight
    weights = neighbour_freq / torch.sum(neighbour_freq)

    # Min-Max normalization
    min_weight = torch.min(weights)
    max_weight = torch.max(weights)
    weights_normalized = (weights - min_weight) / (max_weight - min_weight)  # Perform Min-Max scaling

    return torch.squeeze(weights_normalized)


def sample_labels_in_two_view(fp_embd_w, fp_embd_s, y_noisy, weak_embed, strong_embed, noisy_labels, device='cpu', k=50, n_class=10, use_cosine_similarity=True, to_single_label=True):
    """
    Estimate labels from each augmented view with k-nearest-neighbor label distributions.

    Parameters:
    - fp_embd_w: Feature embeddings for the weakly augmented dataset.
    - fp_embd_s: Feature embeddings for the strongly augmented dataset.
    - y_noisy: Noisy labels.
    - weak_embed: Embeddings for the weakly augmented dataset.
    - strong_embed: Embeddings for the strongly augmented dataset.
    - noisy_labels: Tensor of noisy labels.
    - device: Device to perform computations (default is 'cpu').
    - k: Number of nearest neighbors to consider (default is 50).
    - n_class: Number of classes (default is 10).
    - use_cosine_similarity: Whether to use cosine similarity (default is True).
    - to_single_label: Kept for backward compatibility. This function returns label distributions.

    Returns:
    - y_label_batch_w: kNN label distributions for the weakly augmented dataset.
    - y_label_batch_s: kNN label distributions for the strongly augmented dataset.
    - loss_weights_w: Loss weights for the weakly augmented dataset.
    - loss_weights_s: Loss weights for the strongly augmented dataset.
    """
    y_label_batch_w = knn_label_distribution_excluding_self(
        query_embd=fp_embd_w,
        y_query=y_noisy,
        prior_embd=weak_embed,
        labels=noisy_labels,
        k=k,
        n_class=n_class,
        use_cosine_similarity=use_cosine_similarity
    )

    y_label_batch_s = knn_label_distribution_excluding_self(
        query_embd=fp_embd_s,
        y_query=y_noisy,
        prior_embd=strong_embed,
        labels=noisy_labels,
        k=k,
        n_class=n_class,
        use_cosine_similarity=use_cosine_similarity
    )

    loss_weights_w = get_loss_weights(fp_embd_w, y_noisy, weak_embed, noisy_labels, y_label_batch_w, k=k, n_class=n_class, use_cosine_similarity = use_cosine_similarity)
    loss_weights_s = get_loss_weights(fp_embd_s, y_noisy, strong_embed, noisy_labels, y_label_batch_s, k=k, n_class=n_class, use_cosine_similarity = use_cosine_similarity)

    return y_label_batch_w, y_label_batch_s, loss_weights_w, loss_weights_s


def fit_gmm(historical_diff_w, noisy_labels, n_class, clean_threshold=0.5, noisy_threshold=0.5, hard_threshold=0.9, by_class=True):
    clean_prob = torch.zeros(historical_diff_w.shape, dtype=torch.float32).to(historical_diff_w.device)
    noisy_prob = torch.zeros(historical_diff_w.shape, dtype=torch.float32).to(historical_diff_w.device)
    difficulty = torch.ones(historical_diff_w.shape, dtype=torch.float32).to(historical_diff_w.device)

    # Initialize lists to store sample IDs
    predict_clean_id = []
    predict_noisy_id = []
    predict_hard_id = []

    if by_class:
        # Process each class separately
        for c in range(n_class):
            # Get the indices of the samples belonging to the current class
            class_indices = (noisy_labels == c).nonzero(as_tuple=True)[0]
            
            # Get the data for the current class
            class_data = historical_diff_w[class_indices].unsqueeze(1)  # Shape: (num_samples_in_class, 1)
            class_data_np = class_data.detach().cpu().numpy()

            # Fit the GMM model with 2 components (low and high mean)
            gmm = GaussianMixture(n_components=2)
            gmm.fit(class_data_np)

            # Get the means of the two components
            component_means = gmm.means_.flatten()

            # Calculate CDF values for each component (low and high mean)
            normal_dist_1 = stats.norm(loc=component_means[0], scale=np.sqrt(gmm.covariances_[0, 0]))  # For component 1
            normal_dist_2 = stats.norm(loc=component_means[1], scale=np.sqrt(gmm.covariances_[1, 0]))  # For component 2

            cdf_component_1 = torch.tensor(normal_dist_1.cdf(class_data_np), dtype=torch.float32).to(historical_diff_w.device)
            cdf_component_2 = torch.tensor(normal_dist_2.cdf(class_data_np), dtype=torch.float32).to(historical_diff_w.device)

            # Compare the means to decide which component is "clean" and which is "noisy"
            if component_means[0] < component_means[1]:
                prob_clean = torch.tensor(gmm.predict_proba(class_data_np)[:, 0], dtype=torch.float32).to(historical_diff_w.device)
                prob_noisy = torch.tensor(gmm.predict_proba(class_data_np)[:, 1], dtype=torch.float32).to(historical_diff_w.device)
                tau_clean = prob_clean  # Posterior probability of clean class
                tau_noisy = prob_noisy  # Posterior probability of noisy class
                lambda_clean = cdf_component_1  # CDF of clean component
                lambda_noisy = cdf_component_2  # CDF of noisy component
            else:
                prob_clean = torch.tensor(gmm.predict_proba(class_data_np)[:, 1], dtype=torch.float32).to(historical_diff_w.device)
                prob_noisy = torch.tensor(gmm.predict_proba(class_data_np)[:, 0], dtype=torch.float32).to(historical_diff_w.device)
                tau_clean = prob_clean
                tau_noisy = prob_noisy
                lambda_clean = cdf_component_2
                lambda_noisy = cdf_component_1

            # Assign probabilities to the corresponding samples
            clean_prob[class_indices] = prob_clean
            noisy_prob[class_indices] = prob_noisy

            # Use logical indexing to classify the samples
            clean_mask = clean_prob[class_indices] > clean_threshold
            noisy_mask = noisy_prob[class_indices] > noisy_threshold

            # Directly use tensor operations to append to the lists
            predict_clean_id.extend(class_indices[clean_mask].cpu().tolist())

            # Add noisy samples, excluding those already in clean_id
            noisy_indices = class_indices[noisy_mask]
            clean_id_tensor = _as_long_tensor(predict_clean_id, historical_diff_w.device)
            noisy_indices = noisy_indices[~torch.isin(noisy_indices, clean_id_tensor)]
            predict_noisy_id.extend(noisy_indices.cpu().tolist())

            difficulty[class_indices] = tau_clean * lambda_clean.squeeze() + tau_noisy * (1 - lambda_noisy.squeeze())
    
    else:
        # Process all samples together (without considering class)
        class_data = historical_diff_w.unsqueeze(1)  # Shape: (num_samples, 1)
        class_data_np = class_data.detach().cpu().numpy()

        # Fit the GMM model with 2 components (low and high mean)
        gmm = GaussianMixture(n_components=2)
        gmm.fit(class_data_np)

        # Get the means of the two components
        component_means = gmm.means_.flatten()

        # Calculate CDF values for each component (low and high mean)
        normal_dist_1 = stats.norm(loc=component_means[0], scale=np.sqrt(gmm.covariances_[0, 0]))  # For component 1
        normal_dist_2 = stats.norm(loc=component_means[1], scale=np.sqrt(gmm.covariances_[1, 0]))  # For component 2

        cdf_component_1 = torch.tensor(normal_dist_1.cdf(class_data_np), dtype=torch.float32).to(historical_diff_w.device)
        cdf_component_2 = torch.tensor(normal_dist_2.cdf(class_data_np), dtype=torch.float32).to(historical_diff_w.device)

        # Compare the means to decide which component is "clean" and which is "noisy"
        if component_means[0] < component_means[1]:
            prob_clean = torch.tensor(gmm.predict_proba(class_data_np)[:, 0], dtype=torch.float32).to(historical_diff_w.device)
            prob_noisy = torch.tensor(gmm.predict_proba(class_data_np)[:, 1], dtype=torch.float32).to(historical_diff_w.device)
            tau_clean = prob_clean  # Posterior probability of clean class
            tau_noisy = prob_noisy  # Posterior probability of noisy class
            lambda_clean = cdf_component_1  # CDF of clean component
            lambda_noisy = cdf_component_2  # CDF of noisy component
        else:
            prob_clean = torch.tensor(gmm.predict_proba(class_data_np)[:, 1], dtype=torch.float32).to(historical_diff_w.device)
            prob_noisy = torch.tensor(gmm.predict_proba(class_data_np)[:, 0], dtype=torch.float32).to(historical_diff_w.device)
            tau_clean = prob_clean
            tau_noisy = prob_noisy
            lambda_clean = cdf_component_2
            lambda_noisy = cdf_component_1

        # Assign probabilities to all samples
        clean_prob[:] = prob_clean
        noisy_prob[:] = prob_noisy

        # Use logical masking to classify the samples based on thresholds
        clean_mask = clean_prob > clean_threshold  # Create mask for clean samples
        noisy_mask = noisy_prob > noisy_threshold  # Create mask for noisy samples

        # Use nonzero() to get the indices of the samples satisfying each condition
        clean_indices = torch.nonzero(clean_mask, as_tuple=False).flatten()
        noisy_indices = torch.nonzero(noisy_mask, as_tuple=False).flatten()

        predict_clean_id = clean_indices.cpu().tolist()
        noisy_indices = noisy_indices[~torch.isin(noisy_indices, clean_indices)]
        predict_noisy_id = noisy_indices.cpu().tolist()

        difficulty[:] = tau_clean * lambda_clean.squeeze() + tau_noisy * (1 - lambda_noisy.squeeze())

    return clean_prob, noisy_prob, predict_clean_id, predict_noisy_id, difficulty


def evaluate(predicted_ids, true_indices):
    """
    Calculate precision for predicted indices in comparison with true indices.
    Precision is calculated as the ratio of True Positives (TP) over the total predicted clean samples.
    
    Parameters:
    - predicted_ids: Tensor of predicted sample indices (e.g., clean samples).
    - true_indices: Tensor of true sample indices (e.g., true clean samples).
    
    Returns:
    - precision: The precision of the predicted indices.
    """
    device = predicted_ids.device if isinstance(predicted_ids, torch.Tensor) else (
        true_indices.device if isinstance(true_indices, torch.Tensor) else 'cpu'
    )
    predicted_ids = _as_long_tensor(predicted_ids, device)
    true_indices = _as_long_tensor(true_indices, device)

    # Check if predicted samples are in true indices (True Positives)
    true_positives = torch.isin(predicted_ids, true_indices).sum().item()

    # Calculate precision: TP / (TP + FP)
    precision = true_positives / predicted_ids.numel() * 100 if predicted_ids.numel() > 0 else 0.0
    recall = true_positives / true_indices.numel() * 100 if true_indices.numel() > 0 else 0.0

    return precision, recall
