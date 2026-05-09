import torch
import numpy as np
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
from sklearn.cluster import KMeans

def cluster_and_reassign_labels(query_embd, y_query, k_clusters=10):

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k_clusters, init='k-means++', n_init=10)
    cluster_indices = kmeans.fit_predict(query_embd.cpu().numpy())  # [B,]
    cluster_indices = torch.tensor(cluster_indices, dtype=torch.long, device=query_embd.device)
    cluster_centers = torch.tensor(kmeans.cluster_centers_).to(query_embd.device)  # [k, feature_dim]
    
    # Reassign labels based on most frequent label in each cluster
    cluster_labels = torch.zeros(query_embd.shape[0], dtype=torch.long, device=query_embd.device)
    
    for cluster_id in range(k_clusters):
        # Get indices of samples in this cluster
        cluster_samples = torch.nonzero(cluster_indices == cluster_id).squeeze(1)       
        # Get the labels of these samples
        cluster_sample_labels = y_query[cluster_samples]     
        # Find the most frequent label in this cluster
        most_frequent_label = torch.mode(cluster_sample_labels)[0]  # Most frequent label       
        # Assign the most frequent label to the current cluster
        cluster_labels[cluster_id] = most_frequent_label
    
    return cluster_labels, cluster_centers

def update_soft_labels_with_cluster_centers(query_embd, cluster_centers, cluster_labels, n_class=10, return_hard_labels=False):

    # Calculate distances between query samples and cluster centers
    distances = torch.cdist(query_embd, cluster_centers)  # [B, k_clusters]
    
    # Convert distances to similarity (smaller distance means higher similarity)
    similarity = 1.0 / (distances + 1e-6)  # Avoid division by zero
    similarity_normalized = similarity / similarity.sum(dim=1, keepdim=True)  # Normalize to get probabilities

    soft_labels = torch.zeros(query_embd.shape[0], n_class, device=query_embd.device)

    for i in range(query_embd.shape[0]):
        for cluster_id in range(cluster_centers.shape[0]):
            cluster_label = cluster_labels[cluster_id]
            # Assign the label probability based on the similarity to the cluster center
            soft_labels[i, cluster_label] += similarity_normalized[i, cluster_id]

    _, hard_labels_in_cluster_id = torch.min(distances, dim=1)
    hard_labels = cluster_labels[hard_labels_in_cluster_id]
    
    if return_hard_labels:
        return hard_labels
    else:
        return soft_labels

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

def label_distribution(query_embd, y_query, prior_embd, labels, k=50, n_class=10, weighted=True, use_cosine_similarity=True):
    """
    Compute the label distribution for the query set based on the nearest neighbors in the prior set.

    Parameters:
    - query_embd: Embeddings of the query set.
    - y_query: Labels of the query set.
    - prior_embd: Embeddings of the prior set.
    - labels: Labels of the prior set.
    - k: Number of nearest neighbors to consider (default is 50).
    - n_class: Number of classes (default is 10).
    - weighted: Whether to use weighted averaging (default is True).
    - use_cosine_similarity: Whether to use cosine similarity (default is True).

    Returns:
    - max_prob_label: Labels with the highest probability for each query sample.
    - neighbour_label_distribution: Label distribution for each query sample.
    """
    n_sample = query_embd.shape[0]
    device = query_embd.device

    # Get nearest neighbor indices and weights
    neighbour_v, neighbour_ind = knn_cos(query_embd, prior_embd, k=k, use_cosine_similarity=use_cosine_similarity)

    # Initialize label distribution
    neighbour_label_distribution = torch.zeros((n_sample, n_class), device=device)

    # Get neighbor labels
    neighbour_labels = labels[neighbour_ind]

    if weighted:
        # Compute weights, smaller distance gets larger weight
        weights = 1.0 / (neighbour_v + 1e-6)  # Add small value to avoid division by zero
        weights_normalized = weights / weights.sum(dim=1, keepdim=True)  # Normalize weights

        # Convert labels to one-hot encoding
        labels_one_hot = F.one_hot(neighbour_labels, num_classes=n_class).float()  # [n_sample, k, n_class]

        # Update label distribution using weights
        neighbour_label_distribution = torch.sum(labels_one_hot * weights_normalized.unsqueeze(2), dim=1)

    else:
        # For unweighted case, still use one-hot encoding but calculate mean
        labels_one_hot = F.one_hot(neighbour_labels, num_classes=n_class).float()  # [n_sample, k, n_class]
        neighbour_label_distribution = labels_one_hot.mean(dim=1)  # Calculate mean instead of weighted mean

    # Find the labels with the highest probability in the label distribution
    _, max_prob_label = torch.max(neighbour_label_distribution, dim=1)

    return max_prob_label, neighbour_label_distribution

def KL_label_distribution(neighbour_label_distribution_w, neighbour_label_distribution_s):
    """
    Compute the KL divergence between two label distributions.

    Parameters:
    - neighbour_label_distribution_w: Label distribution for the weakly augmented data.
    - neighbour_label_distribution_s: Label distribution for the strongly augmented data.

    Returns:
    - kl_div_per_sample: KL divergence per sample.
    """
    # Ensure the distributions are properly normalized
    distribution_w = F.softmax(neighbour_label_distribution_w, dim=1)
    distribution_s = F.softmax(neighbour_label_distribution_s, dim=1)

    # Compute KL divergence per sample
    kl_div = F.kl_div(distribution_w.log(), distribution_s, reduction='none')

    # Sum over the class dimension to get KL divergence per sample
    kl_div_per_sample = kl_div.sum(dim=1)

    return kl_div_per_sample

def gmm_binary_split(kl_div_values, n_components=2, random_state=0):
    """
    Split the samples into two sets using a Gaussian Mixture Model based on KL divergence values.

    Parameters:
    - kl_div_values: KL divergence values.
    - n_components: Number of Gaussian components (default is 2).
    - random_state: Random state for reproducibility (default is 0).

    Returns:
    - lower_set_batch: Indices of samples belonging to the lower set.
    - higher_set_batch: Indices of samples belonging to the higher set.
    """
    # Ensure kl_div_values is a NumPy array
    if isinstance(kl_div_values, torch.Tensor):
        kl_div_values = kl_div_values.numpy()

    # Reshape for GMM
    kl_div_values = kl_div_values.reshape(-1, 1)

    # Train GMM
    gmm = GaussianMixture(n_components=n_components, random_state=random_state).fit(kl_div_values)

    # Get means of the two components and determine the lower mean
    means = gmm.means_.flatten()
    lower_mean_idx = np.argmin(means)

    # Predict component labels for each KL divergence value
    component_labels = gmm.predict(kl_div_values)

    # Assign samples to lower or higher set based on component labels
    lower_set_batch = np.where(component_labels == lower_mean_idx)[0]
    higher_set_batch = np.where(component_labels != lower_mean_idx)[0]

    return lower_set_batch, higher_set_batch

def sample_labels(neighbour_label_distribution, y_query, max_prob_label, lower_set_batch, higher_set_batch, to_single_label=False):
    """
    Sample labels based on the neighbor label distribution and the lower and higher sets.

    Parameters:
    - neighbour_label_distribution: Label distribution for each sample.
    - y_query: Original noisy labels.
    - max_prob_label: Labels with the highest probability for each sample.
    - lower_set_batch: Indices of samples in the lower set.
    - higher_set_batch: Indices of samples in the higher set.
    - to_single_label: Whether to convert the output labels to single integer labels (default is False).

    Returns:
    - y_label_batch: Sampled labels.
    """
    y_label_batch = torch.zeros_like(neighbour_label_distribution)

    # For samples in higher set, retain the probability distribution as labels
    y_label_batch[higher_set_batch] = neighbour_label_distribution[higher_set_batch]

    # For samples in lower set, check if the original label matches the max probability label
    for idx in lower_set_batch:
        if y_query[idx] == max_prob_label[idx]:
            # If original label matches max probability label, retain original label in one-hot format
            y_label_batch[idx] = F.one_hot(y_query[idx], num_classes=neighbour_label_distribution.shape[1]).float()
        else:
            # Otherwise, use the max probability label in one-hot format
            y_label_batch[idx] = F.one_hot(max_prob_label[idx], num_classes=neighbour_label_distribution.shape[1]).float()

    if to_single_label:
        # If to_single_label is True, convert to single integer labels
        y_label_batch = torch.argmax(y_label_batch, dim=1)

    return y_label_batch

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


def get_loss_weights_random_sample(query_embd, y_query, prior_embd, labels, k=10, n_class=10):
    """
    Compute loss weights based on the frequency of the sampled labels in the nearest neighbors.

    Parameters:
    - query_embd: Embeddings of the query set.
    - y_query: Labels of the query set.
    - prior_embd: Embeddings of the prior set.
    - labels: Labels of the prior set.
    - k: Number of nearest neighbors to consider (default is 10).
    - n_class: Number of classes (default is 10).

    Returns:
    - weights: Computed loss weights for each sample.
    """
    n_sample = query_embd.shape[0]
    _, neighbour_ind = knn_cos(query_embd, prior_embd, k=k, use_cosine_similarity=False)

    # Compute the labels of the nearest neighbors
    neighbour_label_distribution = labels[neighbour_ind]

    # Append the label of the query
    neighbour_label_distribution = torch.cat((neighbour_label_distribution, y_query[:, None]), 1)

    # Sample a label from the k+1 labels (k neighbors and itself)
    sampled_labels = neighbour_label_distribution[torch.arange(n_sample), torch.randint(0, k+1, (n_sample,))]

    # Convert labels to bincount (row wise)
    y_one_hot_batch = F.one_hot(neighbour_label_distribution, num_classes=n_class).float()

    # Compute the frequency of the sampled labels
    neighbour_freq = torch.sum(y_one_hot_batch, dim=1)[torch.tensor([range(n_sample)]), sampled_labels]

    # Normalize max count as weight
    weights = neighbour_freq / torch.sum(neighbour_freq)

    return torch.squeeze(weights)

def sample_labels_in_two_view(fp_embd_w, fp_embd_s, y_noisy, weak_embed, strong_embed, noisy_labels, device='cpu', k=50, n_class=10, use_cosine_similarity=True, to_single_label=True):
    """
    Compute the label distribution for noisy datasets and perform KL divergence calculation, GMM binary split, and label sampling.

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
    - to_single_label: Whether to convert the output labels to single integer labels (default is True).

    Returns:
    - y_label_batch_w: Sampled labels for the weakly augmented dataset.
    - y_label_batch_s: Sampled labels for the strongly augmented dataset.
    - loss_weights_w: Loss weights for the weakly augmented dataset.
    - loss_weights_s: Loss weights for the strongly augmented dataset.
    """
    # Compute the label distribution for the noisy datasets
    max_prob_label_w, neighbour_label_distribution_w = label_distribution(
        query_embd=fp_embd_w,
        y_query=y_noisy,
        prior_embd=weak_embed,
        labels=noisy_labels,
        k=k,
        n_class=n_class,
        weighted=True,
        use_cosine_similarity=use_cosine_similarity
    )

    max_prob_label_s, neighbour_label_distribution_s = label_distribution(
        query_embd=fp_embd_s,
        y_query=y_noisy,
        prior_embd=strong_embed,
        labels=noisy_labels,
        k=k,
        n_class=n_class,
        weighted=True,
        use_cosine_similarity=use_cosine_similarity
    )

    kl_div = KL_label_distribution(neighbour_label_distribution_w.cpu(), neighbour_label_distribution_s.cpu())

    lower_set_batch, higher_set_batch = gmm_binary_split(kl_div)

    y_label_batch_w = sample_labels(
        neighbour_label_distribution_w,
        y_noisy,
        max_prob_label_w,
        lower_set_batch,
        higher_set_batch,
        to_single_label=to_single_label
    )

    y_label_batch_s = sample_labels(
        neighbour_label_distribution_s,
        y_noisy,
        max_prob_label_s,
        lower_set_batch,
        higher_set_batch,
        to_single_label=to_single_label
    )

    loss_weights_w = get_loss_weights(fp_embd_w, y_noisy, weak_embed, noisy_labels, y_label_batch_w, k=k, n_class=n_class, use_cosine_similarity = use_cosine_similarity)
    loss_weights_s = get_loss_weights(fp_embd_s, y_noisy, strong_embed, noisy_labels, y_label_batch_s, k=k, n_class=n_class, use_cosine_similarity = use_cosine_similarity)

    return y_label_batch_w, y_label_batch_s, loss_weights_w, loss_weights_s


def fit_gmm_w_hard(historical_diff_w, noisy_labels, n_class, clean_threshold=0.5, noisy_threshold=0.5, hard_threshold=0.9, by_class=True):
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
            noisy_indices = noisy_indices[~torch.isin(noisy_indices, torch.tensor(predict_clean_id).to(historical_diff_w.device))]
            predict_noisy_id.extend(noisy_indices.cpu().tolist())


            # Now, based on noisy_threshold_higher, move some noisy samples to hard
            noisy_prob_values = noisy_prob[class_indices]
            noisy_prob_values_for_class = noisy_prob_values[noisy_mask]
            hard_mask = noisy_prob_values_for_class < hard_threshold

            # Only add to hard ID the noisy samples that are below the higher threshold
            hard_samples = noisy_indices[hard_mask]
            predict_hard_id.extend(hard_samples.cpu().tolist())

            # Remove hard samples from predict_noisy_id to prevent duplicates
            noisy_indices = noisy_indices[~torch.isin(noisy_indices, hard_samples)]
            # Convert predict_noisy_id to tensor
            predict_noisy_id_tensor = torch.tensor(predict_noisy_id, dtype=torch.long).to(historical_diff_w.device)

            # Convert hard_samples to tensor using clone().detach() to avoid the warning
            hard_samples_tensor = hard_samples.clone().detach().to(historical_diff_w.device)

            # Use torch.isin() to filter out the hard samples from predict_noisy_id
            predict_noisy_id_tensor = predict_noisy_id_tensor[~torch.isin(predict_noisy_id_tensor, hard_samples_tensor)]

            # Convert back to list and move to CPU
            predict_noisy_id = predict_noisy_id_tensor.cpu().tolist()

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
        hard_mask = ~(clean_mask | noisy_mask)     # Hard samples are those that don't belong to clean or noisy

        # Use nonzero() to get the indices of the samples satisfying each condition
        predict_clean_id = torch.nonzero(clean_mask).cpu().squeeze().tolist()
        mask_no_clean = ~torch.isin(torch.nonzero(noisy_mask).squeeze(), predict_clean_id)
        predict_noisy_id = torch.nonzero(mask_no_clean).cpu().squeeze().tolist()
        predict_hard_id = torch.nonzero(hard_mask).cpu().squeeze().tolist()

        difficulty[:] = tau_clean * lambda_clean.squeeze() + tau_noisy * (1 - lambda_noisy.squeeze())

    return clean_prob, noisy_prob, predict_clean_id, predict_noisy_id, predict_hard_id, difficulty

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
            noisy_indices = noisy_indices[~torch.isin(noisy_indices, torch.tensor(predict_clean_id).to(historical_diff_w.device))]
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
        predict_clean_id = torch.nonzero(clean_mask).cpu().squeeze().tolist()
        mask_no_clean = ~torch.isin(torch.nonzero(noisy_mask).squeeze(), predict_clean_id)
        predict_noisy_id = torch.nonzero(mask_no_clean).cpu().squeeze().tolist()

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
    # Check if predicted samples are in true indices (True Positives)
    true_positives = torch.isin(predicted_ids, true_indices).sum().item()

    # Calculate precision: TP / (TP + FP)
    precision = true_positives / len(predicted_ids) * 100 if len(predicted_ids) > 0 else 0.0
    recall = true_positives / len(true_indices) * 100 if len(true_indices) > 0 else 0.0

    return precision, recall