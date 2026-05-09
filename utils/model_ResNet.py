import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class CustomResNetEncoder(nn.Module):
    """
    Custom ResNet encoder class with a projection head for feature extraction and dimensionality reduction.
    """
    def __init__(self, base_model='resnet50', num_class=10):
        """
        Initialize the CustomResNetEncoder with the specified feature dimension and base model.

        Parameters:
        - base_model: The base ResNet model to use ('resnet34' or 'resnet50').
        """
        super(CustomResNetEncoder, self).__init__()

        # Load the appropriate pre-trained ResNet model
        if base_model == 'resnet34':
            base = models.resnet34(pretrained=True)
            out_features = base.fc.in_features
        elif base_model == 'resnet50':
            base = models.resnet50(pretrained=True)
            out_features = base.fc.in_features
        else:
            raise ValueError("Invalid base model. Choose 'resnet34' or 'resnet50'.")

        # Remove the fully connected layer of the ResNet model (i.e., keep feature extraction layers)
        self.f = nn.Sequential(*(list(base.children())[:-1]))

        # Add a projection head
        self.g = nn.Linear(out_features, num_class, bias=False)

        self.feature_dim = out_features

    def forward_feature(self, x):
        """
        Forward pass through the ResNet encoder (feature extraction part) without projection head.

        Parameters:
        - x: Input tensor with shape [batch_size, channels, height, width].

        Returns:
        - Extracted feature with shape [batch_size, out_features].
        """
        # Forward pass through the feature extraction layers
        x = self.f(x)  # x shape will be [batch_size, out_features, 1, 1]
        x = torch.flatten(x, 1)  # Flatten to [batch_size, out_features]
        return x

    def forward(self, x, return_feature=True):
        """
        Forward pass through the complete model (feature extraction + projection head).

        Parameters:
        - x: Input tensor with shape [batch_size, channels, height, width].
        - return_feature: If True, return the feature vector before projection head.

        Returns:
        - Projected feature output with shape [batch_size, feature_dim].
        """
        # Extract features using the feature extraction layers
        feature = self.forward_feature(x)

        # Forward pass through the projection head
        out = self.g(feature)

        if return_feature:
            return feature, out
        else:
            return out


# Assuming you have a function to train the model
def pretrain_resnet(fp_encoder, train_dataset, test_dataset, device, num_epochs=15, batch_size=256, learning_rate=1e-3):
    """
    Pre-train the ResNet model on the given dataset for the specified number of epochs.

    Parameters:
    - fp_encoder: The feature encoder (ResNet).
    - train_dataset: The training dataset.
    - test_dataset: The test dataset.
    - device: The device (CPU or GPU).
    - num_epochs: Number of epochs for pre-training (default 15).
    - batch_size: Batch size for training (default 256).
    - learning_rate: Learning rate for the optimizer (default 1e-3).

    Returns:
    - The trained ResNet model.
    """
    # DataLoader for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss expects raw logits as output
    optimizer = optim.Adam(fp_encoder.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training loop
    fp_encoder = fp_encoder.to(device)
    fp_encoder.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(train_loader, total=len(train_loader), desc=f"Pre-training Epoch {epoch+1}/{num_epochs}", ncols=100) as pbar:  
            for inputs, labels, idx in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass through the model
                feature, outputs = fp_encoder(inputs)  # Get logits (raw outputs before softmax)
                
                # Calculate the loss (CrossEntropyLoss expects logits)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                pbar.set_postfix(loss=running_loss/len(pbar))  # Display loss in progress bar

        # Step the learning rate scheduler
        scheduler.step()

    # Evaluate the model on the test set
    fp_encoder.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels, idx in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            feature, outputs = fp_encoder(inputs)  # Get logits (raw outputs before softmax)

            # Convert logits to predicted class by choosing the class with max logit value
            _, predicted = torch.max(outputs, 1)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Calculate accuracy
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save the trained model
    model_save_path = './pretrained/pretrained_resnet_model.pth'
    torch.save(fp_encoder.state_dict(), model_save_path)
    print(f"Pretraining Down!  Model saved to {model_save_path}")
    
    return fp_encoder