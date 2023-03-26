import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class SimplePerceptron(nn.Module):
    def __init__(self, input_size, num_authors, hidden_size=None, activation=None):
        super(SimplePerceptron, self).__init__()
        self.activation = activation
        if hidden_size is not None:
            self.hidden = nn.Linear(input_size, hidden_size)
            self.output = nn.Linear(hidden_size, num_authors)
        else:
            self.output = nn.Linear(input_size, num_authors)

    def forward(self, x):
        if hasattr(self, 'hidden'):
            x = self.hidden(x)
            if self.activation == "relu":
                x = nn.functional.relu(x)
            elif self.activation == "tanh":
                x = torch.tanh(x)
        return nn.functional.log_softmax(self.output(x), dim=1)
# Whatever other imports you need

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("--hidden_size", type=int, default=None, help="Size of the hidden layer. If not provided, the model will have no hidden layer.")
    parser.add_argument("--activation", type=str, choices=[None, "relu", "tanh"], default=None, help="Non-linear activation function to use in the hidden layer. Choices: 'relu', 'tanh', or None.")

    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.

    args = parser.parse_args()

    # Load feature data from file
    feature_data = pd.read_csv(args.featurefile)
    print("Reading {}...".format(args.featurefile))

    # Separate features and labels (authors)
    X = feature_data.iloc[:, 2:].values
    y = feature_data.iloc[:, 1].values

    # Encode author names to integers
    unique_authors, author_indices = np.unique(y, return_inverse=True)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, author_indices, test_size=0.2, random_state=42)

    # Convert NumPy arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Create a SimplePerceptron model instance
    input_size = X_train.shape[1]
    num_authors = len(unique_authors)
    model = SimplePerceptron(input_size, num_authors, hidden_size=args.hidden_size, activation=args.activation)


    # Set up the optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    criterion = nn.NLLLoss()

    num_epochs = 500

    # Train the model
    for epoch in range(num_epochs):
        # Forward pass
        output = model(X_train)

        # Calculate loss
        loss = criterion(output, y_train)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()

    # Test the model on test data
    model.eval()
    with torch.no_grad():
        output = model(X_test)
        _, predicted = torch.max(output, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)


    print("Accuracy: {:.2f}%".format(accuracy * 100))

    # Calculate confusion matrix
    conf_mat = confusion_matrix(y_test, predicted)
    print("Confusion matrix:")
    print(conf_mat)
    
