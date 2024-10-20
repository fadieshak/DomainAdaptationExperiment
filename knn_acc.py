"""
Script used to calculate a KNN model's accuracy. 
The KNN model is fitted on the features extracted from the trained model.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import torch
import data_handler
import model
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch',default=256,type=int,help='Batch size')
parser.add_argument('--benchmark', choices=['0','1'], default='0',help='Choice of benchmark: 0: SVHN->MNIST, 1: MNIST->USPS')
parser.add_argument('--device', choices=['cpu','cuda', 'mps'], default='mps',help='Choice of device: cpu, cuda, mps')
parser.add_argument('--model_path',type=str,help='Path to test model\'s weight')

args = parser.parse_args()

device = args.device
model_path = args.model_path

# Getting the training and test datasets.
if args.benchmark == '0':
    testloader = data_handler.get_dataloader_mnist_test(batch_size=args.batch)
    trainloader = data_handler.get_dataloader_svhn_train(batch_size=args.batch)
else:
    testloader = data_handler.get_dataloader_mnist_test(batch_size=args.batch)
    trainloader = data_handler.get_dataloader_usps_train(batch_size=args.batch)


def extract_features(dataloader, model):
    """
    Extracting feature vectors for a certain dataset.
    """
    features = []
    labels = []
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)
            output = model(images)
            output = output.to('cpu')
            targets = targets.to('cpu')
            features.append(output)
            labels.append(targets)
    features = torch.cat(features, 0)
    labels = torch.cat(labels, 0)
    return features, labels

# Load model and extract features.
encoder = model.LeNet_Encoder()
encoder.load_state_dict(torch.load(model_path))
encoder.to(device)

train_features, train_labels = extract_features(trainloader, encoder)
test_features, test_labels = extract_features(testloader, encoder)

# Convert tensors to numpy arrays for the KNN classifier.
train_features = train_features.numpy()
train_labels = train_labels.numpy()
test_features = test_features.numpy()
test_labels = test_labels.numpy()

# Initialize the KNN classifier.
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data.
knn.fit(train_features, train_labels)

# Predict the labels of the test data.
predicted_labels = knn.predict(test_features)

# Calculate the accuracy of the classifier.
accuracy = accuracy_score(test_labels, predicted_labels)
print(f'Accuracy of KNN classifier on test data: {accuracy * 100:.2f}%')
