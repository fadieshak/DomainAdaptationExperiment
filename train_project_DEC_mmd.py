"""
Training script for class-cluster projection 
using the DEC model as teacher and MMD loss.

This script is heavily modified from https://github.com/ayushtues/ADDA_pytorch/tree/master
"""

import torch
import data_handler
import model
import deepcluster_model as cmodel
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--epochs',default=200,type=int,help='Number of epochs to train for')
parser.add_argument('--batch',default=256,type=int,help='Batch size')
parser.add_argument('--validation_step',default=1,type=int,help='Number of epochs after which we validate model')
parser.add_argument('--save_epochs',default=1,type=int,help='Number of epochs after which we save model checkpoint')
parser.add_argument('--benchmark', choices=['0','1'], default='0',help='Choice of benchmark: 0: SVHN->MNIST, 1: MNIST->USPS')
parser.add_argument('--device', choices=['cpu','cuda', 'mps'], default='mps',help='Choice of device: cpu, cuda, mps')

args = parser.parse_args()

checkpoint_dir = "./checkpoints/"

def compute_kernel(x, y):
    """
    Computing kernel for MMD Loss.
    Modified from https://github.com/snap-stanford/GraphRNN
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    """
    Computing MMD Loss.
    Modified from https://github.com/snap-stanford/GraphRNN
    """
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def extract_features(dataloader, model):
    """
    Extracting feature vectors for a certain dataset.
    """
    features = []
    labels = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, targets = data
            output = model.autoencoder.encode(images.to(args.device))
            features.append(output)
            labels.append(targets.to(args.device))
    features = torch.cat(features, 0)
    labels = torch.cat(labels, 0)
    return features, labels

# Get datasets for training and testing.
if args.benchmark == '0':
    source_testloader = data_handler.get_dataloader_svhn_test(batch_size=args.batch)
    target_testloader = data_handler.get_dataloader_mnist_test(batch_size=args.batch)
    target_trainloader = data_handler.get_dataloader_mnist_train(batch_size=args.batch)
    source_target_trainloader = data_handler.get_dataloader_svhn_mnist_train(batch_size=args.batch)
else:
    source_testloader = data_handler.get_dataloader_usps_test(batch_size=args.batch)
    target_testloader = data_handler.get_dataloader_mnist_test(batch_size=args.batch)
    target_trainloader = data_handler.get_dataloader_mnist_train(batch_size=args.batch)
    source_target_trainloader = data_handler.get_dataloader_usps_mnist_train(batch_size=args.batch)

# Get models.
student_encoder = model.LeNet_Encoder()
source_classifier = model.Classifier()
source_to_target_projector = model.Class_Projector()

autoencoder = cmodel.AutoEncoder()
teacher_encoder = cmodel.DEC(n_clusters=10, autoencoder=autoencoder, hidden=10, cluster_centers=None, alpha=1.0)
teacher_encoder.load_state_dict(torch.load("./cluster_models/dec.pth")['state_dict'])

# Init optimizer.
optimizer = optim.Adam([{'params':student_encoder.parameters()},{'params':source_classifier.parameters()},{'params':source_to_target_projector.parameters()}], lr =  0.001)

loss_cross_entropy = nn.CrossEntropyLoss()

student_encoder = student_encoder.to(args.device)
source_classifier = source_classifier.to(args.device)
source_to_target_projector = source_to_target_projector.to(args.device)
teacher_encoder = teacher_encoder.to(args.device)
teacher_encoder.eval()

student_encoder_model_path_latest = checkpoint_dir + "source_enocder_latest.pt"
source_classifier_model_path_latest   = checkpoint_dir + "source_classifier_latest.pt"

# Extract feature vectors for target data.
centroids = []
train_target_features, original_labels = extract_features(target_trainloader, teacher_encoder)

# Kmeans to get cluster centroids.
k_encoded = KMeans(n_clusters=10)
k_encoded.fit(train_target_features.cpu().numpy())
centroids = k_encoded.cluster_centers_
centroids = torch.tensor(centroids, dtype=torch.float32).to(args.device)

best_acc = 0.0
best_epoch = 0

for epoch in range(args.epochs):
    print("EPOCH : " + str(epoch))
    epoch_train_loss = []
    epoch_test_loss = []

    student_encoder.train()
    source_classifier.train()
    source_to_target_projector.train()

    for i,(data_source, data_target) in enumerate(tqdm(source_target_trainloader)):
            
        optimizer.zero_grad()
        source_images = data_source[0]
        source_labels = data_source[1]
        target_images = data_target[0]

        usps_images = usps_images.to(args.device)
        usps_labels = usps_labels.to(args.device)
        mnist_images = mnist_images.to(args.device)

        # Get target data pseudo-labels.
        teacher_target_encodings = teacher_encoder.autoencoder.encode(target_images)
        distances = torch.cdist(teacher_target_encodings, centroids)
        target_pseudo_labels = torch.softmax(1 / (distances.detach() + 1e-10), -1)

        # Get source data encodings and classification predictions.
        student_source_encodings = student_encoder(source_images)
        _, student_source_pred = source_classifier(student_source_encodings)
        loss_class_source  =  loss_cross_entropy(student_source_pred, source_labels)

        # Get target data encodings and cluster projections.
        student_target_encodings = student_encoder(target_images)
        student_target_pre_pred, student_target_pred = source_classifier(student_target_encodings)
        student_encoder_target_proj = source_to_target_projector(student_target_pre_pred)

        # Calculate MMD loss.
        mmd_loss = compute_mmd(student_source_encodings, student_target_encodings)

        loss_cluster = loss_cross_entropy(student_encoder_target_proj, target_pseudo_labels)

        total_loss = loss_class_source + loss_cluster + mmd_loss

        total_loss.backward()
        optimizer.step()
        total_loss = total_loss.detach().cpu().numpy()
    
    # Save model.
    if epoch % args.save_epochs == 0:
        student_encoder_path = os.path.join(checkpoint_dir,str(epoch)+"_student_encoder.pt")
        source_classifier_path = os.path.join(checkpoint_dir,str(epoch)+"_source_classifier.pt")
        torch.save(student_encoder.state_dict(),student_encoder_path)
        torch.save(source_classifier.state_dict(),source_classifier_path)

    # Eval model.
    if epoch % args.validation_step == 0:
        print("Total training Loss for epoch : {epoch} = {total_loss}".format(epoch = epoch , total_loss = total_loss))
        with torch.no_grad() :
            student_encoder = student_encoder.eval()
            source_classifier = source_classifier.eval()
            model_acc = 0
            num_samples = 0
            for i,(image,label) in enumerate(target_testloader):
                image = image.to(args.device)
                label = label.to(args.device)
                student_encodings = student_encoder(image)
                _, pred = source_classifier(student_encodings)
                _, pred_idx = pred.max(dim=-1)
                model_acc += torch.sum(pred_idx == label)
                num_samples += image.shape[0]
                loss  =  loss_cross_entropy(pred,label).cpu().numpy()
                epoch_test_loss.append(loss)
            testing_loss_epoch = np.mean(np.asarray(epoch_test_loss))

            print("Student Target Testing Loss for epoch : {epoch} = {testing_loss_epoch}".format(epoch = epoch , testing_loss_epoch = testing_loss_epoch))
            acc = model_acc/num_samples
            if best_acc < acc:
                best_acc = acc
                best_epoch = epoch
            print("Student Target Accuracy : {acc}".format(acc = acc))
            
            model_acc = 0
            num_samples = 0
            for i,(image,label) in enumerate(source_testloader):
                image = image.to(args.device)
                label = label.to(args.device)
                student_encodings = student_encoder(image)
                _, pred = source_classifier(student_encodings)
                _, pred_idx = pred.max(dim=-1)
                model_acc += torch.sum(pred_idx == label)
                num_samples += image.shape[0]
                loss  =  loss_cross_entropy(pred,label).cpu().numpy()
                epoch_test_loss.append(loss)
            testing_loss_epoch = np.mean(np.asarray(epoch_test_loss))
            print("Student Source Testing Loss for epoch : {epoch} = {testing_loss_epoch}".format(epoch = epoch , testing_loss_epoch = testing_loss_epoch))
            acc = model_acc/num_samples
            print("Student Source Accuracy : {acc}".format(acc = acc))

print("Best model: epoch -> " + str(best_epoch) + " ,acc -> " + str(best_acc.detach().cpu().numpy()))

