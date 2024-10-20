import torch
import data_handler
import model
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch',default=256,type=int,help='Batch size')
parser.add_argument('--benchmark', choices=['0','1'], default='0',help='Choice of benchmark: 0: SVHN->MNIST, 1: MNIST->USPS')
parser.add_argument('--device', choices=['cpu','cuda', 'mps'], default='mps',help='Choice of device: cpu, cuda, mps')
parser.add_argument('--encoder_path',type=str,help='Path to test encoder\'s weight')
parser.add_argument('--classifier_path',type=str,help='Path to test classifier\'s weight')

args = parser.parse_args()

# Getting the training and test datasets.
if args.benchmark == '0':
    target_trainloader = data_handler.get_dataloader_mnist_test(batch_size=args.batch)
    source_trainloader = data_handler.get_dataloader_svhn_test(batch_size=args.batch)
else:
    target_trainloader = data_handler.get_dataloader_mnist_test(batch_size=args.batch)
    source_trainloader = data_handler.get_dataloader_usps_test(batch_size=args.batch)

# Init and load model.
encoder = model.LeNet_Encoder()
classifier = model.Classifier()

encoder.load_state_dict(torch.load(args.encoder_path))
classifier.load_state_dict(torch.load(args.classifier_path))

encoder = encoder.to(args.device)
classifier = classifier.to(args.device)

with torch.no_grad() :
    # Eval model on target dataset.
    encoder = encoder.eval()
    classifier = classifier.eval()
    model_acc = 0
    num_samples = 0
    for i,(image,label) in enumerate(target_trainloader):
        image = image.to(args.device)
        label = label.to(args.device)
        source_encodings = encoder(image)
        _, pred = classifier(source_encodings)
        _, pred_idx = pred.max(dim=-1)
        model_acc += torch.sum(pred_idx == label)
        num_samples += image.shape[0]
    
    acc = model_acc/num_samples
    print("Model Target Accuracy : {acc}".format(acc = acc))
    
    # Eval model on source dataset.
    model_acc = 0
    num_samples = 0
    for i,(image,label) in enumerate(source_trainloader):
        image = image.to(args.device)
        label = label.to(args.device)
        source_encodings = encoder(image)
        _, pred = classifier(source_encodings)
        _, pred_idx = pred.max(dim=-1)
        model_acc += torch.sum(pred_idx == label)
        num_samples += image.shape[0]

    acc = model_acc/num_samples
    print("Model Source Accuracy : {acc}".format(acc = acc))

