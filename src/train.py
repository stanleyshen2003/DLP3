import argparse
from oxford_pet import load_dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
from models.unet import UNet
from torch.utils.tensorboard import SummaryWriter
from evaluate import evaluate

def train(args):
    # implement the training function here
    data_path = args.data_path
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter()
    train_dataset = load_dataset(data_path, mode='train')
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0.0
        for _, sample in enumerate(data_loader):
            image = sample['image'].to(device)
            mask = sample['mask'].to(device)
            trimap = sample['trimap']
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(image)
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        writer.add_scalar('training loss', running_loss / len(data_loader), epoch)
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(data_loader)}')

        if epoch % 10 == 0:
            score = evaluate(model, data_loader, device)
            writer.add_scalar('training dice score', score, epoch)
            print(f'Dice score: {score}')


    

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default="../dataset/oxford-iiit-pet", help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)