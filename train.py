import argparse
import torch
import torch.optim as optim
from torchvision.utils import save_image
from pathlib import Path
from torch.utils.data import DataLoader
from utils.utils import *
from utils.models import *
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-dir', type=str, default='/kaggle/working/NST', help = 'Path to project directory')
    parser.add_argument('--content_dir', type=str, default='/kaggle/input/datasets/siriyd/nstfiles/NST/content_data', help = 'Path to content directory')
    parser.add_argument('--style_dir', type=str, default='/kaggle/input/datasets/siriyd/nstfiles/NST/style_data', help = 'Path to style directory')
    parser.add_argument('--vgg', type=str, default=r'/kaggle/input/datasets/siriyd/nstfiles/NST/vgg_normalised.pth', help = 'Path to pre-trained vgg model')
    parser.add_argument('--experiment', type=str, default='experiment1', help = 'Name of experiment')
    parser.add_argument('--final_size', type=int, default=512, help = 'Size of final image')
    parser.add_argument('--content_size', type=int, default=256, help = 'Size of content image')
    parser.add_argument('--style_size', type=int, default=256, help = 'Size of style image')
    parser.add_argument('--crop', action='store_true', help = 'crop image')
    parser.add_argument('--batch_size', type=int, default=4, help = 'Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help = 'Learning rate')  
    parser.add_argument('--lr_decay', type=float, default=5e-5, help = 'Learning rate decay')
    parser.add_argument('--epochs', type=int, default=160, help = 'Number of epochs')
    parser.add_argument('--content_weight', type=float, default=1, help = 'Content weight')
    parser.add_argument('--style_weight', type=float, default=5, help = 'Style weight')   
    parser.add_argument('--save_interval', type=int, default=20, help = 'Save model every n epochs')
    parser.add_argument('--resume', action='store_true', help = 'Resume from previous checkpoint')
    parser.add_argument('--decoder_path', type=str, help = 'Path to decoder checkpoint')
    parser.add_argument('--optimizer_path', type=str, help = 'Path to optimizer checkpoint')
    args = parser.parse_args()
    return args



def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    save_dir = Path(args.project_dir) / 'experiment' / args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)

    #save arguement values
    with open(save_dir / 'args.txt', 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')

    content_transform=get_transform(args.content_size, args.final_size, args.crop)
    style_transform=get_transform(args.style_size, args.final_size, args.crop)

    content_dataset = ImageFolderDataset(args.content_dir,content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    content_dataloader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True) # shuffle - shuffle the dataset every epoch 
    style_dataloader = DataLoader(style_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True) # # discard the final incomplete batch

    print("Number of batches in content dataloader:", len(content_dataloader))
    print("Number of batches in style dataloader:", len(style_dataloader))

    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)

    scheduler=optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1/(epoch * args.lr_decay +1)  
    )

    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path))
        optimizer.load_state_dict(torch.load(args.optimizer_path))

    mse_loss = torch.nn.MSELoss()

    encoder.eval()

    running_loss = None
    running_closs = None
    running_sloss = None

    for epoch in range(args.epochs):
        num_batches = min(
            len(content_dataloader),
            len(style_dataloader)
        )
        
        progress_bar = tqdm(zip(content_dataloader, style_dataloader), total=num_batches)

        running_loss = 0
        running_closs = 0
        running_sloss = 0

        for content_batch, style_batch in progress_bar:
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            content_features = encoder(content_batch)
            style_features = encoder(style_batch)

            t = adaptive_instance_normalization(content_features[-1], style_features[-1])

            g = decoder(t)

            g_features = encoder(g)

            loss_c = mse_loss(g_features[-1], t) * args.content_weight

            loss_s = 0

            for g_f,s_f in zip(g_features, style_features):
                g_mean,g_std = calculate_mean_std(g_f)
                s_mean,s_std = calculate_mean_std(s_f)
                loss_s += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)

            loss_s *= args.style_weight

            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f'Loss:{loss.item():.4f}, Content Loss:{loss_c.item():.4f}, Style Loss:{loss_s.item():.4f}')

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()

        scheduler.step()

        running_loss /= num_batches
        running_closs /= num_batches
        running_sloss /= num_batches

        tqdm.write(f'Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss:.4f}, Content Loss: {running_closs:.4f}, Style Loss: {running_sloss:.4f}')

        if (epoch+1)%args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f'decoder_epoch_{epoch+1}.pth')
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_epoch_{epoch+1}.pth')

            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0)
                save_image(output, save_dir / f'output_epoch_{epoch+1}.png', nrow=args.batch_size)

if __name__ == '__main__':
    main()