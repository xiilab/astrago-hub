import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os
from tempfile import TemporaryDirectory
from utils.astrago import Astrago
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        
        for epoch in Astrago(range(num_epochs), desc='Epoch'):
            t_start_time = None
            v_start_time = None
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    t_start_time = time.time()
                    model.train()  # Set model to training mode
                if phase == 'val':
                    v_start_time = time.time()
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                
                # Iterate over data.
                for inputs, labels in tqdm(dataloaders[phase], desc='Data Num'):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                if phase == 'train':
                    scheduler.step()
                
                if (phase == 'train') and (t_start_time != None):
                    Astrago.get_elapsed_train_time(t_start_time)
                    
                if (phase == 'val') and (v_start_time != None):
                    Astrago.get_elapsed_val_time(v_start_time)
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


def get_args():
    parser = argparse.ArgumentParser(description='Train ResNeXT')
    parser.add_argument('--data_dir', type=str, default='./data/', help='data directory path')
    parser.add_argument('--batch', type=int, default=4,help="Batch Size")
    parser.add_argument('--epoch', type=int, default=25, help='epochs')
    parser.add_argument('--imgsz', type=int, default=224, help='Input Image Size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--pretrained', type=bool, default=True, help='pre-trained select')
    return parser.parse_args()


if __name__=='__main__':
    args = get_args()\
        
    data_dir = args.data_dir
    batch_size = args.batch
    epoch = args.epoch
    imgsz = args.imgsz
    lr = args.lr  
    pretrained_check = args.pretrained 
    
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
    model_ft = models.resnext101_64x4d(pretrained=pretrained_check)
    
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    
    Astrago.get_gpu_info()
    Astrago.get_model_params(model_ft)
    Astrago.get_image_size(args.imgsz)
    Astrago.get_batch_size(args.batch)
    Astrago.get_data_info(len(image_datasets['train']), len(image_datasets['val']))
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epoch)