from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def dataloader(data_root, input_size, batch_size, num_workers):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    dataset = datasets.ImageFolder(
        root = data_root,
        transform=transform
    )
        
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers) 

    return dl