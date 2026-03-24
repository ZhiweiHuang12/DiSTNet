import torch
from src.condition_diffusion.ddpm import DDPM
import tqdm
from torch.utils.data.dataloader import DataLoader
import random
from src.data.afl import *

def train(
    model: DDPM,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    save_dir = "",
    json_data = [],
    max_value= 100.0
):
    training_losses = []
    val_losses = []
    for epoch in range(epochs):
        # if epoch>0:
        sub_json_data = random.sample(json_data, 1000)
        dataset,train_dataloader = get_dataloader(sub_json_data,max_value=max_value)
        model.train(True)
        training_loss = 0
        val_loss = 0
        pbar = tqdm.tqdm(train_dataloader)
        for index, (imgs, labels) in enumerate(pbar):
            optimizer.zero_grad()
            
            imgs = imgs.to(device)
            labels = labels.to(device)
    
            loss = model(imgs, labels)
    
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            pbar.set_description(f"loss for epoch {epoch}: {training_loss / (index + 1):.4f}")
        if epoch%2==0:
            torch.save(model.state_dict(), save_dir+"model_{}.pth".format(epoch))
        # model.eval()
        # with torch.no_grad():
        #     for (imgs, labels) in val_dataloader:
        #         imgs = imgs.to(device)
        #         labels = labels.to(device)
    
        #         loss = model(imgs, labels)
        
        #         val_loss += loss.item()
        # training_losses.append(training_loss / len(val_dataloader))
        # val_losses.append(val_loss / len(val_dataloader))
    return training_losses, val_losses