import torch
import torch.nn as nn
import wandb
from accelerate import Accelerator
from tqdm import tqdm

from data import tranforms
from data.datasets.imageNette import ImageNette
from model import resnet18, resnet34

IMG_SIZE = 224
train_transformer, eval_transformer= tranforms(IMG_SIZE)

Configs = dict(
    MODEL_NAME='',
    DATA_DIR="../input/imagenette/imagenette",
    TRAIN_DATA_DIR="../input/imagenette/imagenette/train",
    TEST_DATA_DIR="../input/imagenette/imagenette/val",
    DEVICE="cuda",
    PRETRAINED=False,
    LR=1e-5,
    EPOCHS=50,
    NOISE_LEVEL=0, # 0 1 5 25 50
    IMG_SIZE=IMG_SIZE,
    BS=64,
    TRAIN_AUG=train_transformer,
    TEST_AUG=eval_transformer,
)

def train_fn(model, train_data_loader, optimizer, epoch, accelerator,scheduler):
    model.train()
    fin_loss = 0.0
    tk = tqdm(train_data_loader, desc="Epoch" + " [TRAIN] " + str(epoch + 1))

    for t, data in enumerate(tk):
        optimizer.zero_grad()
        out = model(data[0])
        data[1] = data[1].type(torch.LongTensor)
        data[1].to(Configs["DEVICE"])
        loss = nn.CrossEntropyLoss()(
            out, data[1].flatten()
            )
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step(epoch + t / len(train_data_loader))
        fin_loss += loss.item()
        tk.set_postfix(
            {
                "loss": "%.6f" % float(fin_loss / (t + 1)),
                "LR": optimizer.param_groups[0]["lr"],
            }
        )
    return fin_loss/len(train_data_loader), optimizer.param_groups[0]["lr"]

def eval_fn(model, eval_data_loader, epoch):
    model.eval()
    fin_loss = 0.0
    tk = tqdm(eval_data_loader, desc="Epoch" + " [VALID] " + str(epoch + 1))

    with torch.no_grad():
        for t, data in enumerate(tk):
            out = model(data[0])
            data[1] = data[1].type(torch.LongTensor)
            data[1].to(Configs["DEVICE"])
            loss = nn.CrossEntropyLoss()(
                out, data[1].flatten()
                )
            fin_loss += loss.item()
            tk.set_postfix({"loss": "%.6f" % float(fin_loss / (t + 1))})
        return fin_loss/len(eval_data_loader)

def train():
    
    accelerator = Accelerator()

    # wandb inita
    wandb.init(config=Configs, project='imagenette', save_code=True, 
           job_type='train', tags=['lambda', 'imagenette'], 
           name=Configs['MODEL_NAME'])    
    
    train_dataset = ImageNette(
        csv_file='../input/imagenette/imagenette/train_noisy_imagenette.csv', 
        root_dir='../input/imagenette/imagenette', 
        noisy_level=Configs["NOISE_LEVEL"], 
        transform=Configs['TRAIN_AUG'], 
        train=True 
    )
    eval_dataset = ImageNette(
        csv_file='../input/imagenette/imagenette/val_noisy_imagenette.csv', 
        root_dir='../input/imagenette/imagenette', 
        noisy_level=0, 
        transform=Configs['TEST_AUG'], 
        train=False
    )

    # train and eval dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Configs["BS"]
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=Configs["BS"]
    )

    # model
    model = resnet34().cuda()

    # optimizer    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Configs["LR"]
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=0)
    # prepare for DDP
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)    

    for epoch in range(Configs["EPOCHS"]):
        avg_loss_train, lr = train_fn(
            model, train_dataloader, optimizer, epoch, accelerator,scheduler)
        avg_loss_eval = eval_fn(
            model, eval_dataloader, epoch)
        wandb.log({'train_loss': avg_loss_train, 'eval_loss': avg_loss_eval, 'lr': lr})
    torch.save(model.state_dict(), Configs["MODEL_NAME"]+'.pt')

if  __name__ == "__main__" :
    train()
