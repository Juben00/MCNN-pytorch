import os
import torch
import torch.nn as nn
import visdom
import random
import sys

from mcnn_model import MCNN
from my_dataloader import CrowdDataset

# Simple configuration class for train.py
class TrainConfig:
    def __init__(self):
        # Training parameters
        self.epochs = 300
        self.early_stopping_patience = 20
        self.lr_scheduler_patience = 8
        self.lr_scheduler_factor = 0.5
        self.min_lr = 1e-6
        self.gradient_clip_norm = 0.5
        self.weight_decay = 1e-4  # Add weight decay for regularization

if __name__=="__main__":
    torch.backends.cudnn.enabled=False
    vis=visdom.Visdom()
    device=torch.device("cuda")
    mcnn=MCNN().to(device)
    criterion=nn.MSELoss(size_average=False).to(device)
    
    # Add configuration and use weight_decay in optimizer
    cfg = TrainConfig()
    optimizer = torch.optim.SGD(mcnn.parameters(), lr=3e-5, momentum=0.95, weight_decay=cfg.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.lr_scheduler_factor, 
        patience=cfg.lr_scheduler_patience, min_lr=cfg.min_lr
    )
    
    img_root='./data/train_data/images'
    gt_dmap_root='./data/train_data/densitymaps'
    dataset=CrowdDataset(img_root,gt_dmap_root,4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=True)

    test_img_root='./data/test_data/images'
    test_gt_dmap_root='./data/test_data/densitymaps'
    test_dataset=CrowdDataset(test_img_root,test_gt_dmap_root,4)
    test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=2,shuffle=False)

    #training phase
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    min_mae=sys.maxsize  # Use sys.maxsize instead of hardcoded value
    min_epoch=0
    epochs_without_improvement = 0  # Early stopping counter
    train_loss_list=[]
    epoch_list=[]
    test_error_list=[]
    
    for epoch in range(0, cfg.epochs):  # Use config epochs

        mcnn.train()
        epoch_loss=0
        for i,(img,gt_dmap) in enumerate(dataloader):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=mcnn(img)
            # calculate loss
            loss=criterion(et_dmap,gt_dmap)
            epoch_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(mcnn.parameters(), max_norm=cfg.gradient_clip_norm)
            
            optimizer.step()
        #print("epoch:",epoch,"loss:",epoch_loss/len(dataloader))
        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss/len(dataloader))

        mcnn.eval()
        mae=0
        with torch.no_grad():  # Add no_grad for evaluation
            for i,(img,gt_dmap) in enumerate(test_dataloader):
                img=img.to(device)
                gt_dmap=gt_dmap.to(device)
                # forward propagation
                et_dmap=mcnn(img)
                mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
                del img,gt_dmap,et_dmap
        
        current_mae = mae/len(test_dataloader)
        
        # Check for improvement and early stopping
        if current_mae < min_mae:
            min_mae = current_mae
            min_epoch = epoch
            epochs_without_improvement = 0
            # Save best model
            torch.save(mcnn.state_dict(),'./checkpoints/best_epoch_'+str(epoch)+".pth")
            print(f"✅ New best MAE: {min_mae:.2f} at epoch {epoch}")
        else:
            epochs_without_improvement += 1
        
        # Early stopping check
        if epochs_without_improvement >= cfg.early_stopping_patience:
            print(f"🛑 Early stopping triggered after {epochs_without_improvement} epochs without improvement")
            break
        
        # Step the scheduler and log LR changes
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(current_mae)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        test_error_list.append(current_mae)
        print("epoch:"+str(epoch)+" error:"+str(current_mae)+" min_mae:"+str(min_mae)+" min_epoch:"+str(min_epoch))
        vis.line(win=1,X=epoch_list, Y=train_loss_list, opts=dict(title='train_loss'))
        vis.line(win=2,X=epoch_list, Y=test_error_list, opts=dict(title='test_error'))
        # show an image
        index=random.randint(0,len(test_dataloader)-1)
        img,gt_dmap=test_dataset[index]
        vis.image(win=3,img=img,opts=dict(title='img'))
        vis.image(win=4,img=gt_dmap/(gt_dmap.max())*255,opts=dict(title='gt_dmap('+str(gt_dmap.sum())+')'))
        img=img.unsqueeze(0).to(device)
        gt_dmap=gt_dmap.unsqueeze(0)
        et_dmap=mcnn(img)
        et_dmap=et_dmap.squeeze(0).detach().cpu().numpy()
        vis.image(win=5,img=et_dmap/(et_dmap.max())*255,opts=dict(title='et_dmap('+str(et_dmap.sum())+')'))

    import time
    print(time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time())))