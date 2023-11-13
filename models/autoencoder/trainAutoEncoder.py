import torch

def train(model, train_dataloader, val_dataloader, epochs, criterion, optimizer, device):
    model.to(device); model.train()
    train_avg_losses, val_avg_losses = [], []
    train_losses, outputs = [], []
    best_val_loss = float('inf')
    early_stopping_counter = 0; patience = 5
    
    for epoch in range(epochs):
        for image in train_dataloader:               
            image = image.to(device)
            reconstructed_img = model(image)
            loss = criterion(reconstructed_img, image)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        train_avg_loss = (sum(train_losses) / len(train_losses))*10
        
        model.eval()    
        val_losses = []
        with torch.no_grad():
            for image in val_dataloader:
                image = image.to(device)
                reconstructed_img = model(image)
                loss = criterion(reconstructed_img, image)
                val_losses.append(loss.cpu().detach().item())
            val_avg_loss = (sum(val_losses)/len(val_losses))*100
        
        print('epoch [{}/{}], train loss:{:.4f}, val loss:{:-4f}'
              .format(epoch + 1, epochs, train_avg_loss, val_avg_loss)) 
        
        train_avg_losses.append(train_avg_loss)
        val_avg_losses.append(val_avg_loss)
        outputs.append((epoch, image, reconstructed_img))
        
        # Check for early stopping
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping! Validation loss has not decreased for {} epochs.".format(patience))
                break
        
    return train_avg_losses, val_avg_losses, outputs

# def validate(model, dataloader, criterion, device):
#     model.eval()
#     losses = []
#     with torch.no_grad():
#         for image in dataloader:
#             image = image.to(device)
#             reconstructed_img = model(image)
#             loss = criterion(reconstructed_img, image)
#             losses.append(loss.cpu().detach().item())
#         average_loss = sum(losses)/len(losses)
#         return average_loss            