import torch
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer, device, accumulation_steps=4):
    model.train()
    train_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    # Set the gradients to zero
    optimizer.zero_grad()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss = loss / accumulation_steps  # Scale loss by accumulation steps
        train_loss += loss.item() * accumulation_steps  # Accumulate the true loss
        
        loss.backward()

        if (batch + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        y_pred_class = torch.argmax(y_pred, dim=1)
        correct_predictions += (y_pred_class == y).sum().item()
        total_samples += y.size(0)
    
    # If the number of batches is not divisible by accumulation_steps
    if (batch + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    train_loss /= len(dataloader)
    train_acc = correct_predictions / total_samples
    
    return train_loss, train_acc

def test_step(model, dataloader, loss_fn, device):
    model.eval()
    test_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            test_pred_labels = test_pred_logits.argmax(dim=1)
            correct_predictions += (test_pred_labels == y).sum().item()
            total_samples += y.size(0)
    
    test_loss /= len(dataloader)
    test_acc = correct_predictions / total_samples
    
    return test_loss, test_acc

def train(model, epochs, train_dataloader, test_dataloader, optimizer, loss_fn, device, accumulation_steps=4):
    results = {
        "train_acc": [],
        "train_loss": [],
        "test_acc": [],
        "test_loss": []
    }
    
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        # Training
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device, accumulation_steps)
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        
        # Testing
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
        # Print results for this epoch
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    return results
