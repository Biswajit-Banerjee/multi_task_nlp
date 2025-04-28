import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm 
import pandas as pd

# lcoal imports
from .dataset import ATISDataset
from .models import SentenceEncoder, ATISMultiTaskModel
from .utils import compute_intent_accuracy, compute_slot_f1


# for displaying properly
TEMPLATE = (
    "Epoch {epoch:>2} | "
    "Train > Loss: {train_loss:.4f} | Intent Acc: {train_intent_acc:.2%} | Entity F1: {train_entity_f1:.2%} || "
    "Val   > Loss: {val_loss:.4f} | Intent Acc: {val_intent_acc:.2%} | Entity F1: {val_entity_f1:.2%}"
)

def collate_batch(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0] if k != 'gt'}

def multi_loss(intent_pair, slot_pair, l1=0.3, l2=0.7):
    loss_i = nn.CrossEntropyLoss()(*intent_pair)
    loss_s = nn.CrossEntropyLoss()(*slot_pair)
    
    loss = l1*loss_i + l2*loss_s
    
    return loss, loss_i, loss_s

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    
    running_loss, running_i_loss, running_s_loss = 0.0, 0.0, 0.0
    total_i_acc, total_s_f1 = 0.0, 0.0

    
    for bid, batch in (pbar:=tqdm(enumerate(loader, start=1), total=len(loader), desc="Training", leave=False)):
        optimizer.zero_grad()
        # put data to device
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        intent_labels  = batch["intent_label"].to(device)
        slot_labels    = batch["slot_labels"].to(device)

        # forward pass
        logits_i, logits_s = model(input_ids, attention_mask)
        
        # flatten and mask slot predictions
        intent_pair = (logits_i, intent_labels)
        flat_mask   = attention_mask.view(-1) == 1
        slot_pair   = (
            logits_s.view(-1, logits_s.size(-1))[flat_mask],
            slot_labels.view(-1)[flat_mask]
        )
        
        # compute loss and update network 
        loss, loss_i, loss_s = multi_loss(intent_pair, slot_pair)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Accumulate metrics
        running_loss   += loss.item()
        running_i_loss += loss_i.item()
        running_s_loss += loss_s.item()
        
        # Compute training-phase accuracy/F1 on this batch
        preds_i = torch.argmax(logits_i, dim=-1)
        total_i_acc += compute_intent_accuracy(preds_i, intent_labels)
        total_s_f1  += compute_slot_f1(
            torch.argmax(logits_s, dim=-1),
            batch["slot_labels"],
            batch["attention_mask"]
        )
        
        pbar.set_postfix({"batch id": bid, "loss": round(running_loss/bid, 3), "acc": round(total_s_f1/bid, 3)})

    n = len(loader)
    metrics = {
        "train_loss":         running_loss   / n,
        "train_intent_loss":  running_i_loss / n,
        "train_entity_loss":  running_s_loss / n,
        "train_intent_acc":   total_i_acc    / n,
        "train_entity_f1":    total_s_f1     / n,
    }
    
    return metrics

def validate(model, loader, device):
    model.eval()
    
    running_loss, running_i_loss, running_s_loss = 0.0, 0.0, 0.0
    total_i_acc, total_s_f1 = 0.0, 0.0

    with torch.no_grad():
        
        for bid, batch in (pbar:=tqdm(enumerate(loader, start=1), total=len(loader), desc="Validation", leave=False)):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels  = batch["intent_label"].to(device)
            slot_labels    = batch["slot_labels"].to(device)

            logits_i, logits_s = model(input_ids, attention_mask)
            
            intent_pair = (logits_i, intent_labels)
            slot_pair   = (
                logits_s.view(-1, logits_s.size(-1)),
                slot_labels.view(-1)
            )
            
            loss, loss_i, loss_s = multi_loss(intent_pair, slot_pair)
                
            # Accumulate metrics
            running_loss   += loss.item()
            running_i_loss += loss_i.item()
            running_s_loss += loss_s.item()
            
            preds_i = torch.argmax(logits_i, dim=-1)
            total_i_acc += compute_intent_accuracy(preds_i, intent_labels)
            
            total_s_f1 += compute_slot_f1(
                logits_s.argmax(-1), slot_labels, batch["attention_mask"]
            )
            
            pbar.set_postfix({"batch id": bid, "loss": round(running_loss/bid, 3), "acc": round(total_s_f1/bid, 3)})

    n = len(loader)
    metrics = {
        "val_loss":         running_loss   / n,
        "val_intent_loss":  running_i_loss / n,
        "val_entity_loss":  running_s_loss / n,
        "val_intent_acc":   total_i_acc    / n,
        "val_entity_f1":    total_s_f1     / n,
    }
    
    return metrics


def train_model(train_ds, val_ds, model, epochs=3):
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_batch)
    val_loader   = DataLoader(val_ds, batch_size  =64, shuffle=False, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps  = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    losses_data = [] 

    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device=device)

        valid_metrics = validate(model, val_loader, device)
        
        # combines two dicts into 
        metrics = {**train_metrics, **valid_metrics, "epoch": epoch+1}
        metrics['epoch'] = epoch
        losses_data.append(metrics) 
        
        print(TEMPLATE.format(**metrics))
        
    return losses_data

