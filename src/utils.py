from sklearn.metrics import f1_score

# Evaluation metrics
def compute_intent_accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    return correct / labels.shape[0]


def compute_slot_f1(preds, labels, mask):
    
    valid_idxs =   mask.flatten().detach().cpu().numpy().astype(bool)
    y_true     = labels.flatten()[valid_idxs].detach().cpu().numpy()
    y_pred     =  preds.flatten()[valid_idxs].detach().cpu().numpy()
    
    return f1_score(y_true, y_pred, average="micro")
