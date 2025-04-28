from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Evaluation metrics
def compute_intent_accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    return correct / labels.shape[0]


def compute_slot_f1(preds, labels, mask):
    
    valid_idxs =   mask.flatten().detach().cpu().numpy().astype(bool)
    y_true     = labels.flatten()[valid_idxs].detach().cpu().numpy()
    y_pred     =  preds.flatten()[valid_idxs].detach().cpu().numpy()
    
    return f1_score(y_true, y_pred, average="micro")

def plot_metrics(losses_data):
    # Extract epoch & series
    epochs             = [m['epoch']               for m in losses_data]
    train_intent_loss  = [m['train_intent_loss']   for m in losses_data]
    val_intent_loss    = [m['val_intent_loss']     for m in losses_data]
    train_entity_loss  = [m['train_entity_loss']   for m in losses_data]
    val_entity_loss    = [m['val_entity_loss']     for m in losses_data]
    train_intent_acc   = [m['train_intent_acc']    for m in losses_data]
    val_intent_acc     = [m['val_intent_acc']      for m in losses_data]
    train_entity_f1    = [m['train_entity_f1']     for m in losses_data]
    val_entity_f1      = [m['val_entity_f1']       for m in losses_data]

    # Create 2x2 subplots, share y-axis per row
    fig, axs = plt.subplots(
        nrows=2, ncols=2,
        figsize=(12, 8),
        sharey='row'
    )

    # Color map
    colors = {
        'Train': 'tab:blue',
        'Val':   'tab:green'
    }

    # Top-left: Intent Loss
    ax = axs[0, 0]
    ax.plot(epochs, train_intent_loss, label='Train', color=colors['Train'])
    ax.plot(epochs, val_intent_loss,   label='Val',   color=colors['Val'])
    ax.set_title('Intent Loss')
    ax.set_ylabel('< Loss')
    ax.grid(True)

    # Top-right: Entity Loss
    ax = axs[0, 1]
    ax.plot(epochs, train_entity_loss, label='Train', color=colors['Train'])
    ax.plot(epochs, val_entity_loss,   label='Val',   color=colors['Val'])
    ax.set_title('Entity Loss')
    # share y-label with left
    ax.grid(True)

    # Bottom-left: Intent Accuracy
    ax = axs[1, 0]
    ax.plot(epochs, train_intent_acc, label='Train', color=colors['Train'])
    ax.plot(epochs, val_intent_acc,   label='Val',   color=colors['Val'])
    ax.set_title('Intent Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy >')
    ax.grid(True)

    # Bottom-right: Entity F1
    ax = axs[1, 1]
    ax.plot(epochs, train_entity_f1, label='Train', color=colors['Train'])
    ax.plot(epochs, val_entity_f1,   label='Val',   color=colors['Val'])
    ax.set_title('Entity F1 Score')
    ax.set_xlabel('Epoch')
    ax.grid(True)

    # Global legend and title
    fig.legend(
        ['Train', 'Validation'],
        loc='upper center',
        ncol=2,
        frameon=True,
        fontsize='x-large'
    )
    # fig.suptitle('Training vs Validation Metrics', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()
    
    

