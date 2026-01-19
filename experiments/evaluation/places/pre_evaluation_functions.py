import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score
import pandas as pd
from tqdm import tqdm


def evaluate_with_thresholds(model, dataloader, num_classes, thresholds=[i/10 for i in range(10)], device='cuda', custome_targets=None):
    """
    Evaluate precision/recall/f1 at different confidence thresholds
    """
    if custome_targets is None:
        custome_targets = [i for i in range(num_classes)]

    model.eval()
    
    # Collect all predictions first
    all_probs = []
    all_labels = []
    
    print("Collecting predictions...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            
            all_probs.append(probs[:, custome_targets].cpu())
            all_labels.append(labels)
    
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)


    
    # Get top-1 predictions and confidences
    confidences, predictions = torch.max(all_probs, dim=1)
    
    results = []
    print("\nComputing metrics for each threshold...")
    for threshold in tqdm(thresholds, desc="Thresholds"):
        # Filter by threshold
        mask = confidences >= threshold
        
        if mask.sum() == 0:
            print(f"Warning: No predictions above threshold {threshold}")
            continue
        
        filtered_preds = predictions[mask]
        filtered_labels = all_labels[mask]
        
        # Compute metrics on filtered predictions
        acc_metric = Accuracy(task="multiclass", num_classes=num_classes)
        
        prec_macro = Precision(task="multiclass", num_classes=num_classes, average="macro")
        prec_micro = Precision(task="multiclass", num_classes=num_classes, average="micro")
        
        recall_macro = Recall(task="multiclass", num_classes=num_classes, average="macro")
        recall_micro = Recall(task="multiclass", num_classes=num_classes, average="micro")
        
        f1_macro = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        f1_micro = F1Score(task="multiclass", num_classes=num_classes, average="micro")
        
        accuracy = acc_metric(filtered_preds, filtered_labels).item()
        precision_macro = prec_macro(filtered_preds, filtered_labels).item()
        precision_micro = prec_micro(filtered_preds, filtered_labels).item()
        recall_macro = recall_macro(filtered_preds, filtered_labels).item()
        recall_micro = recall_micro(filtered_preds, filtered_labels).item()
        f1_macro = f1_macro(filtered_preds, filtered_labels).item()
        f1_micro = f1_micro(filtered_preds, filtered_labels).item()
        
        coverage = mask.sum().item() / len(all_labels)
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_micro': precision_micro,
            'recall_macro': recall_macro,
            'recall_micro': recall_micro,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'coverage': coverage,
            'num_predictions': mask.sum().item()
        })
    
    return pd.DataFrame(results)


def evaluate_topk_accuracy(model, dataloader, num_classes, k_values=[1, 3, 5], custom_classes = None, device='cuda'):
    """
    Evaluate top-k accuracy for different k values
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        num_classes: Number of classes
        k_values: List of k values for top-k accuracy (e.g., [1, 3, 5])
        device: Device to run evaluation on
    
    Returns:
        DataFrame with top-k accuracy results
    """

    if custom_classes is None:
        custom_classes = [i for i in range(num_classes)]
    model.eval()
    
    # Collect all predictions first
    all_probs = []
    all_labels = []
    
    print("Collecting predictions for top-k evaluation...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            logits = model(images)[:, custom_classes]
            probs = torch.softmax(logits, dim=1)
            
            all_probs.append(probs.cpu())
            all_labels.append(labels)
    
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    results = []
    max_k = max(k_values)
    
    # Get top-k predictions (indices)
    _, top_k_indices = torch.topk(all_probs, k=min(max_k, num_classes), dim=1)
    
    print("\nComputing top-k accuracy...")
    for k in tqdm(k_values, desc="Top-K values"):
        if k > num_classes:
            print(f"Warning: k={k} is larger than num_classes={num_classes}, skipping")
            continue
        
        # Check if true label is in top-k predictions
        top_k_preds = top_k_indices[:, :k]
        correct = torch.any(top_k_preds == all_labels.unsqueeze(1), dim=1)
        
        accuracy = correct.float().mean().item()
        num_correct = correct.sum().item()
        total = len(all_labels)
        
        # Also compute average confidence for top-k predictions
        top_k_probs = torch.gather(all_probs, 1, top_k_preds)
        avg_confidence = top_k_probs.mean().item()
        
        results.append({
            'k': k,
            'top_k_accuracy': accuracy,
            'num_correct': num_correct,
            'total_samples': total,
            'avg_top_k_confidence': avg_confidence
        })
    
    return pd.DataFrame(results)