import torch
import torch.nn.functional as F

def calc_fixed_f1_iou(pred, target):
    pred=pred.unsqueeze(dim=0)
    target=target.squeeze().unsqueeze(dim=0)
    b, n, h, w = pred.size()
    bt, ht, wt = target.size()
    if h != ht or w != wt:
        pred = F.interpolate(pred, size=(ht, wt), mode="bilinear", align_corners=True)
    pred = torch.softmax(pred, dim=1)
    pred_labels = torch.argmax(pred, dim=1) 
    
    tp = torch.sum(pred_labels[target == 1] == 1).float() 
    fp = torch.sum(pred_labels[target == 0] == 1).float() 
    fn = torch.sum(pred_labels[target == 1] == 0).float()  

    precision = tp / (tp + fp + 1e-6) 
    recall = tp / (tp + fn + 1e-6)  

    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)  # F1值

    
    intersection = torch.sum(pred_labels[target == 1] == 1).float() 
    union = torch.sum(pred_labels + target >= 1).float()  

    iou = intersection / (union + 1e-6) 

    return f1_score, iou