import argparse
import torch
from torch.nn import functional as F
from hdagentrec.backbone import SASRec
from hdagentrec.data import load_amazon_csv, temporal_splits

def pad(history, length): return [0] * max(0, length - len(history)) + history[-length:]

parser = argparse.ArgumentParser()
parser.add_argument("--interactions", required=True); parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--device", default="cuda:0"); parser.add_argument("--max-history", type=int, default=50)
args = parser.parse_args()
splits, _, items = temporal_splits(load_amazon_csv(args.interactions))
examples = [(sequence[:i], sequence[i]) for sequence in splits.train.values() for i in range(1, len(sequence))]
model = SASRec(len(items), args.max_history).to(args.device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for epoch in range(args.epochs):
    total = 0.0; model.train()
    for history, target in examples:
        logits = model(torch.tensor([pad(history, args.max_history)], device=args.device))
        loss = F.cross_entropy(logits, torch.tensor([target], device=args.device)); optimizer.zero_grad(); loss.backward(); optimizer.step(); total += loss.item()
    print({"epoch": epoch + 1, "loss": total / max(1, len(examples))})
torch.save({"state_dict": model.state_dict(), "num_items": len(items)}, "sasrec_phase1.pt")
