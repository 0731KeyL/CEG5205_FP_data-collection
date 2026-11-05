"""Train the 84-feature gesture MLP, select on validation, then evaluate on test."""

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gesture_mlp import GestureMLP

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))
from Python.load_dataset import file_sha256, load_split


def tensors(split):
    # GestureMLP applies the saved feature normalization in forward().
    return (torch.from_numpy(split["features"]).float(),
            torch.from_numpy(split["y"]).long())


@torch.inference_mode()
def evaluate(model, data, criterion, device):
    model.eval()
    features, labels = data
    logits = model(features.to(device))
    loss = criterion(logits, labels.to(device)).item()
    predictions = logits.argmax(1).cpu()
    accuracy = (predictions == labels).float().mean().item()
    return loss, accuracy, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=SCRIPT_DIR.parent / "dataset_v004")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "runs")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not args.lr > 0:
        parser.error("epochs, batch size and learning rate must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train = load_split(args.dataset, "train")
    class_names = train["class_names"].tolist()
    features, labels = tensors(train)
    num_classes = len(class_names)
    counts = torch.bincount(labels, minlength=num_classes)
    if num_classes < 2 or len(counts) != num_classes or (counts == 0).any():
        raise ValueError("The training split must contain all configured gesture classes.")
    weights = len(labels) / (num_classes * counts.float())
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    batches = DataLoader(TensorDataset(features, labels), batch_size=args.batch_size,
                         shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    validation = tensors(load_split(args.dataset, "validation"))
    model = GestureMLP(args.dataset / "normalization.npz", num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    print(f"Device: {device}, training samples: {len(labels)}, validation: {len(validation[1])}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_weight = 0.0
        for x, y in batches:
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            batch_weight = weights[y].sum().item()
            total_loss += loss.item() * batch_weight
            total_weight += batch_weight

        val_loss, val_acc, _ = evaluate(model, validation, criterion, device)
        print(f"Epoch {epoch:3d}  train_loss={total_loss / total_weight:.4f}  "
              f"val_loss={val_loss:.4f}  val_accuracy={val_acc:.2%}")
        if val_loss < best_loss:
            best_state = deepcopy(model.state_dict())
            best_loss, best_epoch = val_loss, epoch

    if best_state is None:
        raise RuntimeError("No finite validation loss obtained.")
    model.load_state_dict(best_state)
    test = tensors(load_split(args.dataset, "test"))
    test_loss, test_acc, predictions = evaluate(model, test, criterion, device)
    print(f"Test (epoch {best_epoch}): loss={test_loss:.4f}, accuracy={test_acc:.2%}")
    print(classification_report(test[1], predictions, labels=list(range(num_classes)),
                                target_names=class_names, digits=3, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(test[1], predictions, labels=list(range(num_classes))))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / datetime.now().strftime("gesture_mlp_6class_%Y%m%d_%H%M%S_%f.pt")
    torch.save({"model_state_dict": model.cpu().state_dict(), "best_epoch": best_epoch,
                "best_validation_loss": best_loss, "test_accuracy": test_acc,
                "dataset_id": str(train["dataset_id"].item()), "class_names": class_names,
                "dataset_checksum": file_sha256(args.dataset / "checksums.sha256"),
                "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
                "batch_size": args.batch_size, "class_weights": weights}, path)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

