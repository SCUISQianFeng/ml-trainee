import torch


def get_predictions(loader, model, device):
    model.eval()
    saved_preds = []
    true_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            saved_preds += scores.tolist()
            true_labels += y.tolist()
    return saved_preds, true_labels



def get_submission(model, loader, test_ids, device):
    all_preds = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            print(x.shape)
            x = x.to(device)
            score = model(x)
            predictions = score.float()
            all_preds += predictions.tolist()