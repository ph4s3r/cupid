import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# plot a batch of tiles with masks
def vizBatch(im, tile_masks, tile_labels = None):
    # create a grid of subplots
    _, axes = plt.subplots(4, 4, figsize=(8, 8))  # Adjust figsize as needed
    axes = axes.flatten()

    for i in range(8):  # Display only the first 8 tiles, duplicated
        img = im[i].permute(1, 2, 0).numpy()
        mask = tile_masks[i].permute(1, 2, 0).numpy()

        # Display image without mask
        axes[2*i].imshow(img)
        axes[2*i].axis("off")
        if tile_labels is not None:
            label = ", ".join([f"{v[i]}" for _, v in tile_labels.items()])
        axes[2*i].text(3, 10, label, color="white", fontsize=6, backgroundcolor="black")

        # Display image with mask overlay
        axes[2*i + 1].imshow(img)
        axes[2*i + 1].imshow(mask, alpha=1, cmap='terrain')  # adjust alpha as needed
        axes[2*i + 1].axis("off")

    plt.tight_layout()
    plt.show()

def test_model(test_loader, model_path, device, model):

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    true_labels = []
    predictions = []

    model.eval()

    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            outputs.shape
            outputs.data.shape
            probabilities = torch.sigmoid(outputs)  # the senet does not have a sigmoid output
            predictions.extend(probabilities[:, 1].cpu().numpy()) # getting back the probs for both class, roc_curve needs only the positive 
            true_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy: {:.2f}%'.format(100 * correct / total))

    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)

    # plot roc / auc
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    