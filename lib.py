import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.tensorboard import SummaryWriter # launch with http://localhost:6006/

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

def test_model(test_loader, model_path, device, model, session_name = None) -> dict():

    writer = None
    if session_name is not None:
        writer = SummaryWriter(log_dir=f"G:\\pcam\\tensorboard_data\\{session_name}\\", comment="test-results")

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

    accuracy = 100 * correct / total
    print('Accuracy: {:.2f}%'.format(100 * correct / total))
    writer.add_text("Accuracy", str(accuracy), global_step=None, walltime=None)

    false_negatives = {}

    for i, positive_index in enumerate(true_labels):
        if positive_index == 1 and predictions[i] < 0.3:
            false_negatives[predictions[i]] = i # dict with keys as the prediction scores and values as indexes of images (sorting is easy) 

    worst_fns = sorted(false_negatives.items(), key=lambda item: item[0])

    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)

    # Ensure thresholds are within the expected range
    thresholds = np.clip(thresholds, 0, 1)

    # Normalize the threshold values
    norm = plt.Normalize(vmin=thresholds.min(), vmax=thresholds.max())
    cmap = plt.cm.viridis

    # Create figure and axis
    fig, ax = plt.subplots()

    # Plot each segment of the ROC curve with color mapping to the thresholds
    for i in range(len(fpr) - 1):
        color = cmap(norm(thresholds[i]))
        ax.plot(fpr[i:i+2], tpr[i:i+2], color=color, lw=2)

    # Adding the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Important to ensure the colorbar works with our custom colors
    fig.colorbar(sm, ax=ax, label='Threshold')

    # Plotting the diagonal line for random chance
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Customize the plot
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic with Thresholds')
    ax.legend(['ROC curve (area = {:.2f})'.format(roc_auc)], loc="lower right")

    plt.show()
    writer.add_figure("ROC/AUC fig", fig, global_step=None, close=True, walltime=None)
    
    writer.add_graph(model, images)
    writer.close()

    return worst_fns