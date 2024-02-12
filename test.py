import torch
from dataloaders.pcam_h5_dataloader import load_pcam_test


def test_model(image_path, clinical_data_path, model_path):
    # 1. Load model
    model = YourModel()
    model.load_state_dict(torch.load(model_path))
    model.eval() 

    # 2. Load image and clinical data (you'll need preprocessing steps here)
    image = load_and_preprocess_image(image_path)  
    clinical_data = load_and_preprocess_clinical(clinical_data_path) 

    # 3. Generate prediction
    with torch.no_grad():
        output = model(image, clinical_data)

    # 4. Calculate metrics
    ground_truth = load_ground_truth(clinical_data_path)  # Assuming it's in the clinical data
    metrics = calculate_metrics(ground_truth, output)

    return metrics



device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_loader, _ = load_pcam_test(
        dataset_root='/mnt/bigdata/datasets/camelyon-pcam/h5', 
        batch_size=48
        )

from torchvision.models import densenet161, DenseNet161_Weights

model = densenet161(
    weights=DenseNet161_Weights.IMAGENET1K_V1
)

model.fc = torch.nn.Linear(in_features=2208, out_features=2, bias=True)
model = model.to(device)


from sklearn.metrics import roc_auc_score, confusion_matrix, c_index

def calculate_metrics(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    balanced_accuracy = (sensitivity + specificity) / 2
    cindex = c_index(y_true, y_pred)

    return {'auc': auc, 'sensitivity': sensitivity, 
            'specificity': specificity, 'balanced_accuracy': balanced_accuracy,
            'cindex': cindex}

