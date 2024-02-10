from torch import randn, nn
from pathlib import Path
from models.nvidia_resnets.resnet import se_resnext101_32x4d
from tensorboardX import SummaryWriter

base_dir = Path("/mnt/bigdata/datasets")
tensorboard_model_dir = base_dir / Path('models')


from torchvision.models import densenet161, DenseNet161_Weights

model = densenet161(
    weights=DenseNet161_Weights.IMAGENET1K_V1
)

model.fc = nn.Linear(in_features=2208, out_features=2, bias=True)

dummy_input = randn(1, 3, 224, 224)
writer = SummaryWriter(tensorboard_model_dir  / Path('densenet161'))
writer.add_graph(model, dummy_input)
writer.close()