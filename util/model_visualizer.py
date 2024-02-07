from torch import randn
from pathlib import Path
from nvidia_resnets.resnet import (
    se_resnext101_32x4d,
)
from tensorboardX import SummaryWriter

base_dir = Path("/mnt/bigdata/placenta")
tensorboard_model_dir = base_dir / Path("tensorboard_data") / Path('models')


model = se_resnext101_32x4d(
    pretrained=True
)

dummy_input = randn(1, 3, 224, 224)
writer = SummaryWriter(tensorboard_model_dir  / Path('se_resnext101_32x4d'))
writer.add_graph(model, dummy_input)
writer.close()