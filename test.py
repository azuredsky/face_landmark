from torchstat import stat
import torchvision.models as models
from blazeface import BlazeFace
from facenet_pytorch import PNet,RNet,ONet

from train_config import config as cfg
from lib.helper.logger import logger
from lib.core.base_trainer.net_work import Train
from lib.dataset.dataietr import FaceKeypointDataIter
from lib.core.model.face_model import Net
logger.info('The trainer start')

#model=BlazeFace()
#model = models.shufflenet_v2_x0_5()

#model = ONet()
#stat(model, (3, 48, 48))

model = Net(num_classes=cfg.MODEL.out_channel)
stat(model, (3, 160, 160))
