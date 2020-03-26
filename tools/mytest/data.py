
import sys
sys.path.append('/paddle/PaddleDetection')
sys.path.append('.')

from ppdet.core.workspace import load_config, merge_config, create
from ppdet.data.reader import create_reader

yml_file = 'configs/insects/yolov3_darknet_insects.yml'
# yml_file = 'configs/yolov3_mobilenet_v1_fruit.yml'
cfg = load_config(yml_file)

# data = cfg.TrainReader['dataset']

# cname2id = data._get_cname2cid()
# print(cname2id)
# records = data._get_annotation_records(cname2id)
# print(records)
# exit()

# print(cfg)
# exit()

# train_reader = create_reader(cfg.TrainReader)
eval_reader = create_reader(cfg.EvalReader)

# batch = next(train_reader())
batch = next(eval_reader())
one_smaple = batch[0]
print('batch_size: ', len(batch))
print('one_smaple: ')
print(one_smaple)
# print(type(batch))
# print(len(batch))
# print(type(batch[0]))
# print(len(batch[0]))
print(one_smaple[0].shape)
# print(batch[0][1])
# print(batch[0][2])