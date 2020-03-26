import os
import numpy as np
import xml.etree.ElementTree as ET

from .dataset import DataSet
from ppdet.core.workspace import register, serializable

import logging
logger = logging.getLogger(__name__)


@register
@serializable
class InsectsDataSet(DataSet):
    def __init__(self, 
                 dataset_dir, 
                 anno_dir):
        '''
        Args:
            dataset_dir (str): 数据集根目录
            anno_dir (str): 标注文件目录
        '''
        super(InsectsDataSet, self).__init__(
            # image_dir=image_dir,
            # anno_path=anno_path,
            # dataset_dir=dataset_dir
            # sample_num=sample_num,
            # with_background=with_background
            )

        self.dataset_dir = dataset_dir
        self.anno_dir = anno_dir

        # `roidbs` is list of dict whose structure is:
        # {
        #     'im_file': im_fname, # image file name
        #     'im_id': img_id, # image id
        #     'h': im_h, # height of image
        #     'w': im_w, # width
        #     'is_crowd': is_crowd,
        #     'gt_score': gt_score,
        #     'gt_class': gt_class,
        #     'gt_bbox': gt_bbox,
        #     'gt_poly': gt_poly,
        # }
        self.roidbs = None
        # a dict used to map category name to class id
        self.cname2cid = None

    def load_roidb_and_cname2cid(self):

        cname2cid = self._get_cname2cid()

        records = self._get_annotation_records(cname2cid)

        self.roidbs, self.cname2cid = records, cname2cid

    def _get_cname2cid(self):
        """昆虫cname2cid
        return a dict, as following,
            {'Boerner': 0,
             'Leconte': 1,
             'Linnaeus': 2, 
             'acuminatus': 3,
             'armandi': 4,
             'coleoptera': 5,
             'linnaeus': 6
            }
        It can map the insect name into an integer label.
        """
        # 昆虫名称列表
        INSECT_NAMES = ['Boerner', 'Leconte', 'Linnaeus', 
                        'acuminatus', 'armandi', 'coleoptera', 'linnaeus']

        insect_category2id = {}
        for i, item in enumerate(INSECT_NAMES):
            insect_category2id[item] = i

        return insect_category2id

    def _get_annotation_records(self, cname2cid):
        '''昆虫records
        '''
        datadir = os.path.join(self.dataset_dir, self.anno_dir)

        filenames = os.listdir(os.path.join(datadir, 'annotations', 'xmls'))
        records = []
        ct = 0
        for fname in filenames:
            fid = fname.split('.')[0]
            fpath = os.path.join(datadir, 'annotations', 'xmls', fname)
            img_file = os.path.join(datadir, 'images', fid + '.jpeg')
            tree = ET.parse(fpath)

            if tree.find('id') is None:
                im_id = np.array([ct])
            else:
                im_id = np.array([int(tree.find('id').text)])

            objs = tree.findall('object')
            im_w = float(tree.find('size').find('width').text)
            im_h = float(tree.find('size').find('height').text)
            gt_bbox = np.zeros((len(objs), 4), dtype=np.float32)
            gt_class = np.zeros((len(objs), 1), dtype=np.int32)
            gt_score = np.zeros((len(objs), 1), dtype=np.float32)
            # is_crowd = np.zeros((len(objs), ), dtype=np.int32)
            difficult = np.zeros((len(objs), 1), dtype=np.int32)
            for i, obj in enumerate(objs):
                cname = obj.find('name').text
                gt_class[i][0] = cname2cid[cname]
                _difficult = int(obj.find('difficult').text)
                x1 = float(obj.find('bndbox').find('xmin').text)
                y1 = float(obj.find('bndbox').find('ymin').text)
                x2 = float(obj.find('bndbox').find('xmax').text)
                y2 = float(obj.find('bndbox').find('ymax').text)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(im_w - 1, x2)
                y2 = min(im_h - 1, y2)
                # 这里使用xywh格式来表示目标物体真实框
                # gt_bbox[i] = [(x1+x2)/2.0 , (y1+y2)/2.0, x2-x1+1., y2-y1+1.]
                gt_bbox[i] = [x1, y1, x2, y2]
                gt_score[i][0] = 1.
                # is_crowd[i][0] = 0
                difficult[i][0] = _difficult
            voc_rec = {
                'im_file': img_file,
                'im_id': im_id,
                'h': im_h,
                'w': im_w,
                'gt_class': gt_class,
                'gt_bbox': gt_bbox,
                'gt_score': gt_score,
                # 'is_crowd': is_crowd,
                'difficult': difficult
                }
            if len(objs) != 0:
                records.append(voc_rec)
            ct += 1
        return records