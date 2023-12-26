"""
Create dataset for training and evaluating
"""

import os
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.common.dtype as mstype


def create_dataset(batch_size, data_path, device_num=1, rank=0, drop=True):
    """
    Create dataset

    Inputs:
        batch_size: batch size
        data_path: path of your MindRecord files
        device_num: total device number
        rank: current rank id
        drop: whether drop remainder

    Returns:
        dataset: the dataset for training or evaluating
    """
    home_path = os.path.join(os.getcwd(), data_path)
    data = [os.path.join(home_path, name) for name in os.listdir(data_path) if name.endswith("mindrecord")]
    print(data)
    dataset = ds.MindDataset(data, columns_list=["input_ids"], shuffle=True, num_shards=device_num, shard_id=rank)
    type_cast_op = C.TypeCast(mstype.int32)
    dataset = dataset.map(input_columns="input_ids", operations=type_cast_op)
    dataset = dataset.batch(batch_size, drop_remainder=drop)
    return dataset
