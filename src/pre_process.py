"""
transform dataset to mindrecord.
"""

import argparse
import os
import pickle

import numpy as np
from mindspore.mindrecord import FileWriter
from tqdm.auto import tqdm

SEQ_LEN = 128
VOCAB_SIZE = 3969 # 序列长度和词表大小根据选择的小说和预处理的结果调整


class Tokenizer:
    def __init__(self, data: str, block_size: int):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print(f"the fiction data has {data_size} characters, {vocab_size} unique.")

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_sizes = block_size
        self.data = data

    def get_item(self):
        dix = [self.stoi[s] for s in self.data]
        for chunk in chunks(dix, self.block_sizes + 1):
            sample = {}
            if len(chunk) == self.block_sizes + 1:
                sample["input_ids"] = np.array(chunk, dtype=np.int32)
                yield sample


def chunks(lst, n):
    """yield n sized chunks from list"""
    for i in tqdm(range(len(lst) - n + 1)):
        yield lst[i : i + n]


def read_jinyong(file_path):
    """read Jin Yong fictions"""
    data = ""
    with open(os.path.join(file_path), "r", encoding="utf-8") as f:
        data += f.read().strip()

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file", type=str, default="./dataset/mindrecord"
    )
    parser.add_argument("--file_partition", type=int, default=1)
    parser.add_argument("--file_batch_size", type=int, default=512)
    parser.add_argument("--num_process", type=int, default=16)

    args = parser.parse_args()

    out_dir, out_file = os.path.split(os.path.abspath(args.output_file))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    mindrecord_schema = {
        "input_ids": {"type": "int32", "shape": [-1]},
    }

    data = read_jinyong("./data/倚天屠龙记.txt")
    tokenizer = Tokenizer(data, block_size=SEQ_LEN)
    with open("./dataset/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    transforms_count = 0
    wiki_writer = FileWriter(file_name=args.output_file, shard_num=args.file_partition)
    wiki_writer.add_schema(mindrecord_schema, "JinYong fictions")
    for x in tokenizer.get_item():
        transforms_count += 1
        wiki_writer.write_raw_data([x])
    wiki_writer.commit()
    print("Transformed {} records.".format(transforms_count))

    out_file = args.output_file
    if args.file_partition > 1:
        out_file += "0"
    print("Transform finished, output files refer: {}".format(out_file))
