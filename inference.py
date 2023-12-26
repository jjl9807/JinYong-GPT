import argparse
import pickle

import numpy as np
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.gpt import GPT
from src.pre_process import Tokenizer
from src.utils import GPTConfig

SEQ_LEN = 128
VOCAB_SIZE = 3969  # 序列长度和词表大小根据选择的小说和预处理的结果调整

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=0)


def generate(prompt: str, tokenizer: Tokenizer, gpt: GPT, max_length=2000):
    """
    Text generation
    """
    TOPK = 10
    input_ids = [tokenizer.stoi[ch] for ch in prompt]

    valid_length = len(input_ids)
    while valid_length < max_length:
        if len(input_ids) > SEQ_LEN:
            input_ids = input_ids[-SEQ_LEN:]
        inputs = Tensor(np.array([input_ids], dtype=np.int32))
        logits = gpt(inputs)[0].asnumpy()
        probs = logits[-1, :]
        p_args = probs.argsort()[::-1][:TOPK]

        p = probs[p_args]
        # p = np.exp(p) / np.sum(np.exp(p))
        p = np.exp(p - np.max(p)) / np.sum(np.exp(p - np.max(p)))
        target_index = np.random.choice(len(p), p=p)

        prod = int(p_args[target_index])

        input_ids.append(prod)
        print(tokenizer.itos[prod], end="", flush=True)

        valid_length += 1
    print("")


def continuation(tokenizer: Tokenizer, gpt: GPT, max_length=1024):
    """Using GPT for fiction continuation.

    Args:
        gpt (nn.Cell): GPT model
        max_length(int): max generating length
    """
    print(
        'Continuing the text in the style of Jin Yong\'s novels. Press "Ctrl+D" to'
        " exit."
    )
    while True:
        try:
            print("输入一个开头：", end="")
            prompt = input()

            generate(prompt, tokenizer, gpt)

        except EOFError:
            print("\nBye!")
            break


def main():
    parser = argparse.ArgumentParser(description="GPT inferencing")
    parser.add_argument(
        "--task_type", type=str, default="continuation", help="Evaluation task."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./GPT.ckpt",
        help="path of checkpoint file.",
    )

    args = parser.parse_args()
    task = args.task_type
    ckpt_path = args.ckpt_path

    config = GPTConfig(
        batch_size=1,
        seq_length=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        embedding_size=512,
        num_layers=8,
        num_heads=8,
        expand_ratio=4,
        post_layernorm_residual=False,
        dropout_rate=0.0,
        use_past=False,
    )
    ckpt_dict = load_checkpoint(ckpt_path)

    gpt = GPT(config)

    gpt.set_train(False)
    load_param_into_net(gpt, ckpt_dict)

    with open("./dataset/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    if task == "continuation":
        continuation(tokenizer=tokenizer, gpt=gpt)


if __name__ == "__main__":
    main()
