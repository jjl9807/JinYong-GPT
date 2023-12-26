import argparse

import mindspore.common.dtype as mstype
import mindspore.communication.management as D
import mindspore.nn as nn
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import (
    CheckpointConfig,
    LossMonitor,
    ModelCheckpoint,
    TimeMonitor,
)
from mindspore.train.model import Model

from src.dataset import create_dataset
from src.gpt import GPT, GPTWithLoss
from src.utils import GPTConfig, LearningRate

SEQ_LEN = 128
VOCAB_SIZE = 3969  # 序列长度和词表大小根据选择的小说和预处理的结果调整


def run_train():
    """train function"""
    parser = argparse.ArgumentParser(description="GPT training")
    parser.add_argument(
        "--device_id", type=int, default=0, help="Device id, default is 0."
    )
    parser.add_argument(
        "--device_num", type=int, default=1, help="Use device nums, default is 1."
    )
    parser.add_argument(
        "--distribute",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Run distribute, default is false.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "lamb"],
        help="select which optimizer to be used, default adam",
    )
    parser.add_argument(
        "--epoch_size", type=int, default=2, help="Epoch size, default is 2."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./dataset",
        help="Data path of your MindRecord files.",
    )
    parser.add_argument(
        "--start_lr",
        type=float,
        default="5e-4",
        help="Start learning rate, default is 5e-4.",
    )
    parser.add_argument(
        "--end_lr",
        type=float,
        default="1e-6",
        help="End learning rate, default is 1e-6.",
    )
    parser.add_argument(
        "--sink_size",
        type=int,
        default=100,
        help="Sink size for every iteration, default is 100",
    )

    args_opt = parser.parse_args()

    context.set_context(
        mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=args_opt.device_id
    )
    if args_opt.distribute == "true":
        D.init()
        device_num = args_opt.device_num
        rank = device_id % device_num
        print("device_id is {}, rank_id is {}".format(device_id, rank))

        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )
    else:
        rank = 0
        device_num = 1

    config = GPTConfig(
        batch_size=256,
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
    gpt = GPT(config)
    gpt_with_loss = GPTWithLoss(gpt)

    ds = create_dataset(
        config.batch_size,
        data_path=args_opt.data_path,
        device_num=device_num,
        rank=rank,
    )

    epoch_num = args_opt.epoch_size
    step_per_epoch = ds.get_dataset_size()

    lr = LearningRate(
        learning_rate=args_opt.start_lr,
        end_learning_rate=args_opt.end_lr,
        warmup_steps=int(step_per_epoch * epoch_num * 0.1),
        decay_steps=epoch_num * step_per_epoch,
    )

    decay_filter = (
        lambda x: "layernorm" not in x.name.lower() and "bias" not in x.name.lower()
    )
    params = gpt.trainable_params()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [
        {"params": decay_params, "weight_decay": 1e-2},
        {"params": other_params, "weight_decay": 0.0},
        {"order_params": params},
    ]

    if args_opt.optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=lr)
    else:
        optimizer = nn.AdamWeightDecay(group_params, learning_rate=lr)

    callback_size = args_opt.sink_size
    actual_epoch_num = int(epoch_num * step_per_epoch / callback_size)
    callback = [TimeMonitor(callback_size), LossMonitor(callback_size)]

    config_ck = CheckpointConfig(
        save_checkpoint_steps=step_per_epoch, keep_checkpoint_max=1
    )
    ckpoint_cb = ModelCheckpoint(prefix="GPT", config=config_ck)
    callback.append(ckpoint_cb)


    gpt_with_loss.set_train(True)
    model = Model(gpt_with_loss, optimizer=optimizer)
    model.train(
        actual_epoch_num,
        ds,
        callbacks=callback,
        dataset_sink_mode=True,
        sink_size=callback_size,
    )


if __name__ == "__main__":
    set_seed(2023)
    run_train()
