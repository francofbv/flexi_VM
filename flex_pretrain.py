import datetime
import logging
import time
from os.path import join

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

#from dataset import MetaLoader, create_dataset, create_loader, create_sampler
from models import *
from tasks.retrieval_utils import evaluation_wrapper
from tasks.shared_utils import get_media_types, setup_model
from utils.basic_utils import (MetricLogger, SmoothedValue,
                               remove_files_if_exist, setup_seed)
from tasks.valley_dataloader import VLDL, CustomBatchSampler, multiple_samples_collate
from utils.config_utils import setup_main
from utils.distributed import get_rank, get_world_size, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def train(
    model,
    train_loader,
    optimizer,
    tokenizer,
    epoch,
    global_step,
    device,
    scheduler,
    scaler,
    config,
    data_type,
    writer
):
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=100, fmt="{value:.6f}"))
    metric_logger.add_meter("temperature", SmoothedValue(window=100, fmt="{value:.4f}"))
    loss_names = ["loss_" + k for k, v in config.criterion.loss_weight.items() if v != 0]

    #media_types = get_media_types(train_loaders)
    media_types = ['video']

    for name in loss_names:
        for m in media_types:
            metric_logger.add_meter(
                f"{m}-{name}", SmoothedValue(window=100, fmt="{value:.4f}")
            )

    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    '''
    if config.distributed:
        for d in train_loaders:
            d.sampler.set_epoch(epoch)
            '''
    #train_loader = MetaLoader(name2loader=dict(list(zip(media_types, train_loaders))))

    model_without_ddp = model.module if config.distributed else model
    iterator = metric_logger.log_every(train_loader, log_freq, header)
    for i, ((image, text, idx)) in enumerate(iterator):
        media_type = 'video'
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.inputs.max_txt_l[media_type],
            return_tensors="pt",
        ).to(
            device
        )  # change from "longest" to "max_length"

        with torch.cuda.amp.autocast(enabled=config.fp16, dtype=data_type):
            loss_dict = model(image, text_input, idx=idx)
            loss = sum(loss_dict.values())

        if not config.fp16 or config.get('bf16', True):
            optimizer.zero_grad()
            loss.backward()
            if config.optimizer.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
            optimizer.step()
            scheduler.step()
        else:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if config.optimizer.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # logging
        for name in loss_names:
            value = loss_dict[name]
            value = value if isinstance(value, float) else value.item()
            metric_logger.update(**{f"{media_type}-{name}": value})

            if writer:
                writer.add_scalar(f'{media_type}-{name}', value, global_step)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(temperature=model_without_ddp.temp.item())

        if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
            logs = metric_logger.get_global_avg_dict()
            log_dict_to_wandb(logs, step=global_step, prefix="train/")

        global_step += 1

        if config.debug and global_step % 20 == 0:
            logger.info("debug mode, break training loop")
            break

        if config.debug and global_step % (2 * log_freq + 3) == 0:
            logger.info("debug mode, break training loop")
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger.global_avg()}")
    if global_step  == 500:
        torch.save(model.state_dict(), f'7_26_VM_no_flex_finetune_valley_step_{global_step}_epoch{epoch}.pth')
    return global_step


def setup_dataloaders(config, mode="pt"):
    # train datasets, create a list of data loaders
    logger.info(f"Creating dataset for {mode}")
    #train_datasets = create_dataset(f"{mode}_train", config)
    #media_types = get_media_types(train_datasets)
    media_types = ['video']

    '''
    if config.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        samplers = create_sampler(
            train_datasets, [True] * len(media_types), num_tasks, global_rank
        )
    else:
        samplers = [None] * len(media_types)
        '''

    train_dataset = VLDL(data_split='train', num_frames=8, flexible=config.flexible)

    if config.distributed:
        sampler = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=False
            )
    else: 
        sampler = None

    '''
    train_loaders = create_loader(
        train_datasets,
        samplers,
        batch_size=[config.inputs.batch_size[k] for k in media_types],
        num_workers=[config.num_workers] * len(media_types),
        is_trains=[True] * len(media_types),
        collate_fns=[None] * len(media_types),
    )  # [0]
    '''

    batch_sampler = CustomBatchSampler(
            train_dataset.data,
            num_replicas=1,
            rank=get_rank(),
            shuffle=False
            )


    train_loader = DataLoader(
            train_dataset, 
            batch_sampler=batch_sampler,
            num_workers=config.num_workers, 
            pin_memory=False, 
            collate_fn=multiple_samples_collate,
            persistent_workers=True if config.num_workers > 0 else False,
     )


    return train_loader


def main(config):
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    if is_main_process():
        writer = SummaryWriter(log_dir='./TensorBoard_logs')
    else: writer = None

    is_pretrain = config.mode == "pt"

    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders  = setup_dataloaders(
        config, mode=config.mode
    )
    #num_steps_per_epoch = sum(len(d) for d in train_loaders)
    num_steps_per_epoch = len(train_loaders)
    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs
    # set cudnn.benchmark=True only when input size is fixed
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
    #cudnn.benchmark = len(train_media_types) == 1

    model_cls = eval(config.model.get('model_cls', 'UMT'))
    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model_cls=model_cls,
        has_decoder=False,
        pretrain=is_pretrain,
        find_unused_parameters=True,
    )
    if is_main_process() and config.wandb.enable:
        wandb.watch(model)

    best = 0
    best_epoch = 0

    if config.get('bf16', True):
        data_type = torch.bfloat16
    else:
        data_type = torch.float16

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, config.scheduler.epochs):
        if not config.evaluate:
            global_step = train(
                model,
                train_loaders,
                optimizer,
                tokenizer,
                epoch,
                global_step,
                device,
                scheduler,
                scaler,
                config,
                data_type,
                writer
            )

        if is_main_process():
            save_obj = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "epoch": epoch,
                "global_step": global_step,
            }
            torch.save(save_obj, f'./fine_tune_ckpt_epoch{epoch}.pth')
            if config.get("save_latest", False):
                torch.save(save_obj, join(config.output_dir, "ckpt_latest.pth"))
            else:
                torch.save(save_obj, join(config.output_dir, f"ckpt_{epoch:02d}.pth"))

        with torch.cuda.amp.autocast(enabled=config.fp16, dtype=data_type):
                '''
            eval_res = {}

            for test_name, test_loader in test_name2loaders.items():

                if test_name not in config.test_types:
                    logger.info(
                        f"Skip eval {test_name} split. All test_types {config.test_types}"
                    )
                    continue
                res = evaluation_wrapper(
                    model_without_ddp, test_loader, tokenizer, device, config, data_type=data_type, prefix=test_name
                )
                eval_res.update(res)

        if is_main_process():

            # log to wandb
            if config.wandb.enable:
                for p, v in eval_res.items():
                    log_dict_to_wandb(v, step=global_step, prefix=p)

            if config.stop_key is not None and config.stop_key in eval_res:
                if config.model.multimodal.enable:
                    cur_r_mean = eval_res[config.stop_key]["r_mean"]
                else:
                    cur_r_mean = eval_res[config.stop_key.replace("/", "_emb/")]["r_mean"]
            else:  # None
                cur_r_mean = best + 1  # save the last as the best

            eval_res = pd.DataFrame(eval_res)
            logger.info(f"Epoch {epoch}")
            logger.info(f"\n{eval_res.transpose().to_string(max_cols=30)}")

            eval_res.to_json(join(config.output_dir, "eval_res_latest.json"))

            if not config.evaluate and cur_r_mean > best:
                torch.save(save_obj, join(config.output_dir, "ckpt_best.pth"))
                eval_file = "eval_res_best.json"
                eval_res.to_json(join(config.output_dir, eval_file))
                best = cur_r_mean
                best_epoch = epoch

        if config.evaluate:
            break
        '''

        dist.barrier()

    if writer: writer.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"best epoch {best_epoch} [config.stop_key {config.stop_key}]")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()


if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
