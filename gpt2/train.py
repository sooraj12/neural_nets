import torch
import tiktoken
import math
import time
import os

from data_loader import DataLoader
from utils import get_device_details
from torch.distributed import destroy_process_group
from model_eval import render_example, iterate_examples
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.nn import functional as F
from model import GPT, GPTConfig


def train():
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    enc = tiktoken.get_encoding("gpt2")

    (
        ddp,
        device,
        device_type,
        ddp_world_size,
        master_process,
        ddp_local_rank,
        ddp_rank,
    ) = get_device_details

    total_batch_size = 524288
    # 16(default), 32, 64 for a100
    B = 16
    # 2048(gpt2)
    T = 1024
    train_loader = DataLoader(
        B=B,
        T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="train",
        master_process=master_process,
    )
    val_loader = DataLoader(
        B=B,
        T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="val",
        master_process=master_process,
    )

    assert (
        total_batch_size % (B * T * ddp_world_size) == 0
    ), "make sure total_batch_size is divisible by B*T"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

    if master_process:
        print(f"total desiered batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # torch.set_float32_matmul_precision("high") #TODO: enable on TF32 supported gpus

    # create model
    # model = GPT(GPTConfig(vocab_size=50304))
    # model.to(device)
    # # model = torch.compile(model)
    # if ddp:
    #     model = DDP(model, device_ids=[ddp_local_rank])

    # raw_model = model.module if ddp else model

    # load pretrained model weights(checkpoints)
    model_weights_path = os.path.join("log", "gpt2_v1.pt")
    checkpoint = torch.load(model_weights_path, map_location="cpu")
    csd = checkpoint["model"]
    model = GPT(GPTConfig(vocab_size=50304)).to(device)
    model.load_state_dict(csd, strict=False)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    max_steps = 19073

    def get_most_likely_row(tokens, mask, logits):
        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(
            flat_shift_logits, flat_shift_tokens, reduction="none"
        )
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (
            mask[..., 1:]
        ).contiguous()  # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred_norm = avg_loss.argmin().item()
        return pred_norm

    def get_lr(it):

        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps

        if it > max_steps:
            return min_lr

        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

        return min_lr + coeff * (max_lr - min_lr)

    optimizer = raw_model.configure_optimizer(
        weight_decay=0.1, learning_rate=max_lr, device_type=device_type
    )

    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f:  # open for writing to clear the file
        pass

    scaler = GradScaler()

    for step in range(checkpoint["step"], max_steps):
        t0 = time.time()
        last_step = step == max_steps - 1

        # evaluate model
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0
                val_loss_steps = 20

                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with autocast():
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()

                if ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

                if master_process:
                    print(f"Validation Loss: {val_loss_accum.item():.4f}")
                    with open(log_file, "a") as f:
                        f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                    if step > 0 and (step % 5000 == 0 or last_step):
                        # optionally write model checkpoints
                        checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                        checkpoint = {
                            "model": raw_model.state_dict(),
                            "config": raw_model.config,
                            "step": step,
                            "val_loss": val_loss_accum.item(),
                        }
                        torch.save(checkpoint, checkpoint_path)

        # once in a while evaluate hellaswag
        if step % 250 == 0 or last_step:
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                if i % ddp_world_size != ddp_rank:
                    continue

                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                with torch.no_grad():
                    with autocast():
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(
                    num_correct_norm, dtype=torch.long, device=device
                )
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(
                    f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}"
                )
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        # sample from model
        if (step > 0 and step % 250 == 0) or last_step:
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)

            while xgen.size(1) < max_length:
                with torch.no_grad():
                    with autocast():
                        logits, loss = model(xgen)
                    logits = logits[:, -1, :]  # (B, vocab_size)
                    # get the probabilities
                    probs = F.softmax(logits, dim=-1)
                    # do top-k sampling of 50 (huggingface pipeline default)
                    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    # select a token from the top-k probabilities
                    # note: multinomial does not demand the input to sum to 1
                    ix = torch.multinomial(
                        topk_probs, 1, generator=sample_rng
                    )  # (B, 1)
                    # gather the corresponding indices
                    xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                    # append to the sequence
                    xgen = torch.cat((xgen, xcol), dim=1)

            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):

            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # TODO: enable on bfloat16 supported gpus(ampere)
            # with torch.autocast(device_type=device, dtype=torch.bfloat16):
            # TODO: enable to use mixed pricision tensor calulation on volta
            # with torch.autocast():
            with autocast():
                logits, loss = model(x, y)
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()

            if ddp:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1

            # loss.backward()
            scaler.scale(loss).backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = (
            train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size / dt
        )
        if master_process:
            print(
                f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    train()
