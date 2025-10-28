import torch
from gofi.graphs.inversion_table.probs import PermModel
from gofi.graphs.inversion_table.loss import norm_loss_normalized
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

scaler = GradScaler('cuda') if torch.cuda.is_available() else None

def training(
    dist: PermModel,
    M1: torch.Tensor,
    M2: torch.Tensor,
    loss_function=norm_loss_normalized,
    eps=None,
    max_steps=None,
    adam_parameters=None,
    scheduler=None,
    scheduler_parameters=None,
    scheduler_input=None,
    grad_clipping=1.0,
    verbose=100,
    debug=False
):
    """
    Trains a PermModel dist on graphs, given with adjecency matrices M1 and M2. 

    Parameters
    ----------
    dist : PermModel
        model of the distribution on S_n
    
    eps : 
    
    """
    losses = []

    if eps is None and max_steps is None:
        raise ValueError("Either eps or max_steps must be set.")
    # prepare optimiser
    if adam_parameters is None:
        adam_parameters = {"lr": 3e-4}
    opt = Adam(dist.model.parameters(), **adam_parameters)

    sch = scheduler(opt, **(scheduler_parameters or {})) if scheduler is not None else None

    step = 0
    best = float("inf")

    def debug_log(msg):
        if debug:
            print(msg)

    try: 

        while True:
            step += 1
            opt.zero_grad(set_to_none=True)
            dist.clear_cache()  # critical: drop all cached tensors before forward

            # compute loss
            if scaler is None:
                loss = loss_function(dist, M1, M2)
            else:
                with autocast(device_type='cuda', dtype=torch.float16):
                    loss = loss_function(dist, M1, M2)

            loss_value = float(loss.detach().item())  # detach to plain float
            losses.append(loss_value)

            # backward + clip + step
            if scaler is None:
                loss.backward()
                if grad_clipping is not None:
                    clip_grad_norm_(dist.model.parameters(), max_norm=grad_clipping)
                opt.step()
            else:
                scaler.scale(loss).backward()
                if grad_clipping is not None:
                    scaler.unscale_(opt)  # unscale before clipping
                    clip_grad_norm_(dist.model.parameters(), max_norm=grad_clipping)
                scaler.step(opt)
                scaler.update()

            # scheduler
            if sch is not None:
                if scheduler_input == "loss":
                    sch.step(loss_value)  # pass float, not Tensor
                else:
                    try:
                        sch.step(step)
                    except TypeError:
                        sch.step()

            ### log
            if verbose and (step % verbose == 0 or step == 1):
                print(f"[step {step}] loss={loss_value:.6f} lr={opt.param_groups[0]['lr']:.2e}", end="\r")

            best = min(best, loss_value)
            if (eps is not None and best <= eps) or (max_steps is not None and step >= max_steps):
                break

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        return losses
    except Exception as e:
        print(f"An error occurred during training at step {step}: {e}")
        raise e    
        
    print(f"\nTraining completed in {step} steps. Best loss: {best:.6f}")

    return losses


