import torch
from gofi.graphs.inversion_table.probs import PermModel
from gofi.graphs.inversion_table.loss import norm_loss_normalized
from gofi.graphs.graph import permutation_to_permutation_matrix, permutation_matrix_to_permutation
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler
from gofi.adameve.adamEVE import AdamEVE
from gofi.graphs.loss import LossGraphMatching
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

scaler = GradScaler('cuda') if torch.cuda.is_available() else None

def training(
    dist: PermModel,
    M1: torch.Tensor,
    M2: torch.Tensor,
    loss_function=norm_loss_normalized,
    eps=None,
    grad_eps = None,
    max_steps=None,
    adameve=False,
    adam_parameters=None,
    scheduler=None,
    scheduler_parameters=None,
    scheduler_input=None,
    grad_clipping=1.0,
    verbose=100,
    debug=False,
    store_relation_loss=False
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
    relation_losses = []

    if eps is None and max_steps is None:
        raise ValueError("Either eps or max_steps must be set.")
    # prepare optimiser
    if adam_parameters is None:
        adam_parameters = {"lr": 3e-4}
    
    if adameve:
        opt = AdamWithNoise(dist.model.parameters(), **adam_parameters)
    else:
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

            if store_relation_loss:
                # get permutation
                perm = dist.most_probable_permutation()
                p_matrix = permutation_to_permutation_matrix(perm)
                rloss = LossGraphMatching(p_matrix, M1, M2)
                relation_losses.append(rloss.item())

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

            if grad_eps is not None:
                # get gradient size
                total_gradient_norm = 0.0
                for p in dist.model.parameters():
                    if p.grad is not None:
                        total_gradient_norm += p.grad.data.norm(2).item() ** 2
                total_gradient_norm = total_gradient_norm ** 0.5
                if total_gradient_norm < grad_eps:
                    if verbose:
                        print(f"gradient norm {total_gradient_norm} < {grad_eps}. Stopping learning..")
                    break
            ### log
            if verbose and (step % verbose == 0 or step == 1):
                if grad_eps is None:
                    print(f"[step {step}] loss={loss_value:.6f} lr={opt.param_groups[0]['lr']:.2e}", end="\r")
                else: 
                    print(f"[step {step}] loss={loss_value:.6f} lr={opt.param_groups[0]['lr']:.2e} ||gradient||={total_gradient_norm:.6f}", end="\r")

            best = min(best, loss_value)
            if (eps is not None and best <= eps) or (max_steps is not None and step >= max_steps):
                break

    

    except KeyboardInterrupt:
        if verbose:
            print("Training interrupted by user.")
            print(f"[step {step}] loss={loss_value:.6f} lr={opt.param_groups[0]['lr']:.2e}")
        if store_relation_loss:
            return losses, relation_losses
        return losses
    except Exception as e:
        if verbose:
            print(f"[step {step}] loss={loss_value:.6f} lr={opt.param_groups[0]['lr']:.2e}")
        print(f"An error occurred during training at step {step}: {e}")
        traceback.print_exc()

        raise e    
    if verbose:
        print(f"[step {step}] loss={loss_value:.6f} lr={opt.param_groups[0]['lr']:.2e}")
        print(f"\nTraining completed in {step} steps. Best loss: {best:.6f}")
    
    if store_relation_loss:
        return losses, relation_losses
    return losses


