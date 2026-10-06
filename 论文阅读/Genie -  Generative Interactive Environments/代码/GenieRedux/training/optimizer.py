from torch.optim import AdamW, Adam, lr_scheduler


def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params


def get_optimizer(
    params,
    lr=1e-4,
    wd=1e-2,
    betas=(0.9, 0.99),
    eps=1e-8,
    filter_by_requires_grad=False,
    group_wd_params=True,
    **kwargs
):
    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))

    if wd == 0:
        return Adam(params, lr=lr, betas=betas, eps=eps)

    if group_wd_params:
        wd_params, no_wd_params = separate_weight_decayable_params(params)

        params = [
            {"params": wd_params},
            {"params": no_wd_params, "weight_decay": 0},
        ]

    return AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)


class LinearWarmup_CosineAnnealing:
    """Construct two LR schedulers.
    First one is for a linear warmup.
    Second one is for a cosine annealing."""

    @staticmethod
    def _sync_scheduler_state(scheduler, num_steps):
        """Align a torch scheduler with the number of completed steps without changing the LR."""

        if num_steps > 0:
            scheduler.last_epoch = num_steps - 1
        else:
            scheduler.last_epoch = -1

        if hasattr(scheduler, "_step_count"):
            scheduler._step_count = num_steps

        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr = [
                group["lr"] for group in scheduler.optimizer.param_groups
            ]

    def __init__(
        self,
        optimizer,  # optimizer to schedule
        linear_warmup_start_factor,
        linear_warmup_total_iters,  # linear warmup
        cosine_annealing_T_max,
        cosine_annealing_eta_min,  # cosine annealing
    ):

        self.scheduler_linear = lr_scheduler.LinearLR(
            optimizer,
            # The number we multiply learning rate in the first epoch
            start_factor=linear_warmup_start_factor,
            total_iters=linear_warmup_total_iters,
        )  # The number of iterations that multiplicative factor reaches to 1

        self.scheduler_cosine = lr_scheduler.CosineAnnealingLR(
            optimizer,
            # Maximum number of iterations.
            T_max=cosine_annealing_T_max,
            eta_min=cosine_annealing_eta_min,
        )  # Minimum learning rate.

        self.switch = linear_warmup_total_iters

    def step(self, nb_steps):

        if nb_steps <= self.switch:
            self.scheduler_linear.step()
        else:
            self.scheduler_cosine.step()

        return

    def set_step(self, completed_steps: int):
        """Advance internal scheduler counters to match previously completed steps."""

        if completed_steps is None:
            return

        completed_steps = max(int(completed_steps), 0)
        warmup_steps = min(completed_steps, self.switch)
        cosine_steps = max(completed_steps - self.switch, 0)

        self._sync_scheduler_state(self.scheduler_linear, warmup_steps)
        self._sync_scheduler_state(self.scheduler_cosine, cosine_steps)
