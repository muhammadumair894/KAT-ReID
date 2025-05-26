import torch


def make_optimizer(cfg, model, center_criterion):
    coeff_params, base_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # ------ rational coeffs get 0.1 × LR ------
        if "act1" in name or "act2" in name or ".act." in name:
            coeff_params.append(p)
            continue

        # ------ everything else (original logic) ------
        lr = cfg.SOLVER.BASE_LR
        wd = cfg.SOLVER.WEIGHT_DECAY

        if "bias" in name:
            lr *= cfg.SOLVER.BIAS_LR_FACTOR
            wd  = cfg.SOLVER.WEIGHT_DECAY_BIAS

        if cfg.SOLVER.LARGE_FC_LR and ("classifier" in name or "arcface" in name):
            lr *= 2
            print("Using two times learning rate for fc")

        base_params.append({"params": [p], "lr": lr, "weight_decay": wd})

    # add coeff param-group with scaled LR
    if coeff_params:
        base_params.append(
            {"params": coeff_params,
             "lr": cfg.SOLVER.BASE_LR * 0.1,
             "weight_decay": cfg.SOLVER.WEIGHT_DECAY}
        )

    if cfg.SOLVER.OPTIMIZER_NAME == "SGD":
        optimizer = torch.optim.SGD(
            base_params, momentum=cfg.SOLVER.MOMENTUM
        )
    elif cfg.SOLVER.OPTIMIZER_NAME == "AdamW":
        optimizer = torch.optim.AdamW(
            base_params, betas=(0.9, 0.999)
        )
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(base_params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center
