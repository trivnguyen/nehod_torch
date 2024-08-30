
from ml_collections import config_dict

def get_config():

    config = config_dict.ConfigDict()

    config.seed = 10  # random seed
    config.workdir = 'example/workdir'  # where to save the model
    config.name = 'example-vdm-model'   # name of the model
    config.overwrite = False  # set to False to resume training
    config.checkpoint = None   # path to model checkpoint to resume training

    # Data args
    config.data = data = config_dict.ConfigDict()
    data.data_root = 'example/training_data'  # path to training directory
    data.data_name = 'example'  # name of the dataset in the training directory
    # output parameters of flows
    data.target_parameters = [
        "halo_mvir", "halo_mstar", "center_subhalo_mvir", "center_subhalo_mstar",
        "center_subhalo_vpeculiar", "center_subhalo_vmax_tilde", "log_num_subhalos",
    ]
    # conditioning parameters for flows
    data.conditioning_parameters = ["inv_wdm_mass", "log_sn1", "log_sn2", "log_agn1"]

    # Flows args
    # Flows hyperparameters
    config.flows = flows = config_dict.ConfigDict()
    flows.in_dim = len(data.target_parameters)
    flows.context_dim = len(data.conditioning_parameters)
    flows.hidden_dims = [32, 32, 32, 32]
    flows.projection_dims = [16, 16, 16, 16]
    flows.num_transforms = 4
    flows.dropout = 0.1

    # training and loss args
    # Training hyperparameters
    config.training = training = config_dict.ConfigDict()
    training.batch_size = 32
    training.max_steps = 50_000
    training.val_check_interval = 20
    training.log_every_n_steps = 10
    training.patience = 100

    # optimizer and scheduler args
    # gradient descent and learning rate scheduler hyperparameters
    config.optimizer = optimizer = config_dict.ConfigDict()
    optimizer.name = "AdamW"
    optimizer.lr = 5e-4
    optimizer.betas = [0.9, 0.999]
    optimizer.weight_decay = 0.01
    optimizer.grad_clip = 0.5
    config.scheduler = scheduler = config_dict.ConfigDict()
    scheduler.name = "WarmUpCosineAnnealingLR"
    scheduler.warmup_steps = 100
    scheduler.decay_steps = 1_000
    scheduler.eta_min = 1e-5

    return config
