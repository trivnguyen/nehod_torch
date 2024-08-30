
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
    data.conditioning_parameters = [
        "halo_mvir", "halo_mstar", "center_subhalo_mvir", "center_subhalo_mstar",
        "center_subhalo_vpeculiar", "center_subhalo_vmax_tilde", "log_num_subhalos",
        "inv_wdm_mass", "log_sn1", "log_sn2", "log_agn1"
    ]

    # VDM args
    # Hyperparameters for the entire diffusion model
    config.vdm = vdm = config_dict.ConfigDict()
    vdm.d_in = 9  # output dimension of the data
    vdm.d_cond = len(data.conditioning_parameters)  # dimension of the conditioning vector
    vdm.d_context_embedding = 16
    vdm.timesteps = 0   # set to 0 for continuous diffusion time
    vdm.antithetic_time_sampling = True
    vdm.use_encdec = False   # not implemented
    vdm.embed_context = True
    # score model args
    # Hyperparameters for the score model
    vdm.score_model = score = config_dict.ConfigDict()
    score.name = 'transformer'
    score.d_t_embedding = 16
    score.d_model = 256
    score.d_mlp = 512
    score.d_cond = vdm.d_context_embedding if vdm.embed_context else vdm.d_cond
    score.n_layers = 6
    score.n_heads = 4
    # noise schedule args
    # Hyperparameters for the noise schedule
    vdm.noise_schedule = noise_schedule = config_dict.ConfigDict()
    noise_schedule.name = "learned_linear"
    noise_schedule.gamma_min = -16.0
    noise_schedule.gamma_max = 10.0

    # training and loss args
    # Training hyperparameters
    config.training = training = config_dict.ConfigDict()
    training.batch_size = 128
    training.max_steps = 50_000
    training.noise_scale = 1e-3
    training.val_check_interval = 100
    training.log_every_n_steps = 50
    training.patience = 100
    training.beta = 1.0
    training.rotation_augmentation = True  # if True, augment data with random rotations
    training.n_pos_dim = 3  # number of positional dimensions
    training.n_vel_dim = 3  # number of velocity dimensions
    training.add_mass_recon_loss = True  # if True, add a total mass reconstruction loss
    training.i_mass_start = 6
    training.i_mass_stop = 8

    # optimizer and scheduler args
    # gradient descent and learning rate scheduler hyperparameters
    config.optimizer = optimizer = config_dict.ConfigDict()
    optimizer.name = "AdamW"
    optimizer.lr = 5e-4
    optimizer.betas = [0.9, 0.999]
    optimizer.weight_decay = 0.01
    optimizer.grad_clip = 0.5
    config.scheduler = scheduler = config_dict.ConfigDict()
    scheduler.name = "WarmUpCosineDecayLR"
    scheduler.init_value = 0.0
    scheduler.peak_value = optimizer.lr
    scheduler.warmup_steps = 5_000
    scheduler.decay_steps = training.max_steps

    return config