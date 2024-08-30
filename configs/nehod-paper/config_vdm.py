
from ml_collections import config_dict

def get_config():

    config = config_dict.ConfigDict()

    config.seed = 10
    config.workdir = '/mnt/ceph/users/tnguyen/nehod_torch/trained-models/nehod-paper/'
    config.overwrite = True
    config.name = 'vdm-model-debug-1'
    # config.checkpoint = None

    # Data args
    config.data = data = config_dict.ConfigDict()
    data.data_root = '/mnt/ceph/users/tnguyen/nehod_torch/point-cloud-diffusion-datasets/'\
        'processed_datasets/final-WDM-datasets'
    data.data_name = 'galprop'
    data.conditioning_parameters = [
        "halo_mvir", "halo_mstar", "center_subhalo_mvir", "center_subhalo_mstar",
        "center_subhalo_vpeculiar", "center_subhalo_vmax_tilde", "log_num_subhalos",
        "inv_wdm_mass", "log_sn1", "log_sn2", "log_agn1"
    ]

    # VDM args
    config.vdm = vdm = config_dict.ConfigDict()
    vdm.d_in = 9
    vdm.d_cond = len(data.conditioning_parameters)
    vdm.d_context_embedding = 16
    vdm.timesteps = 0
    vdm.antithetic_time_sampling = True
    vdm.use_encdec = False
    vdm.embed_context = True
    # score model args
    vdm.score_model = score = config_dict.ConfigDict()
    score.name = 'transformer'
    score.d_t_embedding = 16
    score.d_model = 256
    score.d_mlp = 512
    score.d_cond = vdm.d_context_embedding if vdm.embed_context else vdm.d_cond
    score.n_layers = 6
    score.n_heads = 4
    # noise schedule args
    vdm.noise_schedule = noise_schedule = config_dict.ConfigDict()
    noise_schedule.name = "learned_linear"
    noise_schedule.gamma_min = -16.0
    noise_schedule.gamma_max = 10.0

    # training and loss args
    config.training = training = config_dict.ConfigDict()
    training.batch_size = 128
    training.max_steps = 50_000
    training.noise_scale = 1e-3
    training.beta = 1.0
    training.rotation_augmentation = True
    training.n_pos_dim = 3
    training.n_vel_dim = 3
    training.add_mass_recon_loss = True
    training.i_mass_start = 6
    training.i_mass_stop = 8
    training.val_check_interval = 100
    training.log_every_n_steps = 50
    training.patience = 100

    # optimizer and scheduler args
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