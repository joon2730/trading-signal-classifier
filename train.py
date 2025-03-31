class Config:
    # Data parameters
    features =
    lookback = 48

    
    # Model parameters
    seq_len = lookback
    input_size = len(features)
    hidden_size = 128
    num_layers = 2
    dropout_rate = 0.2

    # Training parameters
    do_train = True
    do_predict = True
    add_train = False           # Load existing model for incremental training
    shuffle_train_data = True   # Shuffle training data
    use_cuda = False            # Use GPU training

    train_data_rate = 0.95
    valid_data_rate = 0.15

    batch_size = 64
    learning_rate = 0.001
    epoch = 20
    patience = 5                # Early stopping patience
    random_seed = 42

    do_continue_train = False   # Use final_state of last sample as init_state of next (only for RNNs in PyTorch)
    continue_flag = ""
    if do_continue_train:
        shuffle_train_data = False
        batch_size = 1
        continue_flag = "continue_"

    # Debug mode
    # debug_mode = False
    # debug_num = 500  # Use only N rows in debug mode

    # Framework settings
    model_name = "model_" + continue_flag + ".pth"

    # Paths
    model_save_path = "./checkpoint/"
    figure_save_path = "./figure/"
    log_save_path = "./log/"
    do_log_print_to_screen = True
    do_log_save_to_file = True
    do_figure_save = True
    do_train_visualized = False  # Training loss visualization (visdom for PyTorch)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if do_train and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
        if fetch_data_online:
            log_save_filename = ticker + '_' + interval + '_' + \
                datetime.strptime(start_date, "%Y-%m-%d").strftime("%y-%m-%d") + '_to_' + \
                datetime.strptime(end_date, "%Y-%m-%d").strftime("%y-%m-%d") + '.log'
        else:
            log_save_filename = train_data_path.split(".")[0].split("/")[-1] + ".log"
        log_save_path = log_save_path + cur_time + '_' + "/"
        os.makedirs(log_save_path)