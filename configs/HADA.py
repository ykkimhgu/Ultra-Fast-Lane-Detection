# DATA
dataset='HADA'
data_root = '/home/hada/Ultra-Fast-Lane-Detection/HADAROOT'

# HADA
src_video = '/home/hada/Ultra-Fast-Lane-Detection/video/clips/BON.mp4'

# TRAIN
epoch = 120
batch_size = 32
optimizer = 'Adam'    #['SGD','Adam']
# learning_rate = 0.1
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'     #['multi', 'cos']
# steps = [50,75]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 100

# NETWORK
backbone = '18'
griding_num = 100
use_aux = True

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = '/home/hada/HADA_LOG_PATH'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = None
test_work_dir = None

num_lanes = 4
