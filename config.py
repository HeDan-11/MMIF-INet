# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = -4.5
lr = 10 ** log10_lr
epochs = 1000
weight_decay = 1e-5
init_scale = 0.01

lamda_reconstruction = 5
lamda_guide = 1
lamda_low_frequency = 1
device_ids = [0]

# Train:
batch_size = 4
cropsize = 128
betas = (0.5, 0.999)
weight_step = 1000
gamma = 0.5

# Val:
cropsize_val = 256
batchsize_val = 1
shuffle_val = False
val_freq = 10

data_root = "E:/python/pytorch/Medical_image_fusion/"
# Dataset
TRAIN_PATH = data_root + "data/data1/train3"
VAL_PATH = data_root + "data/TEST_IMGS/test"
format_train = 'jpg'
format_val = 'jpg'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False


# Saving checkpoints:

MODEL_PATH = data_root + 'MyMethod/MMIF_INet/model/'
checkpoint_on_error = True
SAVE_freq = 50

IMAGE_PATH = data_root +  'MyMethod/MMIF_INet/image/'
IMAGE_PATH_cover = IMAGE_PATH + 'cover-rev/'
IMAGE_PATH_s = IMAGE_PATH + 'steg/'
IMAGE_PATH_steg = IMAGE_PATH + 'steg-db/'
IMAGE_PATH_secret_rev = IMAGE_PATH + 'secret-rev/'

# Load:
suffix = 'model_checkpoint_00200.pt'
# suffix = 'model.pt'
tain_next = False
trained_epoch = 0
