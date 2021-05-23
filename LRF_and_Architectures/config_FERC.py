CLASSES = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

MIN_LR = 1e-5
MAX_LR = 1e-2
BATCH_SIZE = 64
STEP_SIZE = 8
CLR_METHOD = "triangular"
NUM_EPOCHS = 100

IMAGE_DIMS = (32, 32, 3)
INPUT_DATASET = 'FERC_sorted'
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

LRFIND_PLOT_PATH = "GoogLeNet_FERC_LRF_plot.png"
TRAINING_PLOT_PATH = "GoogLeNet_FERC_training_plot.png"
CLR_PLOT_PATH = "GoogLeNet_FERC_CLR_plot.png"
REPORT_PATH = "GoogLeNet_FERC_report"

LR_FIND = False
