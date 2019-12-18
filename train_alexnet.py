import datasets
import model

PROJECT_DIR = "/home/vaibhavg/tf-keras-deep-head-pose"

DATA_DIR = '/home/vaibhavg/data/'

AFLW2000_DATA_DIR = DATA_DIR + 'AFLW2000/'
AFLW2000_MODEL_FILE = PROJECT_DIR + 'model/aflw2000_model.h5'
AFLW2000_TEST_SAVE_DIR = DATA_DIR + 'aflw2000_test/'

BIWI_DATA_DIR = DATA_DIR + 'Biwi/kinect_head_pose_db/hpdb/'
BIWI_MODEL_FILE = PROJECT_DIR + 'model/biwi_model.h5'
BIWI_TEST_SAVE_DIR = DATA_DIR + 'biwi_test/'

BIN_NUM = 66
INPUT_SIZE = 64
BATCH_SIZE=16
EPOCHS=20

dataset = datasets.Biwi(BIWI_DATA_DIR, 'filename_list.txt', batch_size=BATCH_SIZE, input_size=INPUT_SIZE, ratio=0.95)

net = model.AlexNet(dataset, BIN_NUM, batch_size=BATCH_SIZE, input_size=INPUT_SIZE)

net.train(BIWI_MODEL_FILE, max_epoches=EPOCHS, load_weight=False)

net.test(BIWI_TEST_SAVE_DIR)
