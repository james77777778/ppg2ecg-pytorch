from absl import app, flags

from modules.data import datasets
from modules.trainer import Trainer


# run
flags.DEFINE_string("run_name", "", "run name for training")
flags.DEFINE_string("logdir", "logs", 'log directory, model will be saved to '
                                      '<log-dir>/<run-name>_<localtime>')
# dataset
flags.DEFINE_enum("dataset", "UQVIT", datasets.keys(), 'specify dataset')
# training
flags.DEFINE_float("lr", 1e-4, '')
flags.DEFINE_integer("batch_size", 256, '')
flags.DEFINE_integer("epoch", 300, '')
flags.DEFINE_bool("lr_sched", False, 'whether to use cosine annealing lr '
                                     'sheduler')
flags.DEFINE_bool("aug", True, 'whether to turn off data augmentation')
# testing
flags.DEFINE_integer("eval_step", 5, '')
flags.DEFINE_integer("save_step", 50, '')
flags.DEFINE_string("seed", "2019", 'set the seed for randomness')
# model
flags.DEFINE_integer("input_size", 200, 'model input size')
flags.DEFINE_bool("conv", True, 'whether to use Conv Network')
flags.DEFINE_bool("stn", True, 'whether to use Sequential Transform Network')
flags.DEFINE_bool("attn", True, 'whether to use Attention Network')
flags.DEFINE_bool("lstm", False, 'whether to use LSTM Network')
# QRS-enhanced loss
flags.DEFINE_bool("qrsloss", True, 'whether to use QRS enhanced loss')
flags.DEFINE_float("qrs_beta", 0.5, '')
flags.DEFINE_float("qrs_sigma", 1., '')
FLAGS = flags.FLAGS


def main(argv):
    trainer = Trainer()
    trainer.run()


if __name__ == "__main__":
    app.run(main)
    print("Finish Training!")
