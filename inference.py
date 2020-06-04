from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from absl import app, flags

from modules.models import PPG2ECG


flags.DEFINE_string("weights", "", "model weights for inferencing")
FLAGS = flags.FLAGS


def main(argv):
    # prepare the parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare the model
    model = PPG2ECG(
        input_size=200,
        use_stn=True,
        use_attention=True).to(device)

    # load the model state
    model.load_state_dict(torch.load(Path(FLAGS.weights))["net"])
    model.eval()

    # prepare the inference data (PPG data)
    ppg = np.load(Path("example", "PPG.npy"))
    print("ppg shape: {}".format(ppg.shape))

    # run through the data
    idx = 0
    step = 100
    ecg = []
    while True:
        # out of range for ppg data
        if (idx+200) > len(ppg):
            break
        # preprocess the single data to match the input size
        input_data = ppg[idx:idx+200]

        # reshape the data to [1, 200]
        input_data = input_data.reshape((1, -1))

        # move ppg data to torch tensor and device
        input_data = torch.from_numpy(input_data).to(device).float()

        # inference
        # in torch, you need (batch, data) for forward
        # so you should unsqueeze the input data by unsqueeze(0)
        # now the input size should be [1, 1, 200]
        with torch.no_grad():
            output_data = model(input_data.unsqueeze(0))
            output_data = output_data["output"].cpu()
            ecg.append(output_data[0, 0])  # [1, 1, 200] -> [200,]
        idx += step

    # model performs better in middle [50:150] for whole output [0:200]
    # also we drop first 50 and last 50 for ppg to align the ppg and ecg
    ecg = [e[50:150] for e in ecg]
    ppg = ppg[50:-50]

    # show the plot
    ecg = torch.cat(ecg)
    plt.plot(ppg, label="ppg")
    plt.plot(ecg, label="ecg")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    app.run(main)
