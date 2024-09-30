import os
import soundfile
import numpy as np

import utils
from train import get_args


def preprocessing(args, file_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    file_list = os.listdir(file_dir)
    file_list.sort()

    for file in file_list:
        file_name = os.path.splitext(file)[0]
        vibration_csv = utils.read_csv(os.path.join(file_dir, file))
        vibration_data = np.array(vibration_csv[1:], dtype=np.float32)
        normalized_wav = vibration_data / np.max(np.abs(vibration_data))

        soundfile.write(
            os.path.join(save_dir, f"{file_name}.wav"), normalized_wav, args.sr
        )

    return 0


if __name__ == "__main__":
    args = get_args()

    train_dir = os.path.join(args.data_dir, "train")
    preprocessing(args, file_dir=train_dir, save_dir=args.train_dir)

    eval_dir = os.path.join(args.data_dir, "eval")
    preprocessing(args, file_dir=eval_dir, save_dir=args.eval_dir)

    test_dir = os.path.join(args.data_dir, "test")
    preprocessing(args, file_dir=test_dir, save_dir=args.test_dir)
