import csv
import torch


def read_csv(file_path):
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        return list(reader)


def save_csv(save_data, save_file_path):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(save_data)


def get_fault_label(file_path):
    file_name = file_path.split("/")[-1]
    if "normal" in file_name:
        return 0
    elif "inner" in file_name:
        return 1
    elif "outer" in file_name:
        return 2
    elif "ball" in file_name:
        return 3
    else:
        return -1


def get_anomaly_label(file_path):
    file_name = file_path.split("/")[-1]
    if "normal" in file_name:
        return 0
    else:
        return 1


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
