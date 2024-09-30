import os
import torch
import sklearn.metrics as metrics

import train
import net
import dataset
import utils


def eval(args):
    print("Evaluation started...")
    os.makedirs(args.result_dir, exist_ok=True)

    model = net.Autoencoder().cuda()
    model_path = os.path.join(args.model_dir, args.model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dataloader, file_list = dataset.get_eval_loader(args)
    criterion = torch.nn.MSELoss()

    score_list = [["File", "Score"]]

    fault_label = []
    y_true, y_pred = [], []

    for idx, data in enumerate(dataloader):
        log_mel = data[0].cuda()
        true_label = data[1].cuda()

        recon_log_mel = model(log_mel)

        loss = criterion(recon_log_mel, log_mel)

        fault_label.append(true_label.item())

        y_true.append(1 if data[1].item() > 0 else 0)
        y_pred.append(loss.item())

        file_name = os.path.splitext(file_list[idx].split("/")[-1])[0]

        score_list.append([file_name, loss.item()])

    auc = metrics.roc_auc_score(y_true, y_pred)
    print("AUC: ", auc)
    utils.save_csv(score_list, os.path.join(args.result_dir, "eval_score.csv"))

    fault_types = ["normal", "inner", "outer", "ball"]

    for fault in fault_types:
        if fault == "normal":
            continue
        else:
            fault_indices = [
                i
                for i, label in enumerate(fault_label)
                if (label == fault_types.index(fault) or label == 0)
            ]
            pred_labels = [y_pred[i] for i in fault_indices]
            true_labels = [y_true[i] for i in fault_indices]
            fault_auc = metrics.roc_auc_score(true_labels, pred_labels)
            print(f"{fault} AUC: {fault_auc}")


if __name__ == "__main__":
    args = train.get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    eval(args)
