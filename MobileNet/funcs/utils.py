from sklearn.metrics import f1_score 
from sklearn.metrics import confusion_matrix
import os

def Get_F1Score(label, prediction, epoch, root):
    
    if not os.path.isdir(root):
        os.makedirs(root)
    
    label = [row.cpu() for row in label]
    prediction = [row.cpu() for row in prediction]

    F1Score = f1_score(label , prediction, average='macro')
    with open(f"{root}/f1scores.log", "a") as f:
        f.write(f"[epoch] {epoch}\n")
        f.write(f"[F1 score] {F1Score}\n")
        f.write(f"\n")
        f.close()

        
def Get_ConfusionMatrix(label, prediction):

    label = [row.cpu() for row in label]
    prediction = [row.cpu() for row in prediction]

    matrix = confusion_matrix(label, prediction)
    return matrix