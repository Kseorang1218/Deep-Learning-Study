import torch

class DeepSVDD:
    def __init__(self, objective: str = 'one-class', nu: float = 0.1):
        self.objective = objective
        self.nu = nu
        self.R = 0.0  # hypersphere radius R
        self.c = None  # hypersphere center c

        self.net = None  # neural network \phi

        self.trainer = None

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None

    def set_network(self, net):
        self.net = net

    def pretrain(self, ae_net, ae_trainer, train_loader, eval_loader, 
                 latent_size, csv_name=None, csv_root=None, save_result=True):
        self.ae_net = ae_net
        self.ae_trainer = ae_trainer
        self.ae_net =  self.ae_trainer.train(self.ae_net, train_loader)
        self.ae_trainer.eval(self.ae_net, eval_loader, latent_size, 
                                    save_result, csv_name, csv_root)
        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def train(self, trainer, train_loader):
        self.trainer = trainer
        self.net, train_loss_list = self.trainer.train(self.net, train_loader)
        self.R = float(self.trainer.R.cpu().data.numpy())  # get float
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get list

        return train_loss_list

    def eval(self, trainer, eval_loader, latent_size, 
             csv_name, csv_root, save_result=True):
        if self.trainer is None:
            self.trainer = trainer
        latent_vectors, fault_label_list, y_pred = self.trainer.eval(self.net, eval_loader, latent_size,
                                                             save_result, csv_name, csv_root)

        return latent_vectors, fault_label_list, y_pred

    def load_model(self, ae_net, model_path, load_ae=False):
        """Load Deep SVDD model from model_path."""
        model_dict = torch.load(model_path)

        self.R = model_dict['R']
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])
        if load_ae:
            if self.ae_net is None:
                self.ae_net = ae_net
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])
        
    def save_model(self, export_model, save_ae=True):
        """Save Deep SVDD model to export_model."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None

        torch.save({'R': self.R,
                    'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)

