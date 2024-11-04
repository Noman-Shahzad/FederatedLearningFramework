import torch

class Server:
    def __init__(self, global_model, device):
        self.global_model = global_model
        self.device = device

    def aggregate(self, client_updates):
        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.stack([client_update[key].float() for client_update in client_updates]).mean(0)
        self.global_model.load_state_dict(global_dict)
        return self.global_model.state_dict()

    def update_global_model(self, aggregated_params):
        self.global_model.load_state_dict(aggregated_params)
