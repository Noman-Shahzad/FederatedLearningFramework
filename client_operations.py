import torch
import torch.nn as nn
import torch.optim as optim
from client.distillation import DistillationLoss


class Client:
    def __init__(self, model, data_loader, device, teacher_model=None, temperature=3.0, alpha=0.5):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.teacher_model = teacher_model
        self.distillation_criterion = DistillationLoss(temperature, alpha)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_local_model(self, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

    def distill_knowledge(self):
        if self.teacher_model is None:
            raise ValueError("Teacher model is required for knowledge distillation.")

        self.model.train()
        self.teacher_model.eval()

        for data, labels in self.data_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            student_output = self.model(data)
            with torch.no_grad():
                teacher_output = self.teacher_model(data)
            loss = self.distillation_criterion(student_output, teacher_output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return self.model.state_dict()
