import torch


class PGLSModel(torch.nn.Module):
    def __init__(self, image_model, tabular_model):
        super(PGLSModel, self).__init__()
        self.image_model = image_model
        self.tabular_model = tabular_model
        self.fc = torch.nn.Linear(1000 + 100, 6)

    def forward(self, image, tabular):
        image_features = self.image_model(image)
        tabular_features = self.tabular_model(tabular)
        features = torch.cat((image_features, tabular_features), 1)
        return self.fc(features)


class SimpleTabularModel(torch.nn.Module):
    def __init__(self, input_data_len):
        super(SimpleTabularModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_data_len, input_data_len*4)
        self.fc2 = torch.nn.Linear(input_data_len*4, 100)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return x
