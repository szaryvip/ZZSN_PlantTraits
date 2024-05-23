import torch


class PGLSModelLateFusion(torch.nn.Module):
    def __init__(self, image_model, tabular_model, tabular_model_out_size=100):
        super(PGLSModelLateFusion, self).__init__()
        image_model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = image_model(dummy_input)
        image_features_number = output.shape[1]
        self.image_model = image_model
        self.tabular_model = tabular_model
        self.fc = torch.nn.Linear(image_features_number + tabular_model_out_size, 6)

    def forward(self, image, tabular):
        image_features = self.image_model(image)
        tabular_features = self.tabular_model(tabular)
        features = torch.cat((image_features, tabular_features), 1)
        return self.fc(features)


class PGLSModel(torch.nn.Module):
    def __init__(self, image_model, tabular_model):
        super(PGLSModel, self).__init__()
        image_model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = image_model(dummy_input)
        image_features_number = output.shape[1]
        self.image_model = image_model
        self.tabular_model = tabular_model
        self.fc = torch.nn.Linear(image_features_number + 100, 6)

    def forward(self, image, tabular):
        image_features = self.image_model(image)
        tabular_features = self.tabular_model(tabular)
        features = torch.cat((image_features, tabular_features), 1)
        return self.fc(features)


class SimpleTabularModel(torch.nn.Module):
    def __init__(self, input_data_len):
        super(SimpleTabularModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_data_len, input_data_len * 4)
        self.fc2 = torch.nn.Linear(input_data_len * 4, 100)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return x


class EnsemblePGLSModel(torch.nn.Module):
    def __init__(self, image_models, tabular_model):
        super(EnsemblePGLSModel, self).__init__()
        self.image_models = image_models
        self.tabular_model = tabular_model
        image_features_number = 0
        for image_model in self.image_models:
            image_model.eval()
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = image_model(dummy_input)
            image_features_number += output.shape[1]
        self.fc = torch.nn.Linear(image_features_number + 100, 6)

    def forward(self, image, tabular):
        image_features = [image_model(image) for image_model in self.image_models]
        tabular_features = self.tabular_model(tabular)
        features = torch.cat(image_features, 1)
        features = torch.cat((features, tabular_features), 1)
        return self.fc(features)
