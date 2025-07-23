import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import lightning as L


class Classifier(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer_config: dict,
    ):
        super().__init__()
        self.model = model
        self.optimizer_config = optimizer_config
        self.loss_fn = nn.CrossEntropyLoss()

        num_classes = self.model.num_classes
        # Ensure the nested model provides num_classes
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.save_hyperparameters(ignore=["model"])

    def configure_optimizers(self):
        """Configure optimizer based on config settings."""
        optimizer_type = self.optimizer_config.get("type", "adamw").lower()

        if optimizer_type == "adamw":
            return optim.AdamW(
                self.parameters(),
                lr=float(self.optimizer_config.get("learning_rate", 0.001)),
                weight_decay=float(self.optimizer_config.get("weight_decay", 0.0001)),
                betas=(
                    float(self.optimizer_config.get("beta1", 0.9)),
                    float(self.optimizer_config.get("beta2", 0.999)),
                ),
                eps=float(self.optimizer_config.get("eps", 0.00000001)),
            )
        elif optimizer_type == "adam":
            return optim.Adam(
                self.parameters(),
                lr=float(self.optimizer_config.get("learning_rate", 0.001)),
                weight_decay=float(self.optimizer_config.get("weight_decay", 0.0001)),
                betas=(
                    float(self.optimizer_config.get("beta1", 0.9)),
                    float(self.optimizer_config.get("beta2", 0.999)),
                ),
                eps=float(self.optimizer_config.get("eps", 0.00000001)),
            )
        elif optimizer_type == "sgd":
            return optim.SGD(
                self.parameters(),
                lr=float(self.optimizer_config.get("learning_rate", 0.01)),
                momentum=float(self.optimizer_config.get("momentum", 0.9)),
                weight_decay=float(self.optimizer_config.get("weight_decay", 0.0001)),
            )
        else:
            raise ValueError(f"❌ Unsupported optimizer type: {optimizer_type}")

    def forward(self, x):
        return self.model(x)

    def _classifier_step(self, batch):
        x, y = batch
        # Ensure inputs are on the correct device
        x = x.to(self.device)
        y = y.to(self.device)

        logits = self(x)
        loss = self.loss_fn(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        return loss, y_hat, y

    def training_step(self, batch, _):
        loss, y_hat, y = self._classifier_step(batch)
        self.train_accuracy(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train_accuracy",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, _):
        loss, y_hat, y = self._classifier_step(batch)
        self.val_accuracy(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, _):
        _, y_hat, y = self._classifier_step(batch)
        self.test_accuracy(y_hat, y)
        self.log(
            "test_accuracy",
            self.test_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
