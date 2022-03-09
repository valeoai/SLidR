import os
import torch
import torch.optim as optim
import pytorch_lightning as pl
from MinkowskiEngine import SparseTensor
from downstream.criterion import DownstreamLoss
from pytorch_lightning.utilities import rank_zero_only
from utils.metrics import confusion_matrix, compute_IoU_from_cmatrix


class LightningDownstream(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.best_mIoU = 0.0
        self.metrics = {"val mIoU": [], "val_loss": [], "train_loss": []}
        self._config = config
        self.train_losses = []
        self.val_losses = []
        self.ignore_index = config["ignore_index"]
        self.n_classes = config["model_n_out"]
        self.epoch = 0
        if config["loss"].lower() == "lovasz":
            self.criterion = DownstreamLoss(
                ignore_index=config["ignore_index"],
                device=self.device,
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                ignore_index=config["ignore_index"],
            )
        self.working_dir = os.path.join(config["working_dir"], config["datetime"])
        if os.environ.get("LOCAL_RANK", 0) == 0:
            os.makedirs(self.working_dir, exist_ok=True)

    def configure_optimizers(self):
        if self._config.get("lr_head", None) is not None:
            print("Use different learning rates between the head and trunk.")

            def is_final_head(key):
                return key.find('final.') != -1
            param_group_head = [
                param for key, param in self.model.named_parameters()
                if param.requires_grad and is_final_head(key)]
            param_group_trunk = [
                param for key, param in self.model.named_parameters()
                if param.requires_grad and (not is_final_head(key))]
            param_group_all = [
                param for key, param in self.model.named_parameters()
                if param.requires_grad]
            assert len(param_group_all) == (len(param_group_head) + len(param_group_trunk))

            weight_decay = self._config["weight_decay"]
            weight_decay_head = self._config["weight_decay_head"] if (self._config["weight_decay_head"] is not None) else weight_decay
            parameters = [
                {"params": iter(param_group_head), "lr": self._config["lr_head"], "weight_decay": weight_decay_head},
                {"params": iter(param_group_trunk)}]
            print(f"==> Head:  #{len(param_group_head)} params with learning rate: {self._config['lr_head']} and weight_decay: {weight_decay_head}")
            print(f"==> Trunk: #{len(param_group_trunk)} params with learning rate: {self._config['lr']} and weight_decay: {weight_decay}")

            optimizer = optim.SGD(
                parameters,
                lr=self._config["lr"],
                momentum=self._config["sgd_momentum"],
                dampening=self._config["sgd_dampening"],
                weight_decay=self._config["weight_decay"],
            )
        else:
            if self._config.get("optimizer") and self._config["optimizer"] == 'adam':
                print('Optimizer: AdamW')
                optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=self._config["lr"],
                    weight_decay=self._config["weight_decay"],
                )
            else:
                print('Optimizer: SGD')
                optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=self._config["lr"],
                    momentum=self._config["sgd_momentum"],
                    dampening=self._config["sgd_dampening"],
                    weight_decay=self._config["weight_decay"],
                )

        if self._config.get("scheduler") and self._config["scheduler"] == 'steplr':
            print('Scheduler: StepLR')
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, int(.9 * self._config["num_epochs"]),
            )
        else:
            print('Scheduler: Cosine')
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self._config["num_epochs"]
            )
        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        # set_to_none=True is a modest speed-up
        optimizer.zero_grad(set_to_none=True)

    def forward(self, x):
        return self.model(x).F

    def training_step(self, batch, batch_idx):
        if self._config["freeze_layers"]:
            self.model.eval()
        else:
            self.model.train()
        sparse_input = SparseTensor(batch["sinput_F"], batch["sinput_C"])
        output_points = self(sparse_input)

        loss = self.criterion(output_points, batch["labels"])
        # empty the cache to reduce the memory requirement: ME is known to slowly
        # filling the cache otherwise
        torch.cuda.empty_cache()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.train_losses.append(loss.detach().cpu())
        return loss

    def training_epoch_end(self, outputs):
        self.epoch += 1

    def validation_step(self, batch, batch_idx):
        sparse_input = SparseTensor(batch["sinput_F"], batch["sinput_C"])
        output_points = self.model(sparse_input).F

        loss = self.criterion(output_points, batch["labels"])
        self.val_losses.append(loss.detach().cpu())
        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )

        # Ensure we ignore the index 0
        # (probably not necessary after some training)
        output_points = output_points.softmax(1)
        if self.ignore_index is not None:
            output_points[:, self.ignore_index] = 0.0
        preds = []
        labels = []
        offset = 0
        output_points = output_points.argmax(1)
        for i, lb in enumerate(batch["len_batch"]):
            preds.append(output_points[batch["inverse_indexes"][i] + offset])
            labels.append(batch["evaluation_labels"][i])
            offset += lb
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        c_matrix = confusion_matrix(preds, labels, self.n_classes)
        return loss, c_matrix

    def validation_epoch_end(self, outputs):
        c_matrix = sum([o[1] for o in outputs])

        # remove the ignore_index from the confusion matrix
        c_matrix = torch.sum(self.all_gather(c_matrix), 0)

        m_IoU, fw_IoU, per_class_IoU = compute_IoU_from_cmatrix(
            c_matrix, self.ignore_index
        )

        self.train_losses = []
        self.val_losses = []
        self.log("m_IoU", m_IoU, prog_bar=True, logger=True, sync_dist=False)
        self.log("fw_IoU", fw_IoU, prog_bar=True, logger=True, sync_dist=False)
        if self.epoch == self._config["num_epochs"]:
            self.save()

    @rank_zero_only
    def save(self):
        path = os.path.join(self.working_dir, "model.pt")
        torch.save(
            {"model_points": self.model.state_dict(), "config": self._config}, path
        )
