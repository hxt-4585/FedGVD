"""Local multimodal learning with synthetic-node logit distillation."""
import torch
import torch.nn.functional as F

from openfgl.flcore.base import BaseClient
from openfgl.model.multimodal_utils import multimodal_supervised_loss
from .SimGC_transductive import data_gc
from .fedgvd_config import config
from .utils import append_synthetic_data, graph_to_payload


class MultimodalFedGVDClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super().__init__(args, client_id, data, data_dir, message_pool, device)
        self.synthetic = None
        self.local_data_len = self.task.processed_data['data'].num_nodes
        self.local_trainings = 0
        self._upload_pending = False

    def execute(self):
        # Participation count is local: a late-arriving client still condenses first.
        if self.synthetic is None:
            self.synthetic = data_gc(self.task.processed_data, self.client_id,
                                     self.args, self.device)
            self._upload_pending = True
            return
        if self.local_trainings == 0:
            self.task.processed_data = append_synthetic_data(self.task.processed_data, self.synthetic)

        split = self.task.processed_data
        data = split['data']
        real_mask = split['train_mask'].clone()
        real_mask[self.local_data_len:] = False
        synthetic_mask = torch.zeros_like(real_mask)
        synthetic_mask[self.local_data_len:] = True
        targets = self.message_pool.get('server', {}).get('syn_logits', {}).get(self.client_id)
        self.task.model.train()
        for _ in range(self.args.num_epochs):
            self.task.optim.zero_grad()
            _, logits, modal_logits, _ = self.task.model.forward_modalities(data)
            loss = multimodal_supervised_loss(
                modal_logits, logits, data.y, real_mask,
                modality_mask=data.modality_mask, branch_weight=config['branch_weight'])
            loss = loss + config['gama'] * multimodal_supervised_loss(
                modal_logits, logits, data.y, synthetic_mask,
                modality_mask=data.modality_mask, branch_weight=config['branch_weight'])
            if self.local_trainings > 0 and targets is not None:
                loss = loss + config['lambda'] * F.mse_loss(
                    logits[self.local_data_len:], targets.detach().to(self.device))
            loss.backward()
            if getattr(self.task, 'step_preprocess', None) is not None:
                self.task.step_preprocess()
            self.task.optim.step()
        self.local_trainings += 1

    def send_message(self):
        payload = {'num_samples': self.local_data_len}
        if self._upload_pending:
            payload['graph'] = graph_to_payload(self.synthetic)
            self._upload_pending = False
        self.message_pool[f'client_{self.client_id}'] = payload
