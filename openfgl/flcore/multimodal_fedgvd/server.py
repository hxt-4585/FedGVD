"""Train on cached synthetic modal graphs and return per-client logits."""
import torch

from openfgl.flcore.base import BaseServer
from openfgl.model.multimodal_utils import multimodal_supervised_loss
from .contrastive import get_coordinated_data
from .fedgvd_config import config
from .utils import graph_from_payload, graph_to_payload


class MultimodalFedGVDServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super().__init__(args, global_data, data_dir, message_pool, device)
        self.graph_cache = {}
        self.global_data = None
        self.client_slices = {}
        self.syn_logits = {}

    def execute(self):
        changed = False
        for cid in self.message_pool['sampled_clients']:
            message = self.message_pool.get(f'client_{cid}', {})
            if 'graph' in message:
                self.graph_cache[cid] = graph_from_payload(message['graph'], self.device)
                changed = True
        if not self.graph_cache:
            return
        if changed:
            self.global_data, self.client_slices = get_coordinated_data(
                self.graph_cache,
                cross_k=getattr(self.args, 'multimodal_cross_k', 2),
                similarity_threshold=getattr(self.args, 'multimodal_similarity_threshold', 0.0))
        data = self.global_data
        mask = torch.ones(data.num_nodes, dtype=torch.bool, device=self.device)
        self.task.model.train()
        for _ in range(getattr(self.args, 'multimodal_server_epochs', 10)):
            self.task.optim.zero_grad()
            _, logits, modal_logits, _ = self.task.model.forward_modalities(data)
            loss = multimodal_supervised_loss(
                modal_logits, logits, data.y, mask,
                modality_mask=data.modality_mask, branch_weight=config['branch_weight'])
            loss.backward()
            self.task.optim.step()
        self.task.model.eval()
        with torch.no_grad():
            _, logits, _, _ = self.task.model.forward_modalities(data)
        self.syn_logits = {cid: logits[part].detach().cpu().clone()
                           for cid, part in self.client_slices.items()}

    def send_message(self):
        graphs = {}
        for recipient in self.message_pool.get('sampled_clients', []):
            known = self.message_pool.get(f'client_{recipient}', {}).get('known_graph_ids', [])
            graphs[recipient] = {
                cid: graph_to_payload(graph)
                for cid, graph in self.graph_cache.items()
                if cid != recipient and cid not in known
            }
        self.message_pool['server'] = {'syn_logits': self.syn_logits, 'graphs': graphs}
