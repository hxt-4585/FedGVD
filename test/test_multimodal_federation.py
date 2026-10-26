import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from openfgl.flcore.multimodal_fedgvd.client import MultimodalFedGVDClient
from openfgl.flcore.multimodal_fedgvd.server import MultimodalFedGVDServer


def graph():
    return Data(x=torch.randn(3, 4), x_text=torch.randn(3, 2),
                x_image=torch.randn(3, 2), y=torch.tensor([0, 1, 0]),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
                edge_weight=torch.ones(2), modality_mask=torch.ones(3, 2, dtype=torch.bool))


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward_modalities(self, data):
        logits = self.linear(data.x)
        return data.x, logits, torch.stack((logits, logits), 1), logits.new_full((len(logits), 2), .5)


def task():
    data = graph()
    model = ToyModel()
    split = {'data': data, 'train_mask': torch.tensor([True, True, False]),
             'val_mask': torch.tensor([False, False, True]), 'test_mask': torch.zeros(3, dtype=torch.bool)}
    return SimpleNamespace(model=model, optim=torch.optim.SGD(model.parameters(), lr=.1),
                           processed_data=split, splitted_data=split, data=data, num_samples=3)


class FederationTests(unittest.TestCase):
    def test_client_participation_and_evaluation_isolation(self):
        pool = {'round': 12, 'server': {'syn_logits': {4: torch.zeros(3, 2)}}}
        local_task = task()
        original = local_task.splitted_data
        def initialize(obj, args, client_id, data, data_dir, message_pool, device):
            obj.args, obj.client_id, obj.message_pool, obj.device, obj.task = args, client_id, message_pool, device, local_task
        with patch('openfgl.flcore.base.BaseClient.__init__', initialize):
            client = MultimodalFedGVDClient(SimpleNamespace(num_epochs=1), 4, None, '', pool, 'cpu')
        with patch('openfgl.flcore.multimodal_fedgvd.client.data_gc', return_value=graph()) as condense:
            with patch('openfgl.flcore.multimodal_fedgvd.client.F.mse_loss', wraps=torch.nn.functional.mse_loss) as mse:
                client.execute()
                mse.assert_not_called()
            client.send_message()
            self.assertIn('graph', pool['client_4'])
            before = local_task.model.linear.weight.detach().clone()
            with patch('openfgl.flcore.multimodal_fedgvd.client.F.mse_loss', wraps=torch.nn.functional.mse_loss) as mse:
                client.execute()
                mse.assert_not_called()
            client.send_message()
            self.assertNotIn('graph', pool['client_4'])
            self.assertFalse(torch.equal(before, local_task.model.linear.weight))
            with patch('openfgl.flcore.multimodal_fedgvd.client.F.mse_loss', wraps=torch.nn.functional.mse_loss) as mse:
                client.execute()
                mse.assert_called_once()
            self.assertEqual(condense.call_count, 1)
        self.assertIs(local_task.splitted_data, original)
        self.assertEqual(original['data'].num_nodes, 3)
        self.assertEqual(local_task.processed_data['data'].num_nodes, 6)

    def test_server_caches_clients_and_returns_detached_slices(self):
        from openfgl.flcore.multimodal_fedgvd.utils import graph_to_payload
        pool = {'sampled_clients': [7], 'client_7': {'graph': graph_to_payload(graph())}}
        server_task = task()
        def initialize(obj, args, data, data_dir, message_pool, device):
            obj.args, obj.message_pool, obj.device, obj.task = args, message_pool, device, server_task
        with patch('openfgl.flcore.base.BaseServer.__init__', initialize):
            server = MultimodalFedGVDServer(SimpleNamespace(multimodal_server_epochs=1), None, '', pool, 'cpu')
        server.send_message()
        self.assertEqual(pool['server']['syn_logits'], {})
        server.execute()
        pool.update(sampled_clients=[2], client_2={'graph': graph_to_payload(graph())})
        server.execute()
        server.send_message()
        with torch.no_grad():
            expected = server_task.model.forward_modalities(server.global_data)[1]
        for cid in (2, 7):
            self.assertEqual(tuple(pool['server']['syn_logits'][cid].shape), (3, 2))
            self.assertFalse(pool['server']['syn_logits'][cid].requires_grad)
            torch.testing.assert_close(pool['server']['syn_logits'][cid], expected[server.client_slices[cid]])
        self.assertEqual(set(pool['server']), {'syn_logits'})
        coordinated = server.global_data
        pool.update(sampled_clients=[7], client_7={})
        server.execute()
        self.assertIs(server.global_data, coordinated)
        self.assertEqual(set(server.graph_cache), {2, 7})


if __name__ == '__main__':
    unittest.main()
