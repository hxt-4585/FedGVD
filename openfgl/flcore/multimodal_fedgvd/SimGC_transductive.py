"""Joint text/image graph condensation, following FedGVD's SimGC stages.

Train a local teacher, optimize paired synthetic features and a shared PGE,
then select the synthetic graph by a GCN student's real validation accuracy.
No global argument parser or training side effects occur on import.
"""

import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from openfgl.model.multimodal_gcn import MultimodalGCN
from openfgl.model.multimodal_sgc import MultimodalSGC
from openfgl.model.multimodal_utils import multimodal_supervised_loss
from openfgl.flcore.fedgvd.models.parametrized_adj import PGE
from .fedgvd_config import config
from .utils import cache_signature, graph_from_payload, graph_to_payload, modality_mask


def generate_labels_syn(labels_train, reduction_rate):
    classes, counts = labels_train.unique(sorted=True, return_counts=True)
    sizes = (counts.float() * reduction_rate).ceil().long()
    return torch.repeat_interleave(classes, sizes)


def _loss(model, data, mask):
    _, fused, branches, _ = model.forward_modalities(data)
    return multimodal_supervised_loss(branches, fused, data.y, mask,
                                      modality_mask(data), config["branch_weight"])


@torch.no_grad()
def _accuracy(model, data, mask):
    mask = mask & modality_mask(data).any(dim=1)
    if not mask.any():
        return None
    model.eval()
    return (model(data)[1][mask].argmax(dim=1) == data.y[mask]).float().mean().item()


def train_teacher(data, train_mask, val_mask, model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score, best_state = -1.0, None
    for epoch in range(args.multimodal_teacher_epochs):
        model.train()
        optimizer.zero_grad()
        loss = _loss(model, data, train_mask)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % args.multimodal_validation_interval == 0 or epoch + 1 == args.multimodal_teacher_epochs:
            score = _accuracy(model, data, val_mask)
            score = -loss.item() if score is None else score
            if score > best_score or best_state is None:
                best_score, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    model.requires_grad_(False)
    return model


def _propagated_features(x, edges, weight, available, hops):
    """Differentiable symmetric normalization, including self loops."""
    keep = available[edges[0]] & available[edges[1]] & (edges[0] != edges[1])
    edges = edges[:, keep]
    weight = x.new_ones(edges.size(1)) if weight is None else weight[keep]
    loops = available.nonzero(as_tuple=True)[0]
    edges = torch.cat([edges, torch.stack([loops, loops])], dim=1)
    weight = torch.cat([weight, x.new_ones(loops.numel())])
    degree = x.new_zeros(x.size(0)).index_add(0, edges[1], weight)
    inv = degree.clamp_min(1e-12).rsqrt()
    norm = inv[edges[0]] * weight * inv[edges[1]]
    x = F.normalize(x, dim=1) * available[:, None]
    features = [x]
    for _ in range(hops):
        x = torch.zeros_like(x).index_add(0, edges[1], x[edges[0]] * norm[:, None])
        features.append(x)
    return torch.cat(features, dim=1)


def _alignment(real_stats, synthetic, labels, available):
    loss = synthetic.sum() * 0
    total = 0
    for label, (mean, std, count) in real_stats.items():
        selected = synthetic[(labels == label) & available]
        if selected.size(0):
            loss = loss + count * (F.mse_loss(selected.mean(0), mean)
                                   + F.mse_loss(selected.std(0, unbiased=False), std))
            total += count
    return loss / max(total, 1)


def _synthetic_graph(text, image, labels, mask, edges, weight):
    return Data(x=torch.cat([F.normalize(image, dim=1), F.normalize(text, dim=1)], dim=1),
                x_text=text, x_image=image, y=labels, modality_mask=mask,
                edge_index=edges, edge_weight=weight)


def _validate_synthetic(synthetic, real, val_mask, args, dimensions, classes):
    # Give every candidate the same initialization and training randomness.
    devices = [real.x.device.index] if real.x.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(args.seed)
        student = MultimodalGCN(dimensions, args.hid_dim, classes, args.num_layers,
                               args.dropout, args.multimodal_fusion_temperature).to(real.x.device)
        optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        mask = torch.ones(synthetic.num_nodes, dtype=torch.bool, device=real.x.device)
        for _ in range(args.multimodal_student_epochs):
            student.train()
            optimizer.zero_grad()
            _loss(student, synthetic, mask).backward()
            optimizer.step()
        return _accuracy(student, real, val_mask)


def train_syn(data, train_mask, val_mask, teacher, args, dimensions, classes):
    available = modality_mask(data)
    train_mask = train_mask & available.any(dim=1)
    labels = generate_labels_syn(data.y[train_mask], args.multimodal_reduction_rate)
    selected = []
    for label in labels.unique(sorted=True):
        candidates = ((data.y == label) & train_mask).nonzero(as_tuple=True)[0]
        count = int((labels == label).sum())
        selected.append(candidates[torch.randperm(candidates.numel(), device=data.x.device)[:count]])
    selected = torch.cat(selected)
    mask = available[selected].clone()
    text = torch.nn.Parameter(F.normalize(data.x_text[selected], dim=1).clone())
    image = torch.nn.Parameter(F.normalize(data.x_image[selected], dim=1).clone())
    n = labels.numel()
    edges = torch.cartesian_prod(torch.arange(n, device=data.x.device),
                                 torch.arange(n, device=data.x.device)).T
    edges = edges[:, edges[0] != edges[1]]
    pge = PGE(sum(dimensions), n, nhid=args.hid_dim, device=data.x.device,
              args=args).to(data.x.device) if n > 1 else None
    feature_optimizer = torch.optim.Adam([text, image], lr=config["lr_feat"])
    edge_optimizer = torch.optim.Adam(pge.parameters(), lr=config["lr_adj"]) if pge else None
    real_stats = []
    for modal, feature in enumerate([data.x_text, data.x_image]):
        propagated = _propagated_features(feature, data.edge_index,
                        getattr(data, "edge_weight", None), available[:, modal], args.num_layers).detach()
        stats = {}
        for label in labels.unique().tolist():
            values = propagated[train_mask & (data.y == label) & available[:, modal]]
            if values.size(0):
                stats[label] = (values.mean(0), values.std(0, unbiased=False), values.size(0))
        real_stats.append(stats)
    best_score, best_graph = -float("inf"), None
    syn_mask = torch.ones(n, dtype=torch.bool, device=data.x.device)
    for epoch in range(args.multimodal_condense_epochs):
        feature_optimizer.zero_grad()
        if edge_optimizer:
            edge_optimizer.zero_grad()
        paired = torch.cat([text * mask[:, :1], image * mask[:, 1:]], dim=1)
        weight = pge(paired)[edges[0], edges[1]] if pge else text.new_empty(0)
        synthetic = _synthetic_graph(text, image, labels, mask, edges, weight)
        loss = _loss(teacher, synthetic, syn_mask)
        smoothness = text.sum() * 0
        for modal, feature in enumerate([text, image]):
            propagated = _propagated_features(feature, edges, weight, mask[:, modal], args.num_layers)
            loss = loss + config["feat_alpha"] * _alignment(real_stats[modal], propagated, labels, mask[:, modal])
            valid = mask[edges[0], modal] & mask[edges[1], modal]
            if valid.any():
                normalized = F.normalize(feature, dim=1)
                distance = (normalized[edges[0, valid]] - normalized[edges[1, valid]]).square().mean(dim=1)
                smoothness = smoothness + (weight[valid] * distance).sum() / weight[valid].sum().clamp_min(1e-12)
        loss = loss + config["smoothness_alpha"] * smoothness
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite multimodal condensation loss")
        loss.backward()
        feature_optimizer.step()
        if edge_optimizer:
            edge_optimizer.step()
        if (epoch + 1) % args.multimodal_validation_interval == 0 or epoch + 1 == args.multimodal_condense_epochs:
            with torch.no_grad():
                paired = torch.cat([text * mask[:, :1], image * mask[:, 1:]], dim=1)
                weight = pge(paired)[edges[0], edges[1]] if pge else text.new_empty(0)
                keep = weight >= config["adj_threshold"]
                candidate = _synthetic_graph(text.detach().clone(), image.detach().clone(), labels,
                                             mask, edges[:, keep], weight[keep].detach().clone())
            score = _validate_synthetic(candidate, data, val_mask, args, dimensions, classes)
            score = -loss.item() if score is None else score
            if score > best_score:
                best_score, best_graph = score, candidate
            print(f"[condense] epoch={epoch + 1} loss={loss.item():.5f} validation={score:.4f}", flush=True)
    return best_graph


def data_gc(processed_data, client_id, args, device):
    """Return a cached or newly condensed paired graph on the requested device."""
    if not 0 < args.multimodal_reduction_rate <= 1:
        raise ValueError("multimodal_reduction_rate must be in (0, 1]")
    for name in ("multimodal_teacher_epochs", "multimodal_condense_epochs",
                 "multimodal_student_epochs", "multimodal_validation_interval"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    data = processed_data["data"].to(device)
    train_mask = processed_data["train_mask"].to(device)
    val_mask = processed_data["val_mask"].to(device)
    if not (train_mask & modality_mask(data).any(dim=1)).any():
        raise ValueError("Condensation requires training nodes with an available modality")
    dimensions = (data.x_text.size(1), data.x_image.size(1))
    classes = int(data.num_global_classes)
    settings = {name: getattr(args, name) for name in (
        "seed", "num_clients", "hid_dim", "num_layers", "dropout", "lr", "weight_decay",
        "multimodal_reduction_rate", "multimodal_teacher_epochs", "multimodal_condense_epochs",
        "multimodal_student_epochs", "multimodal_validation_interval", "multimodal_fusion_temperature")}
    settings.update(version=1, client_id=client_id, classes=classes, config=config)
    signature = cache_signature(processed_data, settings)
    path = Path(args.multimodal_condensed_pattern.format(dataset=args.dataset[0].lower(),
                reduction_rate=args.multimodal_reduction_rate, seed=args.seed, client_id=client_id))
    if path.exists():
        cached = torch.load(path, map_location="cpu")
        if cached.get("signature") == signature:
            print(f"[client {client_id}] loaded condensed graph: {path}", flush=True)
            return graph_from_payload(cached["graph"], device)
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(args.seed + client_id)
        teacher = MultimodalSGC(dimensions, args.hid_dim, classes, args.num_layers,
                               args.dropout, args.multimodal_fusion_temperature).to(device)
        train_teacher(data, train_mask, val_mask, teacher, args)
        graph = train_syn(data, train_mask, val_mask, teacher, args, dimensions, classes)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"signature": signature, "graph": graph_to_payload(graph)}, path)
    print(f"[client {client_id}] saved {graph.num_nodes} paired nodes: {path}", flush=True)
    return graph
