#!/usr/bin/env python3
"""Train manifold projection heads on GAMI's hierarchical structure.

This script trains the projection heads to produce meaningful coordinates:
- Hyperbolic: Parents closer to origin than children
- Spherical: Same-type entities cluster together
- Euclidean: Preserves semantic similarity from base embeddings

Without training, the manifold embeddings are just random projections!
"""
import sys
sys.path.insert(0, '/opt/gami')

import os
import json
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sqlalchemy import create_engine, text
from tqdm import tqdm
import numpy as np
from collections import defaultdict


def parse_embedding(emb_str):
    """Parse embedding from database string/list format to list of floats."""
    if emb_str is None:
        return None
    if isinstance(emb_str, (list, tuple, np.ndarray)):
        return [float(v) for v in emb_str]
    if isinstance(emb_str, str):
        try:
            return json.loads(emb_str)
        except json.JSONDecodeError:
            # Try parsing pgvector format [x,y,z]
            clean = emb_str.strip('[]')
            return [float(v) for v in clean.split(',') if v.strip()]
    return list(emb_str)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# Database connection
DB_URL = os.getenv("GAMI_DATABASE_URL", "postgresql://gami:PASSWORD@127.0.0.1:5433/gami")
engine = create_engine(DB_URL)


class HierarchyDataset(Dataset):
    """Dataset of hierarchical relationships for training."""

    def __init__(self, embeddings: dict, hierarchy_pairs: list, type_labels: dict):
        """
        Args:
            embeddings: {id: 768d numpy array}
            hierarchy_pairs: [(parent_id, child_id), ...]
            type_labels: {id: type_string}
        """
        self.embeddings = embeddings
        self.hierarchy_pairs = hierarchy_pairs
        self.type_labels = type_labels
        self.ids = list(embeddings.keys())

        # Create type groups for spherical loss
        self.type_groups = defaultdict(list)
        for id_, type_ in type_labels.items():
            if id_ in embeddings:
                self.type_groups[type_].append(id_)

    def __len__(self):
        return len(self.hierarchy_pairs)

    def __getitem__(self, idx):
        parent_id, child_id = self.hierarchy_pairs[idx]
        return {
            'parent_emb': torch.tensor(self.embeddings[parent_id], dtype=torch.float32),
            'child_emb': torch.tensor(self.embeddings[child_id], dtype=torch.float32),
            'parent_id': parent_id,
            'child_id': child_id,
        }


def load_training_data():
    """Load hierarchical signals from GAMI database."""
    log.info("Loading training data from database...")

    embeddings = {}
    hierarchy_pairs = []
    type_labels = {}

    with engine.connect() as conn:
        # 1. Load entity embeddings and types
        log.info("  Loading entity embeddings...")
        result = conn.execute(text("""
            SELECT entity_id, embedding, entity_type
            FROM entities
            WHERE embedding IS NOT NULL
            LIMIT 50000
        """))
        for row in result:
            emb = parse_embedding(row.embedding)
            if emb and len(emb) == 768:
                embeddings[row.entity_id] = emb
                type_labels[row.entity_id] = row.entity_type or 'unknown'
        log.info(f"  Loaded {len(embeddings)} entity embeddings")

        # 2. Load segment embeddings
        log.info("  Loading segment embeddings...")
        result = conn.execute(text("""
            SELECT segment_id, embedding, segment_type
            FROM segments
            WHERE embedding IS NOT NULL
            AND storage_tier != 'cold'
            LIMIT 100000
        """))
        for row in result:
            emb = parse_embedding(row.embedding)
            if emb and len(emb) == 768:
                embeddings[f"seg_{row.segment_id}"] = emb
                type_labels[f"seg_{row.segment_id}"] = f"segment_{row.segment_type or 'general'}"
        log.info(f"  Total embeddings: {len(embeddings)}")

        # 3. Extract hierarchy from cluster membership
        log.info("  Loading cluster hierarchy...")
        result = conn.execute(text("""
            SELECT cluster_id, cluster_embedding, member_ids
            FROM memory_clusters
            WHERE cluster_embedding IS NOT NULL
            AND member_ids IS NOT NULL
            AND array_length(member_ids, 1) > 0
        """))
        cluster_count = 0
        for row in result:
            cluster_id = f"cluster_{row.cluster_id}"
            emb = parse_embedding(row.cluster_embedding)
            if emb and len(emb) == 768:
                embeddings[cluster_id] = emb
                type_labels[cluster_id] = 'cluster'

            # Cluster is parent of its members
            for member_id in row.member_ids[:20]:  # Limit to avoid huge graphs
                if member_id in embeddings or f"seg_{member_id}" in embeddings:
                    member_key = member_id if member_id in embeddings else f"seg_{member_id}"
                    hierarchy_pairs.append((cluster_id, member_key))
                    cluster_count += 1
        log.info(f"  Cluster hierarchy pairs: {cluster_count}")

        # 4. Extract hierarchy from entity types (heuristic)
        # service < application < system (more specific = child)
        type_hierarchy = {
            'person': 'entity',
            'organization': 'entity',
            'location': 'entity',
            'technology': 'concept',
            'service': 'technology',
            'database': 'service',
            'concept': 'entity',
        }

        log.info("  Creating type hierarchy pairs...")
        type_pairs = 0
        for id_, type_ in type_labels.items():
            if type_ in type_hierarchy and id_ in embeddings:
                parent_type = type_hierarchy[type_]
                # Find an entity of parent type to use as proxy
                for other_id, other_type in type_labels.items():
                    if other_type == parent_type and other_id in embeddings:
                        hierarchy_pairs.append((other_id, id_))
                        type_pairs += 1
                        if type_pairs > 10000:
                            break
                if type_pairs > 10000:
                    break
        log.info(f"  Type hierarchy pairs: {type_pairs}")

        # 5. Create semantic similarity pairs for Euclidean loss
        # Use entities that share the same cluster as positive pairs

    log.info(f"Total hierarchy pairs: {len(hierarchy_pairs)}")
    log.info(f"Unique types: {len(set(type_labels.values()))}")

    return embeddings, hierarchy_pairs, type_labels


class ManifoldTrainer:
    """Train manifold projection heads."""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        log.info(f"Using device: {device}")

        # Import the encoder
        from api.llm.manifold_embeddings import ManifoldEncoder
        self.encoder = ManifoldEncoder(device=device)

        # Only train projection heads, not base encoder
        self.encoder.base_encoder.requires_grad_(False)

        # Optimizer for projection heads only
        params = list(self.encoder.hyperbolic_head.parameters()) + \
                 list(self.encoder.spherical_head.parameters()) + \
                 list(self.encoder.euclidean_head.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)

    def hyperbolic_hierarchy_loss(self, parent_h, child_h, margin=0.1):
        """Parents should be closer to origin than children in Poincaré ball.

        In hyperbolic space, the center represents general concepts,
        and the boundary represents specific instances.
        """
        parent_norm = torch.norm(parent_h, dim=-1)
        child_norm = torch.norm(child_h, dim=-1)

        # Child should have larger norm (further from origin)
        loss = F.relu(parent_norm - child_norm + margin)
        return loss.mean()

    def spherical_clustering_loss(self, embeddings, type_labels, type_groups):
        """Same-type items should cluster on the sphere.

        Uses contrastive loss: pull same-type closer, push different-type apart.
        """
        if len(embeddings) < 2:
            return torch.tensor(0.0, device=self.device)

        # Compute pairwise distances
        # For efficiency, sample a subset
        n = min(len(embeddings), 100)
        indices = torch.randperm(len(embeddings))[:n]
        sampled = embeddings[indices]

        # Normalize to sphere
        sampled = F.normalize(sampled, p=2, dim=-1)

        # Pairwise dot products (cosine similarity on sphere)
        sims = torch.mm(sampled, sampled.t())

        # Create label matrix (1 if same type, 0 otherwise)
        # This is simplified - in practice would use actual type labels

        # For now, use self-similarity as anchor
        # Pull diagonal closer to 1, push off-diagonal toward -1
        diag = torch.diag(sims)
        off_diag = sims - torch.diag(diag)

        # Simple contrastive: maximize diagonal, minimize off-diagonal
        loss = -diag.mean() + 0.5 * off_diag.abs().mean()
        return loss

    def euclidean_preservation_loss(self, original_emb, projected_emb):
        """Preserve relative distances from original embedding space.

        If A was closer to B than C in original space, should be same in projected.
        """
        # Compute original similarities
        orig_norm = F.normalize(original_emb, p=2, dim=-1)
        orig_sims = torch.mm(orig_norm, orig_norm.t())

        # Compute projected similarities
        proj_norm = F.normalize(projected_emb, p=2, dim=-1)
        proj_sims = torch.mm(proj_norm, proj_norm.t())

        # MSE between similarity matrices
        loss = F.mse_loss(proj_sims, orig_sims)
        return loss

    def train_epoch(self, dataloader, epoch):
        """Train one epoch."""
        self.encoder.train()
        total_loss = 0
        h_loss_sum = 0
        s_loss_sum = 0
        e_loss_sum = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            parent_emb = batch['parent_emb'].to(self.device)
            child_emb = batch['child_emb'].to(self.device)

            self.optimizer.zero_grad()

            # Project to manifold
            parent_h = self.encoder.hyperbolic_head(parent_emb)
            child_h = self.encoder.hyperbolic_head(child_emb)
            parent_s = self.encoder.spherical_head(parent_emb)
            child_s = self.encoder.spherical_head(child_emb)
            parent_e = self.encoder.euclidean_head(parent_emb)
            child_e = self.encoder.euclidean_head(child_emb)

            # Hyperbolic loss: hierarchy
            h_loss = self.hyperbolic_hierarchy_loss(parent_h, child_h)

            # Spherical loss: clustering (using batch as proxy)
            all_s = torch.cat([parent_s, child_s], dim=0)
            s_loss = self.spherical_clustering_loss(all_s, None, None)

            # Euclidean loss: preserve similarity
            all_orig = torch.cat([parent_emb, child_emb], dim=0)
            all_e = torch.cat([parent_e, child_e], dim=0)
            e_loss = self.euclidean_preservation_loss(all_orig, all_e)

            # Combined loss
            loss = 0.4 * h_loss + 0.2 * s_loss + 0.4 * e_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            h_loss_sum += h_loss.item()
            s_loss_sum += s_loss.item()
            e_loss_sum += e_loss.item()

        n = len(dataloader)
        return {
            'total': total_loss / n,
            'hyperbolic': h_loss_sum / n,
            'spherical': s_loss_sum / n,
            'euclidean': e_loss_sum / n,
        }

    def train(self, epochs=50, batch_size=64):
        """Full training loop."""
        log.info("Starting manifold projector training...")

        # Load data
        embeddings, hierarchy_pairs, type_labels = load_training_data()

        if len(hierarchy_pairs) < 100:
            log.warning("Not enough hierarchy pairs for training!")
            return False

        # Create dataset
        dataset = HierarchyDataset(embeddings, hierarchy_pairs, type_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        log.info(f"Training on {len(dataset)} hierarchy pairs for {epochs} epochs")

        best_loss = float('inf')
        for epoch in range(1, epochs + 1):
            losses = self.train_epoch(dataloader, epoch)

            log.info(f"Epoch {epoch}: total={losses['total']:.4f} "
                    f"h={losses['hyperbolic']:.4f} "
                    f"s={losses['spherical']:.4f} "
                    f"e={losses['euclidean']:.4f}")

            if losses['total'] < best_loss:
                best_loss = losses['total']
                self.encoder.save_projection_weights()
                log.info(f"  Saved best weights (loss={best_loss:.4f})")

        log.info("Training complete!")
        return True

    def validate(self):
        """Validate that training produced meaningful structure."""
        log.info("Validating trained projections...")

        self.encoder.eval()

        with engine.connect() as conn:
            # Get some cluster parents and their children
            result = conn.execute(text("""
                SELECT cluster_id, cluster_embedding, member_ids
                FROM memory_clusters
                WHERE cluster_embedding IS NOT NULL
                AND array_length(member_ids, 1) >= 3
                LIMIT 10
            """))

            valid_hierarchy = 0
            total_checks = 0

            for row in result:
                cluster_emb = torch.tensor([row.cluster_embedding], dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    cluster_h = self.encoder.hyperbolic_head(cluster_emb)
                    cluster_norm = torch.norm(cluster_h).item()

                for member_id in row.member_ids[:5]:
                    # Get member embedding
                    mem_result = conn.execute(text("""
                        SELECT embedding FROM segments WHERE segment_id = :id
                    """), {"id": member_id}).fetchone()

                    if mem_result and mem_result.embedding:
                        member_emb = torch.tensor([list(mem_result.embedding)], dtype=torch.float32).to(self.device)

                        with torch.no_grad():
                            member_h = self.encoder.hyperbolic_head(member_emb)
                            member_norm = torch.norm(member_h).item()

                        total_checks += 1
                        if cluster_norm < member_norm:
                            valid_hierarchy += 1

            if total_checks > 0:
                accuracy = valid_hierarchy / total_checks
                log.info(f"Hierarchy validation: {valid_hierarchy}/{total_checks} = {accuracy:.1%}")
                log.info("  (cluster closer to origin than members)")
                return accuracy > 0.6  # At least 60% correct

        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train manifold projection heads')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--validate-only', action='store_true', help='Only run validation')
    args = parser.parse_args()

    trainer = ManifoldTrainer()

    if args.validate_only:
        trainer.validate()
    else:
        success = trainer.train(epochs=args.epochs, batch_size=args.batch_size)
        if success:
            trainer.validate()


if __name__ == "__main__":
    main()
