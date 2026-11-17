"""Supervised Phase 1 diagnostic, not yet CWoLa hunting.

Mass and EFPs use all event constituents, not clustered jets or dijet mass.
A changed selected-background shape is a diagnostic, not proof of a resonance.
"""
import argparse
from pathlib import Path
import warnings
import h5py
import hdf5plugin  # Register Blosc compression used by the supplied HDF5 file.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# EnergyFlow probes fork at import even when no multiprocessing is used.
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message=".*'fork' is not available.*", category=UserWarning)
    import energyflow as ef


def extract_features(path, count, chunk_size):
    efpset = ef.EFPSet('d<=4', measure='hadr', beta=1.0)
    with h5py.File(path, 'r') as handle:
        # Pandas fixed-format layout verified in the supplied LHCO file.
        values = handle['df/block0_values']
        columns = handle['df/block0_items'][:]
        if values.ndim != 2 or values.shape[1] != 2101 or not np.array_equal(columns, np.arange(2101)):
            raise ValueError('Expected LHCO rows with 2100 kinematic columns followed by label.')
        count = min(count, values.shape[0])
        features = np.empty((count, len(efpset.specs)), dtype=np.float64)
        masses = np.empty(count, dtype=np.float64)
        labels = np.empty(count, dtype=np.float32)
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            block = values[start:stop]  # Slice on disk BEFORE materializing in RAM.
            for offset, row in enumerate(block):
                i = start + offset
                particles = row[:2100].reshape(700, 3)
                particles = particles[particles[:, 0] > 0]  # Remove zero padding.
                if not len(particles) or not np.isfinite(particles).all():
                    raise ValueError(f'Invalid constituent data at row {i}')
                pt, eta, phi = particles.T
                energy = np.sum(pt * np.cosh(eta))
                px = np.sum(pt * np.cos(phi))
                py = np.sum(pt * np.sin(phi))
                pz = np.sum(pt * np.sinh(eta))
                masses[i] = np.sqrt(max(0., energy**2 - px**2 - py**2 - pz**2))
                # Three coordinates are sufficient for massless particles (y=eta).
                # Single-event compute never creates a multiprocessing pool.
                features[i] = efpset.compute(particles)
                labels[i] = row[-1]
            print(f'Extracted {stop}/{count} events', flush=True)
    if not np.isfinite(features).all() or not np.isfinite(masses).all():
        raise ValueError('Non-finite extracted features or masses')
    if set(np.unique(labels)) != {0., 1.}:
        raise ValueError('Subset must contain both background and signal; increase --events.')
    print(f'Computed {features.shape[1]} EFPs; signal fraction {labels.mean():.3f}')
    return features, masses, labels


class LHCOlympicsDataset(Dataset):
    def __init__(self, features, masses, labels):
        self.features = torch.FloatTensor(features)
        self.masses = torch.FloatTensor(masses)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.masses[idx], self.labels[idx]


class BaselineEFPNetwork(nn.Module):
    def __init__(self, input_dim):
        super(BaselineEFPNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).with_name('events_anomalydetection.h5'))
    parser.add_argument('--events', type=int, default=50000)
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('phase1_mass_sculpting.png'))
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if min(args.events, args.chunk_size, args.epochs) <= 0:
        parser.error('events, chunk-size, and epochs must be positive')
    np.random.seed(42)
    torch.manual_seed(42)
    efps, jet_mass, y = extract_features(args.data, args.events, args.chunk_size)
    X_train, X_test, m_train, m_test, y_train, y_test = train_test_split(
        efps, jet_mass, y, test_size=0.3, random_state=42, stratify=y)
    # Fit preprocessing on training data only. The constant EFP has zero variance.
    mean, scale = X_train.mean(axis=0), X_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    X_train = ((X_train - mean) / scale).astype(np.float32)
    X_test = ((X_test - mean) / scale).astype(np.float32)
    train_loader = DataLoader(LHCOlympicsDataset(X_train, m_train, y_train), batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(LHCOlympicsDataset(X_test, m_test, y_test), batch_size=512, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineEFPNetwork(input_dim=efps.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ====================================================================
    # 5. TRAINING LOOP
    # ====================================================================
    epochs = args.epochs
    print(f"Training Baseline Model on {device}...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_efps, _, batch_labels in train_loader:
            batch_efps, batch_labels = batch_efps.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            predictions = model(batch_efps)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {epoch_loss / len(train_loader):.4f}")

    # ====================================================================
    # 6. EVALUATION: PROVING MASS SCULPTING (THE FLAW)
    # ====================================================================
    model.eval()
    all_preds, all_masses, all_labels = [], [], []

    with torch.no_grad():
        for batch_efps, batch_masses, batch_labels in test_loader:
            preds = model(batch_efps.to(device)).cpu().numpy()
            all_preds.extend(preds)
            all_masses.extend(batch_masses.numpy())
            all_labels.extend(batch_labels.numpy())

    all_preds = np.array(all_preds).squeeze()
    all_masses = np.array(all_masses)
    all_labels = np.array(all_labels).squeeze()

    # Isolate ONLY the True QCD Background events
    bkg_mask = (all_labels == 0)
    bkg_masses = all_masses[bkg_mask]
    bkg_preds = all_preds[bkg_mask]

    # Find the top 5% most "signal-like" background events (The False Positives)
    threshold = np.percentile(bkg_preds, 95)
    sculpted_bkg_masses = bkg_masses[bkg_preds > threshold]

    # ====================================================================
    # 7. PLOTTING THE BUMP
    # ====================================================================
    plt.figure(figsize=(10, 6))

    # Plot underlying mass distribution of all background jets
    plt.hist(bkg_masses, bins=60, range=(2500, 4500), density=True,
             alpha=0.5, color='blue', label='True QCD Background Mass')

    # Plot the mass of background jets the model confidently classified as "signal"
    plt.hist(sculpted_bkg_masses, bins=60, range=(2500, 4500), density=True,
             alpha=0.7, color='red', label=f'Model-Selected Background (Threshold > {threshold:.2f})')

    plt.title("Phase 1: Artificial Mass Sculpting by the EFP Baseline Model")
    plt.xlabel("All-particle Event Invariant Mass (GeV)")
    plt.ylabel("Normalized Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(args.output, dpi=160, bbox_inches="tight")
    print(f"Saved plot to {args.output}")
    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()

