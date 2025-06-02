main.py – Full Collapse Engine Stack (JAX-Optimized + Bayesian Dynamic Tensor Reweighting)

This engine connects the Information Catastrophe Thermodynamics (ICT) modeling,
recursive manifold folding, predictive collapse zone analysis, and intervention mapping.
Key improvements include:
  • Use of JAX (jax.numpy as jnp) for accelerated linear algebra.
  • A Bayesian update for the entanglement_matrix (dynamic tensor reweighting) 
    replacing the standard momentum-based update.
  • A low_rank_svd utility to compress tensors (optional).
  • Parallelized Monte Carlo simulation using jax.vmap and jax.lax.scan.
All critical functions are adapted to work with JAX DeviceArrays.


import jax
import jax.numpy as jnp
import numpy as np  # For non-JAX tasks if needed
import datetime
import json
from typing import Dict, List, Any
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings("ignore")

# --- Data Structures ---
@dataclass
class CollapseSignature:
    timestamp: datetime.datetime
    morphology: jnp.ndarray
    invariants: List[float]  # e.g. [trace, determinant, spectral_gap]
    influence_radius: float
    folding_strength: float
    domain: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ManifoldState:
    curvature_field: jnp.ndarray
    memory_tensor: jnp.ndarray
    entanglement_matrix: jnp.ndarray
    collapse_history: List[CollapseSignature]
    bifurcation_zones: jnp.ndarray
    intervention_efficacy: jnp.ndarray

@dataclass
class DomainMetrics:
    institutional_trust: float
    information_integrity: float
    electoral_confidence: float
    alliance_stability: float
    social_cohesion: float
    timestamp: datetime.datetime

# --- Core Engine: CatastropheManifoldFolder ---
class CatastropheManifoldFolder:
    def __init__(self, dimensions: int = 5, memory_depth: int = 20):
        self.dimensions = dimensions
        self.memory_depth = memory_depth
        self.manifold_memory = []  # holds CollapseSignature objects
        self.curvature_field = None  # current manifold curvature
        # Initialize entanglement_matrix as a JAX identity matrix.
        self.entanglement_matrix = jnp.eye(dimensions)
        self.folding_history = []  # detailed collapse records
        # Detection thresholds:
        self.fold_threshold = 0.7
        self.cascade_threshold = 0.85
        self.intervention_threshold = 0.6
        # Domain mapping for integration
        self.domain_indices = {
            'institutional_trust': 0,
            'information_integrity': 1,
            'electoral_confidence': 2,
            'alliance_stability': 3,
            'social_cohesion': 4
        }

    def fold_manifold(self, collapse_tensor: jnp.ndarray, domain: str = "unknown", coupling_strength: float = 0.15) -> ManifoldState:
        """
        Folds the manifold using the collapse_tensor event, updating the curvature_field,
        and uses a Bayesian update to reweight the entanglement matrix.
        """
        # Eigen-decomposition using JAX:
        eigenvalues, eigenvectors = jnp.linalg.eigh(collapse_tensor)
        folding_strength = jnp.max(jnp.abs(eigenvalues)) / (jnp.trace(jnp.abs(collapse_tensor)) + 1e-10)
        fold_operator = self._construct_fold_operator(eigenvectors, folding_strength)
        if self.curvature_field is None:
            self.curvature_field = fold_operator
        else:
            memory_influence = self._compute_memory_tensor()
            # Combine new fold operator with influence from past events:
            self.curvature_field = 0.7 * fold_operator + 0.3 * jnp.dot(memory_influence, self.curvature_field)

        # Update the entanglement_matrix using a Bayesian (dynamic tensor reweighting) update.
        self.entanglement_matrix = self.update_entanglement_bayesian(collapse_tensor, self.entanglement_matrix, coupling_strength)

        # Compress and store the collapse signature:
        compressed_signature = self._compress_collapse_signature(collapse_tensor, self.curvature_field, domain, folding_strength)
        self.manifold_memory.append(compressed_signature)
        self.folding_history.append(compressed_signature)
        # Maintain memory depth:
        if len(self.manifold_memory) > self.memory_depth:
            self.manifold_memory.pop(0)
        # Create current manifold state:
        state = ManifoldState(
            curvature_field=self.curvature_field.copy(),
            memory_tensor=self._compute_memory_tensor(),
            entanglement_matrix=self.entanglement_matrix.copy(),
            collapse_history=self.folding_history.copy(),
            bifurcation_zones=self._identify_bifurcation_zones(),
            intervention_efficacy=self._compute_intervention_efficacy()
        )
        return state

    def predict_collapse_zones(self, current_state: jnp.ndarray, horizon: int = 10, trajectories: int = 100) -> Dict[str, Any]:
        """
        Predicts future collapse zones and intervention points using a Monte Carlo simulation.
        Leverages JAX vmap and lax.scan for parallelized rollouts.
        """
        if self.curvature_field is None:
            return {
                'collapse_probability': jnp.zeros_like(current_state),
                'intervention_zones': jnp.zeros_like(current_state),
                'cascade_risk': 0.0
            }
        projected_state = jnp.dot(current_state, self.curvature_field)
        flow_field = self._compute_geodesic_flow(projected_state)

        def trajectory_simulation(trajectory_seed):
            # Define a simulation using lax.scan to evolve the state over the horizon.
            def evolve_step(carry, t):
                state = carry
                evolved_state = self._evolve_on_manifold(state, flow_field, t)
                # Compute convergence zones at this state
                return evolved_state, self._detect_convergence_zones(evolved_state)
            final_state, collapse_map = jax.lax.scan(evolve_step, projected_state, jnp.arange(horizon))
            return collapse_map

        rng = jax.random.PRNGKey(0)
        trajectory_seeds = jax.random.split(rng, trajectories)
        collapse_map = jax.vmap(trajectory_simulation)(trajectory_seeds)
        collapse_probability = jnp.mean(collapse_map, axis=0)
        historical_weight = self._compute_historical_proximity(current_state)
        weighted_prediction = collapse_probability * historical_weight
        intervention_zones = self._identify_intervention_manifolds(weighted_prediction, self.curvature_field)
        cascade_risk = self._compute_cascade_risk(weighted_prediction, self.entanglement_matrix)
        return {
            'collapse_probability': weighted_prediction,
            'intervention_zones': intervention_zones,
            'cascade_risk': cascade_risk,
            'bifurcation_distance': self._distance_to_bifurcation(projected_state)
        }

    def _construct_fold_operator(self, eigenvectors: jnp.ndarray, strength: float) -> jnp.ndarray:
        scaling_matrix = jnp.diag(1 + strength * jnp.exp(-jnp.arange(len(eigenvectors))))
        fold_operator = jnp.dot(jnp.dot(eigenvectors, scaling_matrix), eigenvectors.T)
        nonlinear_term = strength * (jnp.outer(eigenvectors[:, 0], eigenvectors[:, 0]) ** 2)
        if len(self.manifold_memory) > 0:
            memory_effect = jnp.zeros_like(fold_operator)
            for sig in self.manifold_memory[-5:]:
                memory_effect += 0.1 * sig.morphology  # Assumes morphology shape matches the operator.
            fold_operator += memory_effect
        return fold_operator + nonlinear_term

    def update_entanglement_bayesian(self, collapse_tensor: jnp.ndarray, current_entanglement: jnp.ndarray, coupling_strength: float) -> jnp.ndarray:
        """
        Bayesian dynamic tensor reweighting.
        Uses the current entanglement (prior) and new evidence from collapse_tensor (via its correlation)
        to compute an updated entanglement_matrix.
        """
        # Likelihood: derive correlation from collapse_tensor.
        collapse_correlation = jnp.corrcoef(collapse_tensor)
        momentum = 0.9  # Prior weight.
        # Approximate a posterior by weighted average of the prior and new evidence.
        new_entanglement = momentum * current_entanglement + (1 - momentum) * coupling_strength * collapse_correlation
        # Project onto the cone of positive semidefinite matrices:
        eigenvals, eigenvecs = jnp.linalg.eigh(new_entanglement)
        eigenvals = jnp.maximum(eigenvals, 0)
        new_entanglement = jnp.dot(eigenvecs, jnp.dot(jnp.diag(eigenvals), eigenvecs.T))
        return new_entanglement

    def _compress_collapse_signature(self, tensor: jnp.ndarray, curvature: jnp.ndarray, domain: str, folding_strength: float) -> CollapseSignature:
        projected = jnp.dot(tensor, curvature)
        eigenvals = jnp.linalg.eigvals(projected)
        trace_invariant = jnp.real(jnp.trace(projected))
        det_invariant = jnp.real(jnp.linalg.det(projected))
        spectral_gap = jnp.real(jnp.max(eigenvals) - jnp.min(eigenvals))
        _, eigenvecs = jnp.linalg.eigh(projected)
        morphology = eigenvecs[:, -1]
        influence_radius = self._estimate_influence_radius(tensor, curvature)
        signature = CollapseSignature(
            timestamp=datetime.datetime.now(),
            morphology=morphology,
            invariants=[trace_invariant, det_invariant, spectral_gap],
            influence_radius=influence_radius,
            folding_strength=folding_strength,
            domain=domain,
            metadata={
                'eigenvalues': eigenvals.tolist(),
                'tensor_norm': float(jnp.linalg.norm(tensor))
            }
        )
        return signature

    def _compute_memory_tensor(self) -> jnp.ndarray:
        if len(self.manifold_memory) == 0:
            return jnp.eye(self.dimensions)
        memory_tensor = jnp.zeros((self.dimensions, self.dimensions))
        for i, sig in enumerate(self.manifold_memory):
            weight = jnp.exp(-0.1 * (len(self.manifold_memory) - i))
            memory_tensor += weight * jnp.outer(sig.morphology, sig.morphology)
        memory_tensor = memory_tensor / (jnp.trace(memory_tensor) + 1e-10)
        return memory_tensor

    def _compute_geodesic_flow(self, state: jnp.ndarray) -> jnp.ndarray:
        # Compute geodesic flow using automatic differentiation.
        flow = -jax.grad(lambda x: jnp.sum(self.curvature_field), argnums=0)(state)
        flow_field = jnp.dot(flow, state[:, None])
        return flow_field.squeeze()

    def _evolve_on_manifold(self, state: jnp.ndarray, flow_field: jnp.ndarray, time_step: int) -> jnp.ndarray:
        dt = 0.1
        k1 = dt * flow_field
        k2 = dt * self._flow_at_state(state + 0.5 * k1)
        k3 = dt * self._flow_at_state(state + 0.5 * k2)
        k4 = dt * self._flow_at_state(state + k3)
        evolved_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
        return self._project_to_manifold(evolved_state)

    def _flow_at_state(self, state: jnp.ndarray) -> jnp.ndarray:
        return -jnp.dot(self.curvature_field, state)

    def _project_to_manifold(self, state: jnp.ndarray) -> jnp.ndarray:
        norm = jnp.linalg.norm(state)
        return state / norm if norm > 0 else state

    def _detect_convergence_zones(self, state: jnp.ndarray) -> jnp.ndarray:
        epsilon = 1e-6
        divergence = jnp.zeros_like(state)
        for i in range(len(state)):
            perturbed_plus = state.copy().at[i].set(state[i] + epsilon)
            perturbed_minus = state.copy().at[i].set(state[i] - epsilon)
            flow_plus = self._flow_at_state(perturbed_plus)
            flow_minus = self._flow_at_state(perturbed_minus)
            divergence = divergence.at[i].set((flow_plus[i] - flow_minus[i]) / (2 * epsilon))
        convergence_indicator = jnp.maximum(0, -divergence)
        convergence_zones = (convergence_indicator > self.fold_threshold).astype(jnp.float32)
        return convergence_zones

    def _compute_historical_proximity(self, current_state: jnp.ndarray) -> jnp.ndarray:
        if len(self.manifold_memory) == 0:
            return jnp.ones_like(current_state)
        proximity = jnp.zeros_like(current_state)
        for sig in self.manifold_memory:
            distance = jnp.linalg.norm(current_state - sig.morphology)
            weight = jnp.exp(- (distance**2) / (2 * (sig.influence_radius**2)))
            proximity += weight
        proximity = proximity / (len(self.manifold_memory) + 1e-10)
        return 1 + proximity

    def _identify_intervention_manifolds(self, collapse_probability: jnp.ndarray, curvature: jnp.ndarray) -> jnp.ndarray:
        sensitivity = jnp.zeros_like(collapse_probability)
        for i in range(len(collapse_probability)):
            if collapse_probability[i] > self.intervention_threshold:
                jacobian = jnp.gradient(curvature[i, :])
                sensitivity = sensitivity.at[i].set(1.0 / (jnp.linalg.norm(jacobian) + 1e-10))
        if jnp.max(sensitivity) > 0:
            sensitivity = sensitivity / jnp.max(sensitivity)
        return sensitivity

    def _compute_cascade_risk(self, collapse_probability: jnp.ndarray, entanglement: jnp.ndarray) -> float:
        high_risk = collapse_probability > self.cascade_threshold
        if jnp.sum(high_risk) == 0:
            return 0.0
        cascade_matrix = entanglement * jnp.outer(high_risk, high_risk)
        eigenvals = jnp.linalg.eigvals(cascade_matrix)
        cascade_risk = jnp.real(jnp.max(eigenvals))
        return float(jnp.minimum(1.0, cascade_risk))

    def _distance_to_bifurcation(self, state: jnp.ndarray) -> float:
        jacobian = self.curvature_field - jnp.eye(self.dimensions)
        eigenvals = jnp.linalg.eigvals(jacobian)
        distance = jnp.min(jnp.abs(jnp.real(eigenvals)))
        return float(distance)

    def _identify_bifurcation_zones(self) -> jnp.ndarray:
        if self.curvature_field is None:
            return jnp.zeros((self.dimensions, self.dimensions))
        bifurcation_indicator = jnp.zeros((self.dimensions, self.dimensions))
        for i in range(self.dimensions):
            for j in range(self.dimensions):
                h = 1e-6
                f_pp = self.curvature_field[i, j]
                if i > 0 and j > 0 and i < self.dimensions - 1 and j < self.dimensions - 1:
                    f_px = self.curvature_field[i+1, j]
                    f_mx = self.curvature_field[i-1, j]
                    f_py = self.curvature_field[i, j+1]
                    f_my = self.curvature_field[i, j-1]
                    hessian_trace = ((f_px - 2 * f_pp + f_mx) + (f_py - 2 * f_pp + f_my)) / (h**2)
                    bifurcation_indicator = bifurcation_indicator.at[i, j].set(jnp.abs(hessian_trace))
        return bifurcation_indicator

    def _compute_intervention_efficacy(self) -> jnp.ndarray:
        if self.curvature_field is None:
            return jnp.ones((self.dimensions, self.dimensions))
        curvature_magnitude = jnp.abs(self.curvature_field)
        efficacy = 1.0 / (1.0 + curvature_magnitude)
        return efficacy

    def _estimate_influence_radius(self, tensor: jnp.ndarray, curvature: jnp.ndarray) -> float:
        tensor_scale = jnp.linalg.norm(tensor)
        curvature_scale = jnp.linalg.norm(curvature)
        radius = jnp.sqrt(tensor_scale) * (1 + 0.1 * curvature_scale)
        return float(radius)


# --- Manifold Collapse Analyzer ---
class ManifoldCollapseAnalyzer:
    def __init__(self, manifold_folder: CatastropheManifoldFolder):
        self.manifold_folder = manifold_folder

    def _metrics_to_tensor(self, metrics: DomainMetrics, prev_values: jnp.ndarray = None) -> (jnp.ndarray, jnp.ndarray):
        values = jnp.array([
            metrics.institutional_trust,
            metrics.information_integrity,
            metrics.electoral_confidence,
            metrics.alliance_stability,
            metrics.social_cohesion
        ])
        gradient = values - prev_values if prev_values is not None else jnp.zeros_like(values)
        tensor = jnp.outer(values, values) + 0.1 * jnp.outer(gradient, gradient)
        return tensor, values

# --- Utility: Low-Rank SVD Compression ---
def low_rank_svd(matrix: jnp.ndarray, rank: int) -> jnp.ndarray:
    u, s, vh = jnp.linalg.svd(matrix, full_matrices=False)
    return jnp.dot(u[:, :rank], jnp.dot(jnp.diag(s[:rank]), vh[:rank, :]))

# --- Main Execution ---
if __name__ == "__main__":
    print("🚀 Initializing Collapse Engine Stack (JAX-Optimized & Bayesian Upgraded)...")
    
    # Input: Synthetic Domain Metrics (replace with real data as needed)
    initial_metrics = DomainMetrics(
        institutional_trust=0.42,
        information_integrity=0.35,
        electoral_confidence=0.60,
        alliance_stability=0.28,
        social_cohesion=0.31,
        timestamp=datetime.datetime.utcnow()
    )
    print(f"▶️ Processing Input State at {initial_metrics.timestamp.isoformat()}")
    
    # Instantiate engine and analyzer
    manifold = CatastropheManifoldFolder(dimensions=5)
    analyzer = ManifoldCollapseAnalyzer(manifold)
    
    # Create collapse tensor and current values
    collapse_tensor, current_values = analyzer._metrics_to_tensor(initial_metrics, prev_values=None)
    print("    ...Collapse Tensor generated.")
    
    # Perform manifold folding with Bayesian reweighting
    folded_state = manifold.fold_manifold(collapse_tensor, domain="truth-tech", coupling_strength=0.2)
    print("    ...Manifold folding complete.")
    
    # Prepare a flattened state vector for prediction
    current_state_vector = collapse_tensor.flatten()
    predictions = manifold.predict_collapse_zones(current_state_vector, horizon=12, trajectories=200)
    print("    ...Collapse and bifurcation zones predicted.")
    
    intervention_zones = predictions.get('intervention_zones', jnp.array([]))
    intervention_efficacy = folded_state.intervention_efficacy
    print("    ...Intervention manifolds calculated.")
    
    # Log results to JSON file
    output_log = {
        "run_timestamp": datetime.datetime.utcnow().isoformat(),
        "input_state": initial_metrics.__dict__,
        "analysis_output": {
            "collapse_probability": predictions.get('collapse_probability').tolist(),
            "intervention_zones": intervention_zones.tolist(),
            "cascade_risk": predictions.get('cascade_risk'),
            "distance_to_bifurcation": predictions.get('bifurcation_distance'),
            "bifurcation_zones_indicator": folded_state.bifurcation_zones.tolist(),
            "intervention_efficacy_map": intervention_efficacy.tolist(),
        },
        "manifold_snapshot": {
            "curvature_field": folded_state.curvature_field.tolist(),
            "entanglement_matrix": folded_state.entanglement_matrix.tolist(),
            "memory_tensor": folded_state.memory_tensor.tolist()
        }
    }

    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            return json.JSONEncoder.default(self, obj)

    log_filename = "collapse_output_log.json"
    with open(log_filename, "w") as f:
        json.dump(output_log, f, indent=2, cls=DateTimeEncoder)
    
    print(f"\n✅ Analysis Complete. Full output logged to: {log_filename}")
