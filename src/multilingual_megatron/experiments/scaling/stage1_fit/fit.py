"""Pure scaling models, variable projection, and deterministic numerical starts."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Hashable, Iterator, Literal, Mapping, Sequence
import numpy as np
ModelName = Literal[
    'C0', 'J0', 'JH', 'M0', 'M1', 'M1-PhiE', 'M1-PsiE', 'M2',
    'M2-KN', 'M2-KD', 'M2-KND', 'M3', 'M4', 'M5', 'M6',
]
N_SCALE = 100000000.0
D_SCALE = 1000000000.0
ATLAS_TRANSFER_TAU_SCALE = 0.03
ATLAS_TRANSFER_TAU_EXPANDED_MAXIMUM = 0.1
E_MIN = 1.0
E_MAX = 32.0
LOG_E_SPAN = math.log(E_MAX / E_MIN)
PRIMARY_MODELS: tuple[ModelName, ...] = ('M0', 'M1', 'M2')
LITERATURE_MODELS: tuple[ModelName, ...] = ('C0', 'J0', 'JH')
EXPLORATORY_INTERACTION_MODELS: tuple[ModelName, ...] = ('M2-KN', 'M2-KD', 'M2-KND')
EXPLORATORY_EXPOSURE_MODELS: tuple[ModelName, ...] = ('M3', 'M4')
EXPLORATORY_COMPOSITION_MODELS: tuple[ModelName, ...] = ('M5', 'M6')
EXPLORATORY_M1_ABLATION_MODELS: tuple[ModelName, ...] = ('M1-PhiE', 'M1-PsiE')
M5_M6_REGULARIZATION_SCHEMA = 'stage1_m5_m6_ridge/v1'
M5_M6_PHI_ENDPOINT_SCALE = 0.05
M5_M6_PSI_ENDPOINT_SCALE = 0.05
M5_M6_GAMMA_N_SCALE = 0.10
JOINT_MOE_E_START = 2.0732
JOINT_MOE_E_MAX = 290.4521
D16_ADAPTIVE_PROTOCOL = 'stage1_fit_d16_adaptive_starts/v1'
D16_ADAPTIVE_START_TARGETS = (32, 128, 512)
OUTER_PARAMETER_NAMES: dict[ModelName, tuple[str, ...]] = {
    'C0': ('alpha_e1', 'beta_e1'),
    'J0': ('alpha_e1', 'alpha_e32', 'beta_e1', 'beta_e32', 'delta', 'omega'),
    'JH': ('alpha_e1', 'alpha_e32', 'beta_e1', 'beta_e32', 'delta', 'omega'),
    'M0': ('alpha_e1', 'beta_e1', 'phi_e1', 'psi_e1'),
    'M1': ('alpha_e1', 'alpha_e32', 'beta_e1', 'beta_e32', 'delta', 'omega', 'phi_e1', 'psi_e1'),
    'M1-PhiE': ('alpha_e1', 'alpha_e32', 'beta_e1', 'beta_e32', 'delta', 'omega', 'phi_e1', 'phi_e32', 'psi_e1'),
    'M1-PsiE': ('alpha_e1', 'alpha_e32', 'beta_e1', 'beta_e32', 'delta', 'omega', 'phi_e1', 'psi_e1', 'psi_e32'),
    'M2': ('alpha_e1', 'alpha_e32', 'beta_e1', 'beta_e32', 'delta', 'omega', 'phi_e1', 'phi_e32', 'psi_e1', 'psi_e32'),
    'M2-KN': ('alpha_e1', 'alpha_e32', 'beta_e1', 'beta_e32', 'delta', 'omega', 'phi_e1', 'phi_e32', 'psi_e1', 'psi_e32', 'gamma_n'),
    'M2-KD': ('alpha_e1', 'alpha_e32', 'beta_e1', 'beta_e32', 'delta', 'omega', 'phi_e1', 'phi_e32', 'psi_e1', 'psi_e32', 'gamma_d'),
    'M2-KND': ('alpha_e1', 'alpha_e32', 'beta_e1', 'beta_e32', 'delta', 'omega', 'phi_e1', 'phi_e32', 'psi_e1', 'psi_e32', 'gamma_n', 'gamma_d'),
    'M3': ('alpha_e1', 'alpha_e32', 'beta_e1', 'beta_e32', 'delta', 'omega', 'phi_e1', 'phi_e32', 'psi_e1', 'psi_e32', 'tau_other_u'),
    'M4': ('alpha_e1', 'alpha_e32', 'beta_e1', 'beta_e32', 'delta', 'omega', 'phi_e1', 'phi_e32', 'psi_e1', 'psi_e32', 'tau_transfer_u', 'tau_other_u')}
OUTER_PARAMETER_NAMES['M5'] = OUTER_PARAMETER_NAMES['M2-KN']
OUTER_PARAMETER_NAMES['M6'] = (*OUTER_PARAMETER_NAMES['M2-KN'],
    'eta_family', 'eta_script', 'gamma_other')
ParameterKind = Literal['positive', 'signed', 'nonnegative']
PARAMETER_KINDS: dict[str, ParameterKind] = {'alpha_e1': 'positive', 'alpha_e32': 'positive', 'beta_e1': 'positive', 'beta_e32': 'positive', 'delta': 'signed', 'omega': 'signed', 'phi_e1': 'signed', 'phi_e32': 'signed', 'psi_e1': 'signed', 'psi_e32': 'signed', 'gamma_n': 'signed', 'gamma_d': 'signed', 'tau_transfer_u': 'nonnegative', 'tau_other_u': 'nonnegative', 'eta_family': 'nonnegative', 'eta_script': 'nonnegative', 'gamma_other': 'nonnegative'}

@dataclass(frozen=True)
class SharedParameters:
    """Registered shared coefficients, including zeros for absent effects."""
    alpha_0: float
    alpha_1: float
    beta_0: float
    beta_1: float
    delta: float = 0.0
    omega: float = 0.0
    phi_0: float = 0.0
    phi_1: float = 0.0
    psi_0: float = 0.0
    psi_1: float = 0.0
    gamma_n: float = 0.0
    gamma_d: float = 0.0
    tau_transfer: float = 0.0
    tau_other: float = 0.0
    eta_family: float = 0.0
    eta_script: float = 0.0
    gamma_other: float = 0.0

    def as_dict(self) -> dict[str, float]:
        # Preserve the registered M0--M2 serialization exactly.  The executor
        # adds transfer weights only for explicitly exploratory M3/M4 records.
        return {name: float(getattr(self, name)) for name in ('alpha_0', 'alpha_1', 'beta_0', 'beta_1', 'delta', 'omega', 'phi_0', 'phi_1', 'psi_0', 'psi_1', 'gamma_n', 'gamma_d')}

@dataclass(frozen=True)
class M5M6Regularization:
    """Manifest-bound ridge contract shared by M5 and M6."""
    strength: float
    m2_reference_objective: float
    schema_version: str = M5_M6_REGULARIZATION_SCHEMA
    phi_endpoint_scale: float = M5_M6_PHI_ENDPOINT_SCALE
    psi_endpoint_scale: float = M5_M6_PSI_ENDPOINT_SCALE
    gamma_n_scale: float = M5_M6_GAMMA_N_SCALE

    def __post_init__(self) -> None:
        values = (self.strength, self.m2_reference_objective,
            self.phi_endpoint_scale, self.psi_endpoint_scale, self.gamma_n_scale)
        if self.schema_version != M5_M6_REGULARIZATION_SCHEMA:
            raise ValueError('M5/M6 regularization schema changed')
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError('M5/M6 regularization values must be finite')
        if self.strength < 0.0 or self.m2_reference_objective <= 0.0:
            raise ValueError('M5/M6 ridge lambda must be non-negative and the M2 reference objective positive')
        if (self.phi_endpoint_scale != M5_M6_PHI_ENDPOINT_SCALE
                or self.psi_endpoint_scale != M5_M6_PSI_ENDPOINT_SCALE
                or self.gamma_n_scale != M5_M6_GAMMA_N_SCALE):
            raise ValueError('M5/M6 ridge scales changed')

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> 'M5M6Regularization':
        required = {'schema_version', 'lambda', 'm2_reference_objective',
            'phi_endpoint_scale', 'psi_endpoint_scale', 'gamma_n_scale'}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError('M5/M6 regularization keys changed')
        numeric = required - {'schema_version'}
        if any(isinstance(value[name], bool) or not isinstance(value[name], (int, float))
                for name in numeric):
            raise ValueError('M5/M6 regularization values must be JSON numbers')
        return cls(strength=float(value['lambda']),
            m2_reference_objective=float(value['m2_reference_objective']),
            schema_version=str(value['schema_version']),
            phi_endpoint_scale=float(value['phi_endpoint_scale']),
            psi_endpoint_scale=float(value['psi_endpoint_scale']),
            gamma_n_scale=float(value['gamma_n_scale']))

    def as_dict(self) -> dict[str, float | str]:
        return {'schema_version': self.schema_version, 'lambda': self.strength,
            'm2_reference_objective': self.m2_reference_objective,
            'phi_endpoint_scale': self.phi_endpoint_scale,
            'psi_endpoint_scale': self.psi_endpoint_scale,
            'gamma_n_scale': self.gamma_n_scale}

@dataclass(frozen=True)
class ObjectiveComponents:
    data_fit: float
    regularization: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {'data_fit': float(self.data_fit),
            'regularization': float(self.regularization), 'total': float(self.total)}

def _validate_model(model: str) -> ModelName:
    if model not in OUTER_PARAMETER_NAMES:
        raise ValueError(f'unknown Stage 1 fit model: {model!r}')
    return model

def outer_parameter_names(model: str) -> tuple[str, ...]:
    """Return the stable optimizer-coordinate order for ``model``."""
    return OUTER_PARAMETER_NAMES[_validate_model(model)]

def outer_parameter_kinds(model: str) -> tuple[ParameterKind, ...]:
    return tuple((PARAMETER_KINDS[name] for name in outer_parameter_names(model)))

def line_from_endpoints(value_e1: float, value_e32: float, *, e_max: float=E_MAX) -> tuple[float, float]:
    """Return ``(intercept, log-E slope)`` from values at E=1 and E=e_max."""
    values = np.asarray([value_e1, value_e32, e_max], dtype=np.float64)
    if not np.isfinite(values).all() or e_max <= 1.0:
        raise ValueError('endpoint values must be finite and e_max must exceed one')
    return (float(value_e1), float((value_e32 - value_e1) / math.log(e_max)))

def endpoints_from_line(intercept: float, slope: float, *, e_max: float=E_MAX) -> tuple[float, float]:
    """Inverse of :func:`line_from_endpoints`."""
    values = np.asarray([intercept, slope, e_max], dtype=np.float64)
    if not np.isfinite(values).all() or e_max <= 1.0:
        raise ValueError('line values must be finite and e_max must exceed one')
    return (float(intercept), float(intercept + slope * math.log(e_max)))

def _coordinates(model: ModelName, values: Mapping[str, float] | Sequence[float] | np.ndarray) -> np.ndarray:
    names = outer_parameter_names(model)
    if isinstance(values, Mapping):
        missing = [name for name in names if name not in values]
        extra = sorted(set(values) - set(names))
        if missing or extra:
            raise ValueError(f'coordinate mismatch for {model}: missing={missing}, extra={extra}')
        array = np.asarray([values[name] for name in names], dtype=np.float64)
    else:
        array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size != len(names):
        raise ValueError(f'{model} requires {len(names)} outer coordinates')
    if array.dtype != np.dtype(np.float64) or not np.isfinite(array).all():
        raise ValueError('outer coordinates must be finite float64 values')
    return array

def shared_from_outer(model: str, values: Mapping[str, float] | Sequence[float] | np.ndarray) -> SharedParameters:
    """Translate endpoint optimizer coordinates to registered coefficients."""
    checked_model = _validate_model(model)
    coordinates = _coordinates(checked_model, values)
    p = dict(zip(outer_parameter_names(checked_model), coordinates.tolist()))
    alpha_e1 = p['alpha_e1']
    alpha_e32 = p.get('alpha_e32', alpha_e1)
    beta_e1 = p['beta_e1']
    beta_e32 = p.get('beta_e32', beta_e1)
    alpha_0, alpha_1 = line_from_endpoints(alpha_e1, alpha_e32)
    beta_0, beta_1 = line_from_endpoints(beta_e1, beta_e32)
    phi_e1 = p.get('phi_e1', 0.0)
    phi_e32 = p.get('phi_e32', phi_e1)
    psi_e1 = p.get('psi_e1', 0.0)
    psi_e32 = p.get('psi_e32', psi_e1)
    phi_0, phi_1 = line_from_endpoints(phi_e1, phi_e32)
    psi_0, psi_1 = line_from_endpoints(psi_e1, psi_e32)
    tau_transfer = ATLAS_TRANSFER_TAU_SCALE * float(p.get('tau_transfer_u', 0.0))
    tau_other = ATLAS_TRANSFER_TAU_SCALE * float(p.get('tau_other_u', 0.0))
    if checked_model == 'M3':
        tau_transfer = tau_other
    return SharedParameters(alpha_0=alpha_0, alpha_1=alpha_1, beta_0=beta_0, beta_1=beta_1, delta=float(p.get('delta', 0.0)), omega=float(p.get('omega', 0.0)), phi_0=phi_0, phi_1=phi_1, psi_0=psi_0, psi_1=psi_1, gamma_n=float(p.get('gamma_n', 0.0)), gamma_d=float(p.get('gamma_d', 0.0)), tau_transfer=tau_transfer, tau_other=tau_other, eta_family=float(p.get('eta_family', 0.0)), eta_script=float(p.get('eta_script', 0.0)), gamma_other=float(p.get('gamma_other', 0.0)))

def outer_from_shared(model: str, parameters: SharedParameters) -> np.ndarray:
    """Return endpoint coordinates for a nested warm start or round-trip test."""
    checked_model = _validate_model(model)
    alpha_e1, alpha_e32 = endpoints_from_line(parameters.alpha_0, parameters.alpha_1)
    beta_e1, beta_e32 = endpoints_from_line(parameters.beta_0, parameters.beta_1)
    phi_e1, phi_e32 = endpoints_from_line(parameters.phi_0, parameters.phi_1)
    psi_e1, psi_e32 = endpoints_from_line(parameters.psi_0, parameters.psi_1)
    lookup = {'alpha_e1': alpha_e1, 'alpha_e32': alpha_e32, 'beta_e1': beta_e1, 'beta_e32': beta_e32, 'delta': parameters.delta, 'omega': parameters.omega, 'phi_e1': phi_e1, 'phi_e32': phi_e32, 'psi_e1': psi_e1, 'psi_e32': psi_e32, 'gamma_n': parameters.gamma_n, 'gamma_d': parameters.gamma_d, 'tau_transfer_u': parameters.tau_transfer / ATLAS_TRANSFER_TAU_SCALE, 'tau_other_u': parameters.tau_other / ATLAS_TRANSFER_TAU_SCALE, 'eta_family': parameters.eta_family, 'eta_script': parameters.eta_script, 'gamma_other': parameters.gamma_other}
    return np.asarray([lookup[name] for name in outer_parameter_names(checked_model)], dtype=np.float64)

def regularization_objective_and_gradient(model: str, values: Any,
    regularization: M5M6Regularization | None) -> tuple[float, np.ndarray]:
    """Return the locked M5/M6 ridge value and outer-coordinate gradient."""
    checked_model = _validate_model(model)
    coordinates = _coordinates(checked_model, values)
    if checked_model not in EXPLORATORY_COMPOSITION_MODELS:
        if regularization is not None:
            raise ValueError(f'{checked_model} does not accept M5/M6 regularization')
        return (0.0, np.zeros_like(coordinates, dtype=np.float64))
    if not isinstance(regularization, M5M6Regularization):
        raise ValueError(f'{checked_model} requires manifest-bound M5/M6 regularization')
    names = outer_parameter_names(checked_model)
    lookup = dict(zip(names, coordinates.tolist()))
    phi_difference = lookup['phi_e32'] - lookup['phi_e1']
    psi_difference = lookup['psi_e32'] - lookup['psi_e1']
    gamma_n = lookup['gamma_n']
    bracket = ((phi_difference / regularization.phi_endpoint_scale) ** 2
        + (psi_difference / regularization.psi_endpoint_scale) ** 2
        + (gamma_n / regularization.gamma_n_scale) ** 2)
    prefactor = regularization.strength * regularization.m2_reference_objective / 2.0
    objective = float(prefactor * bracket)
    gradient = np.zeros_like(coordinates, dtype=np.float64)
    phi_derivative = (regularization.strength * regularization.m2_reference_objective
        * phi_difference / (regularization.phi_endpoint_scale ** 2))
    psi_derivative = (regularization.strength * regularization.m2_reference_objective
        * psi_difference / (regularization.psi_endpoint_scale ** 2))
    gamma_derivative = (regularization.strength * regularization.m2_reference_objective
        * gamma_n / (regularization.gamma_n_scale ** 2))
    gradient[names.index('phi_e1')] = -phi_derivative
    gradient[names.index('phi_e32')] = phi_derivative
    gradient[names.index('psi_e1')] = -psi_derivative
    gradient[names.index('psi_e32')] = psi_derivative
    gradient[names.index('gamma_n')] = gamma_derivative
    if not math.isfinite(objective) or not np.isfinite(gradient).all():
        raise FloatingPointError('M5/M6 regularization overflowed')
    return (objective, gradient)

def data_column(model: str) -> Literal['D_t', 'D_total']:
    """Return the frozen data predictor for one registered form."""
    return 'D_total' if _validate_model(model) == 'JH' else 'D_t'

def effective_exposure_data(model: str, parameters: SharedParameters, D_target: Any,
    D_transfer: Any | None = None, D_other: Any | None = None) -> np.ndarray:
    """Return the target or ATLAS-inspired effective data coordinate.

    ``D_transfer`` and ``D_other`` are required only for M3/M4.  Both are
    immutable, loss-blind exposure components; the fitted coordinates are
    non-negative weights in physical units.
    """
    checked_model = _validate_model(model)
    target = np.asarray(D_target, dtype=np.float64)
    if not np.isfinite(target).all() or np.any(target <= 0.0):
        raise ValueError('target exposure must be finite and strictly positive')
    if checked_model not in EXPLORATORY_EXPOSURE_MODELS:
        return target
    if D_transfer is None or D_other is None:
        raise ValueError(f'{checked_model} requires transfer and remainder exposure')
    try:
        target, transfer, other = tuple(np.asarray(item, dtype=np.float64)
            for item in np.broadcast_arrays(target, D_transfer, D_other))
    except ValueError as error:
        raise ValueError('exposure components must be broadcast-compatible') from error
    if not np.isfinite(transfer).all() or not np.isfinite(other).all() \
            or np.any(transfer < 0.0) or np.any(other < 0.0):
        raise ValueError('non-target exposure components must be finite and non-negative')
    expanded_limit = float(np.nextafter(np.float64(ATLAS_TRANSFER_TAU_EXPANDED_MAXIMUM), np.float64(np.inf)))
    if not all(math.isfinite(value) and 0.0 <= value <= expanded_limit
            for value in (parameters.tau_transfer, parameters.tau_other)):
        raise ValueError('transfer weights must be finite and within the frozen expanded interval')
    if checked_model == 'M3':
        # Keep the arithmetic order identical to the M4 diagonal so the
        # M3-in-M4 nesting is bit-exact as well as algebraically exact.
        result = target + np.float64(parameters.tau_other) * transfer \
            + np.float64(parameters.tau_other) * other
    else:
        result = target + np.float64(parameters.tau_transfer) * transfer \
            + np.float64(parameters.tau_other) * other
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise FloatingPointError('effective exposure is non-finite or non-positive')
    return np.asarray(result, dtype=np.float64)

def composition_exposure_coordinates(parameters: SharedParameters, D_target: Any,
    D_family: Any, D_script: Any, D_other: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return M6 effective data and its three immutable log coordinates."""
    try:
        target, family, script, other = tuple(np.asarray(item, dtype=np.float64)
            for item in np.broadcast_arrays(D_target, D_family, D_script, D_other))
    except ValueError as error:
        raise ValueError('M6 exposure components must be broadcast-compatible') from error
    if (not all(np.isfinite(item).all() for item in (target, family, script, other))
            or np.any(target <= 0.0)
            or any(np.any(item < 0.0) for item in (family, script, other))):
        raise ValueError('M6 target exposure must be positive and composition exposures non-negative')
    if not (math.isfinite(parameters.eta_family)
            and 0.0 <= parameters.eta_family <= 1.0
            and math.isfinite(parameters.eta_script)
            and 0.0 <= parameters.eta_script <= 1.0
            and math.isfinite(parameters.gamma_other)
            and 0.0 <= parameters.gamma_other <= 0.5):
        raise ValueError('M6 composition parameters lie outside their locked bounds')
    log_family = np.asarray(np.log1p(family / target), dtype=np.float64)
    log_script = np.asarray(np.log1p(script / target), dtype=np.float64)
    denominator = target + family + script
    if not np.isfinite(denominator).all() or np.any(denominator <= 0.0):
        raise FloatingPointError('M6 capacity-composition denominator overflowed')
    log_other = np.asarray(np.log1p(other / denominator), dtype=np.float64)
    effective = np.asarray(target * np.exp(
        np.float64(parameters.eta_family) * log_family
        + np.float64(parameters.eta_script) * log_script), dtype=np.float64)
    if (not all(np.isfinite(item).all()
            for item in (log_family, log_script, log_other, effective))
            or np.any(effective <= 0.0)):
        raise FloatingPointError('M6 exposure transformation overflowed')
    return (effective, log_family, log_script, log_other)

def transformed_experts(experts: Any) -> np.ndarray:
    """Joint-MoE's published saturating expert-count transformation."""
    e = np.asarray(experts, dtype=np.float64)
    if not np.isfinite(e).all() or np.any(e <= 0.0):
        raise ValueError('expert counts must be finite and strictly positive')
    bridge = 1.0 / (1.0 / JOINT_MOE_E_START - 1.0 / JOINT_MOE_E_MAX)
    result = 1.0 / (1.0 / (e - 1.0 + bridge) + 1.0 / JOINT_MOE_E_MAX)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise FloatingPointError('transformed expert counts are invalid')
    return np.asarray(result, dtype=np.float64)

def expert_log_coordinate(model: str, experts: Any) -> np.ndarray:
    """Return log-E, normalized at E=1/32 for the published-input sensitivity."""
    checked_model = _validate_model(model)
    e = np.asarray(experts, dtype=np.float64)
    if not np.isfinite(e).all() or np.any(e <= 0.0):
        raise ValueError('expert counts must be finite and strictly positive')
    if checked_model != 'JH':
        return np.asarray(np.log(e), dtype=np.float64)
    log_hat = np.log(transformed_experts(e))
    lower = math.log(JOINT_MOE_E_START)
    upper = math.log(float(transformed_experts(np.asarray(E_MAX))))
    return np.asarray((log_hat - lower) * LOG_E_SPAN / (upper - lower), dtype=np.float64)

def multilingual_k_coordinate(language_counts: Any) -> np.ndarray:
    """Return the exploratory K12-anchored multilingual interaction coordinate.

    K1 is an English-only control and K12 is the multilingual reference, so
    neither informs the new slope.  K50 lies between zero and the K200 endpoint.
    """
    k = np.asarray(language_counts, dtype=np.float64)
    if not np.isfinite(k).all() or np.any(k <= 0.0):
        raise ValueError('language counts must be finite and strictly positive')
    coordinate = np.where(k <= 12.0, 0.0, np.log(k / 12.0) / math.log(200.0 / 12.0))
    return np.asarray(coordinate, dtype=np.float64)

def effective_exponents(parameters: SharedParameters, experts: Any, language_counts: Any | None=None, *, model: str='M2', require_positive_exponents: bool=True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``alpha(E,K), beta(E,K), phi(E), psi(E)`` as float64 arrays."""
    log_e = expert_log_coordinate(model, experts)
    alpha = np.asarray(parameters.alpha_0 + parameters.alpha_1 * log_e, dtype=np.float64)
    beta = np.asarray(parameters.beta_0 + parameters.beta_1 * log_e, dtype=np.float64)
    if model in EXPLORATORY_INTERACTION_MODELS + EXPLORATORY_COMPOSITION_MODELS:
        if language_counts is None:
            raise ValueError('exploratory interaction exponents require language counts')
        coordinate = multilingual_k_coordinate(language_counts)
        alpha = np.asarray(alpha + parameters.gamma_n * coordinate, dtype=np.float64)
        beta = np.asarray(beta + parameters.gamma_d * coordinate, dtype=np.float64)
        if require_positive_exponents and (np.any(alpha <= 0.0) or np.any(beta <= 0.0)):
            raise FloatingPointError('exploratory effective decay exponent is non-positive')
    return (alpha, beta, np.asarray(parameters.phi_0 + parameters.phi_1 * log_e, dtype=np.float64), np.asarray(parameters.psi_0 + parameters.psi_1 * log_e, dtype=np.float64))

def _predictor_arrays(N: Any, D: Any, E: Any, K: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple((np.asarray(value, dtype=np.float64) for value in (N, D, E, K)))
    try:
        broadcast = tuple((np.asarray(value, dtype=np.float64) for value in np.broadcast_arrays(*arrays)))
    except ValueError as error:
        raise ValueError('N, D, E, and K must be broadcast-compatible') from error
    if any((not value.flags.c_contiguous and value.ndim == 1 for value in broadcast)):
        broadcast = tuple((np.ascontiguousarray(value, dtype=np.float64) for value in broadcast))
    if not all((np.isfinite(value).all() for value in broadcast)):
        raise ValueError('predictors must be finite')
    if any((np.any(value <= 0.0) for value in broadcast)):
        raise ValueError('N, D, E, and K must be strictly positive')
    return broadcast

def predictor_terms(model: str, parameters: SharedParameters, N: Any, D: Any, E: Any, K: Any,
    *, n_scale: float=N_SCALE, d_scale: float=D_SCALE,
    D_family: Any | None=None, D_script: Any | None=None,
    D_other: Any | None=None, require_positive_exponents: bool=True) -> tuple[np.ndarray, np.ndarray]:
    """Return one registered law's unit-coefficient capacity and data terms."""
    checked_model = _validate_model(model)
    if not (math.isfinite(n_scale) and n_scale > 0.0):
        raise ValueError('n_scale must be finite and positive')
    if not (math.isfinite(d_scale) and d_scale > 0.0):
        raise ValueError('d_scale must be finite and positive')
    composition_log_other: np.ndarray | None = None
    if checked_model == 'M6':
        if D_family is None or D_script is None or D_other is None:
            raise ValueError('M6 requires family, script, and other exposure')
        effective_d, _, _, composition_log_other = composition_exposure_coordinates(
            parameters, D, D_family, D_script, D_other)
        n, d, e, k = _predictor_arrays(N, effective_d, E, K)
    else:
        if D_family is not None or D_script is not None or D_other is not None:
            raise ValueError(f'{checked_model} does not accept M6 composition exposure')
        n, d, e, k = _predictor_arrays(N, D, E, K)
    log_n = np.log(n / np.float64(n_scale))
    log_d = np.log(d / np.float64(d_scale))
    log_e = expert_log_coordinate(checked_model, e)
    log_k = np.log(k)
    alpha, beta, phi, psi = effective_exponents(parameters, e, k, model=checked_model, require_positive_exponents=require_positive_exponents)
    log_capacity = parameters.delta * log_e + phi * log_k - alpha * log_n
    if composition_log_other is not None:
        log_capacity = np.asarray(log_capacity + np.float64(parameters.gamma_other)
            * np.broadcast_to(composition_log_other, log_capacity.shape), dtype=np.float64)
    log_data = parameters.omega * log_e + psi * log_k - beta * log_d
    capacity = np.asarray(np.exp(log_capacity), dtype=np.float64)
    data = np.asarray(np.exp(log_data), dtype=np.float64)
    if not np.isfinite(capacity).all() or not np.isfinite(data).all():
        raise FloatingPointError('model predictor terms overflowed')
    return (capacity, data)

def predictor_terms_and_jacobian(model: str, values: Mapping[str, float] | Sequence[float] | np.ndarray, N: Any, D: Any, E: Any, K: Any, *, n_scale: float=N_SCALE, d_scale: float=D_SCALE, D_transfer: Any | None=None, D_other: Any | None=None, D_family: Any | None=None, D_script: Any | None=None, require_positive_exponents: bool=True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return terms and analytic derivatives with respect to outer endpoints."""
    checked_model = _validate_model(model)
    coordinates = _coordinates(checked_model, values)
    parameters = shared_from_outer(checked_model, coordinates)
    composition_coordinates: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if checked_model == 'M6':
        if D_family is None or D_script is None or D_other is None:
            raise ValueError('M6 requires family, script, and other exposure')
        effective_d, log_family, log_script, log_other = composition_exposure_coordinates(
            parameters, D, D_family, D_script, D_other)
        n, d, e, k = _predictor_arrays(N, effective_d, E, K)
        capacity, data = predictor_terms(checked_model, parameters, N, D, E, K,
            n_scale=n_scale, d_scale=d_scale, D_family=D_family,
            D_script=D_script, D_other=D_other, require_positive_exponents=require_positive_exponents)
        composition_coordinates = tuple(np.broadcast_to(item, capacity.shape)
            for item in (log_family, log_script, log_other))
    else:
        if D_family is not None or D_script is not None:
            raise ValueError(f'{checked_model} does not accept M6 composition exposure')
        effective_d = effective_exposure_data(checked_model, parameters, D, D_transfer, D_other)
        n, d, e, k = _predictor_arrays(N, effective_d, E, K)
        capacity, data = predictor_terms(checked_model, parameters, n, d, e, k, n_scale=n_scale, d_scale=d_scale, require_positive_exponents=require_positive_exponents)
    shape = capacity.shape
    flat_n = np.log(n / np.float64(n_scale)).reshape(-1)
    flat_d = np.log(d / np.float64(d_scale)).reshape(-1)
    flat_e = expert_log_coordinate(checked_model, e).reshape(-1)
    flat_k = np.log(k).reshape(-1)
    q = flat_e / np.float64(LOG_E_SPAN)
    one_minus_q = 1.0 - q
    names = outer_parameter_names(checked_model)
    capacity_log_jac = np.zeros((flat_n.size, len(names)), dtype=np.float64)
    data_log_jac = np.zeros_like(capacity_log_jac)
    for column, name in enumerate(names):
        if name == 'alpha_e1':
            capacity_log_jac[:, column] = -(flat_n if 'alpha_e32' not in names else one_minus_q * flat_n)
        elif name == 'alpha_e32':
            capacity_log_jac[:, column] = -q * flat_n
        elif name == 'beta_e1':
            data_log_jac[:, column] = -(flat_d if 'beta_e32' not in names else one_minus_q * flat_d)
        elif name == 'beta_e32':
            data_log_jac[:, column] = -q * flat_d
        elif name == 'delta':
            capacity_log_jac[:, column] = flat_e
        elif name == 'omega':
            data_log_jac[:, column] = flat_e
        elif name == 'phi_e1':
            capacity_log_jac[:, column] = flat_k if 'phi_e32' not in names else one_minus_q * flat_k
        elif name == 'phi_e32':
            capacity_log_jac[:, column] = q * flat_k
        elif name == 'psi_e1':
            data_log_jac[:, column] = flat_k if 'psi_e32' not in names else one_minus_q * flat_k
        elif name == 'psi_e32':
            data_log_jac[:, column] = q * flat_k
        elif name == 'gamma_n':
            capacity_log_jac[:, column] = -multilingual_k_coordinate(k).reshape(-1) * flat_n
        elif name == 'gamma_d':
            data_log_jac[:, column] = -multilingual_k_coordinate(k).reshape(-1) * flat_d
        elif name == 'eta_family':
            assert composition_coordinates is not None
            beta = effective_exponents(parameters, e, k, model=checked_model, require_positive_exponents=require_positive_exponents)[1].reshape(-1)
            data_log_jac[:, column] = -beta * composition_coordinates[0].reshape(-1)
        elif name == 'eta_script':
            assert composition_coordinates is not None
            beta = effective_exponents(parameters, e, k, model=checked_model, require_positive_exponents=require_positive_exponents)[1].reshape(-1)
            data_log_jac[:, column] = -beta * composition_coordinates[1].reshape(-1)
        elif name == 'gamma_other':
            assert composition_coordinates is not None
            capacity_log_jac[:, column] = composition_coordinates[2].reshape(-1)
        elif name in {'tau_transfer_u', 'tau_other_u'}:
            transfer = np.asarray(D_transfer, dtype=np.float64)
            other = np.asarray(D_other, dtype=np.float64)
            transfer, other, effective = tuple(np.asarray(item, dtype=np.float64).reshape(-1)
                for item in np.broadcast_arrays(transfer, other, d))
            beta = effective_exponents(parameters, e, k, model=checked_model, require_positive_exponents=require_positive_exponents)[1].reshape(-1)
            component = transfer if name == 'tau_transfer_u' else (
                transfer + other if checked_model == 'M3' else other)
            data_log_jac[:, column] = -beta * np.float64(ATLAS_TRANSFER_TAU_SCALE) * component / effective
        else:
            raise AssertionError(f'unhandled outer parameter {name}')
    capacity_jac = capacity.reshape(-1, 1) * capacity_log_jac
    data_jac = data.reshape(-1, 1) * data_log_jac
    return (capacity, data, capacity_jac.reshape(shape + (len(names),)), data_jac.reshape(shape + (len(names),)))

def predict(model: str, parameters: SharedParameters, N: Any, D: Any, E: Any, K: Any, L_inf: Any, A: Any, B: Any, *, n_scale: float=N_SCALE, d_scale: float=D_SCALE, D_family: Any | None=None, D_script: Any | None=None, D_other: Any | None=None) -> np.ndarray:
    """Evaluate one registered model with already-aligned nuisance coefficients."""
    capacity, data = predictor_terms(model, parameters, N, D, E, K,
        n_scale=n_scale, d_scale=d_scale, D_family=D_family,
        D_script=D_script, D_other=D_other)
    floor, coefficient_a, coefficient_b = np.broadcast_arrays(np.asarray(L_inf, dtype=np.float64), np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64), capacity)[:3]
    if not all((np.isfinite(value).all() for value in (floor, coefficient_a, coefficient_b))):
        raise ValueError('language coefficients must be finite')
    if np.any(coefficient_a < 0.0) or np.any(coefficient_b < 0.0):
        raise ValueError('A and B must be non-negative')
    result = np.asarray(floor + coefficient_a * capacity + coefficient_b * data, dtype=np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError('model prediction overflowed')
    return result

MAD_NORMAL_SCALE = 1.4826
HUBER_NORMAL_TRANSITION = 1.345

@dataclass(frozen=True)
class RobustScale:
    """Training-only M0 residual scale and the resulting fold objective."""
    location: float
    weighted_mad: float
    scale: float
    delta: float | None
    use_ols: bool

def _float64_vector(values: Any, *, name: str, allow_empty: bool=False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or (not allow_empty and array.size == 0):
        qualifier = 'one-dimensional' if allow_empty else 'non-empty and one-dimensional'
        raise ValueError(f'{name} must be {qualifier}')
    if array.dtype != np.dtype(np.float64) or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain finite float64 values')
    return array

def _validated_weights(weights: Any, size: int) -> np.ndarray:
    array = _float64_vector(weights, name='weights')
    if array.size != size:
        raise ValueError('weights and values must have the same length')
    if np.any(array < 0.0) or not float(array.sum()) > 0.0:
        raise ValueError('weights must be non-negative with positive total')
    return array

def trajectory_weights(trajectory_ids: Sequence[Hashable]) -> np.ndarray:
    """Give every trajectory total weight ``1/T`` and make rows sum to one."""
    identifiers = tuple(trajectory_ids)
    if not identifiers:
        raise ValueError('at least one trajectory row is required')
    counts: dict[Hashable, int] = {}
    for identifier in identifiers:
        try:
            counts[identifier] = counts.get(identifier, 0) + 1
        except TypeError as error:
            raise ValueError('trajectory identifiers must be hashable') from error
    trajectories = len(counts)
    result = np.asarray([1.0 / (trajectories * counts[identifier]) for identifier in identifiers], dtype=np.float64)
    result /= np.float64(result.sum())
    return result

def weighted_median(values: Any, weights: Any) -> float:
    """Return the first stable-sorted value reaching half normalized weight."""
    array = _float64_vector(values, name='values')
    weight_array = _validated_weights(weights, array.size)
    order = np.argsort(array, kind='stable')
    ordered_values = array[order]
    ordered_weights = weight_array[order] / np.float64(weight_array.sum())
    cumulative = np.cumsum(ordered_weights, dtype=np.float64)
    index = int(np.searchsorted(cumulative, np.float64(0.5), side='left'))
    return float(ordered_values[min(index, ordered_values.size - 1)])

def weighted_mad(values: Any, weights: Any) -> float:
    """Weighted median absolute deviation around the weighted median."""
    array = _float64_vector(values, name='values')
    weight_array = _validated_weights(weights, array.size)
    center = weighted_median(array, weight_array)
    return weighted_median(np.abs(array - np.float64(center)), weight_array)

def weighted_mad_scale(values: Any, weights: Any) -> float:
    return float(MAD_NORMAL_SCALE * weighted_mad(values, weights))

def adaptive_huber_scale(values: Any, weights: Any) -> RobustScale:
    """Apply the frozen weighted-MAD rule, including exact-zero OLS fallback."""
    array = _float64_vector(values, name='residuals')
    weight_array = _validated_weights(weights, array.size)
    location = weighted_median(array, weight_array)
    mad = weighted_median(np.abs(array - np.float64(location)), weight_array)
    scale = float(MAD_NORMAL_SCALE * mad)
    if scale == 0.0:
        return RobustScale(location=location, weighted_mad=mad, scale=0.0, delta=None, use_ols=True)
    delta = float(HUBER_NORMAL_TRANSITION * scale)
    return RobustScale(location=location, weighted_mad=mad, scale=scale, delta=delta, use_ols=False)

def huber_loss(residuals: Any, delta: float) -> np.ndarray:
    """Elementwise standard Huber rho under the D9 convention."""
    residual = _float64_vector(residuals, name='residuals', allow_empty=True)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError('Huber delta must be finite and strictly positive')
    absolute = np.abs(residual)
    result = np.where(absolute <= np.float64(delta), 0.5 * residual * residual, np.float64(delta) * (absolute - 0.5 * np.float64(delta)))
    return np.asarray(result, dtype=np.float64)

def huber_score(residuals: Any, delta: float) -> np.ndarray:
    """Analytic derivative of :func:`huber_loss` with respect to residual."""
    residual = _float64_vector(residuals, name='residuals', allow_empty=True)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError('Huber delta must be finite and strictly positive')
    return np.asarray(np.clip(residual, -np.float64(delta), np.float64(delta)), dtype=np.float64)

def loss_and_score(residuals: Any, delta: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Return the sole per-row objective and derivative used by all solvers.

    ``delta=None`` is the explicit zero-MAD/OLS path.  A numeric zero is not
    accepted as a Huber threshold, which prevents an accidental all-zero loss.
    """
    residual = _float64_vector(residuals, name='residuals', allow_empty=True)
    if delta is None:
        return (np.asarray(0.5 * residual * residual, dtype=np.float64), residual.copy())
    return (huber_loss(residual, delta), huber_score(residual, delta))

def weighted_objective(residuals: Any, weights: Any, delta: float | None) -> float:
    residual = _float64_vector(residuals, name='residuals')
    weight_array = _validated_weights(weights, residual.size)
    losses, _ = loss_and_score(residual, delta)
    return float(np.dot(weight_array, losses))

def weighted_objective_and_gradient(residuals: Any, prediction_jacobian: Any, weights: Any, delta: float | None) -> tuple[float, np.ndarray]:
    """Objective and analytic parameter gradient for ``prediction-y`` rows."""
    residual = _float64_vector(residuals, name='residuals')
    weight_array = _validated_weights(weights, residual.size)
    jacobian = np.asarray(prediction_jacobian, dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.shape[0] != residual.size:
        raise ValueError('prediction_jacobian must have one row per residual')
    if not np.isfinite(jacobian).all():
        raise ValueError('prediction_jacobian must be finite')
    losses, score = loss_and_score(residual, delta)
    objective = float(np.dot(weight_array, losses))
    gradient = jacobian.T @ (weight_array * score)
    return (objective, np.asarray(gradient, dtype=np.float64))

def weighted_rmse(residuals: Any, weights: Any) -> float:
    residual = _float64_vector(residuals, name='residuals')
    weight_array = _validated_weights(weights, residual.size)
    return float(np.sqrt(np.dot(weight_array, residual * residual) / weight_array.sum()))

def weighted_r2(observed: Any, predicted: Any, weights: Any) -> float:
    y = _float64_vector(observed, name='observed')
    prediction = _float64_vector(predicted, name='predicted')
    if prediction.size != y.size:
        raise ValueError('observed and predicted lengths differ')
    weight_array = _validated_weights(weights, y.size)
    mean = float(np.dot(weight_array, y) / weight_array.sum())
    total = float(np.dot(weight_array, (y - mean) ** 2))
    if total == 0.0:
        return float('nan')
    error = float(np.dot(weight_array, (prediction - y) ** 2))
    return float(1.0 - error / total)
INNER_FTOL = 1e-13
INNER_GTOL = 1e-08
INNER_MAX_ITERATIONS = 2000
FLOOR_IDENTIFIED_LOWER = 0.01
FLOOR_IDENTIFIED_UPPER = 0.99

class InnerSolveError(RuntimeError):
    """Raised when an inner problem violates the contract or does not converge."""

@dataclass(frozen=True)
class LanguageProblem:
    """One fixed-basis, three-coefficient convex regression problem."""
    y: np.ndarray
    capacity: np.ndarray
    data: np.ndarray
    weights: np.ndarray
    physical_floor: float
    training_minimum: float

    def __post_init__(self) -> None:
        arrays = {name: np.asarray(value, dtype=np.float64) for name, value in (('y', self.y), ('capacity', self.capacity), ('data', self.data), ('weights', self.weights))}
        sizes = {value.size for value in arrays.values()}
        if any((value.ndim != 1 for value in arrays.values())) or sizes != {arrays['y'].size}:
            raise ValueError('inner arrays must be aligned one-dimensional vectors')
        if arrays['y'].size == 0 or not all((np.isfinite(value).all() for value in arrays.values())):
            raise ValueError('inner arrays must be non-empty and finite')
        if np.any(arrays['capacity'] <= 0.0) or np.any(arrays['data'] <= 0.0):
            raise ValueError('inner predictor bases must be strictly positive')
        if np.any(arrays['weights'] < 0.0) or not arrays['weights'].sum() > 0.0:
            raise ValueError('inner weights must be non-negative with positive total')
        if not (math.isfinite(self.physical_floor) and math.isfinite(self.training_minimum) and (self.physical_floor < self.training_minimum)):
            raise ValueError('physical_floor must be below the training minimum')
        for name, value in arrays.items():
            object.__setattr__(self, name, np.asarray(value, dtype=np.float64))

    @property
    def design(self) -> np.ndarray:
        return np.column_stack((np.ones(self.y.size, dtype=np.float64), self.capacity, self.data))

    @property
    def strict_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower_floor = np.nextafter(np.float64(self.physical_floor), np.float64(self.training_minimum))
        upper_floor = np.nextafter(np.float64(self.training_minimum), np.float64(self.physical_floor))
        strict_positive = np.nextafter(np.float64(0.0), np.float64(np.inf))
        return (np.asarray([lower_floor, strict_positive, strict_positive], dtype=np.float64), np.asarray([upper_floor, np.inf, np.inf], dtype=np.float64))

@dataclass(frozen=True)
class LanguageSolution:
    language: str | None
    L_inf: float
    A: float
    B: float
    objective: float
    prediction: np.ndarray
    residual: np.ndarray
    success: bool
    status: int
    message: str
    iterations: int
    function_evaluations: int
    gradient_infinity_norm: float
    floor_fraction: float
    floor_saturated: bool
    A_zero_contact: bool
    B_zero_contact: bool

@dataclass(frozen=True)
class ProfileSolution:
    objective: float
    prediction: np.ndarray
    residual: np.ndarray
    coefficients: tuple[LanguageSolution, ...]
    success: bool

    def by_language(self) -> dict[str, LanguageSolution]:
        return {item.language: item for item in self.coefficients if item.language is not None}

def language_objective_and_gradient(coefficients: Any, problem: LanguageProblem, delta: float | None) -> tuple[float, np.ndarray]:
    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError('language coefficients must be three finite float64 values')
    prediction = problem.design @ values
    residual = prediction - problem.y
    return weighted_objective_and_gradient(residual, problem.design, problem.weights, delta)

def constrained_least_squares_initialization(problem: LanguageProblem) -> np.ndarray:
    """Return the declared bounded trajectory-weighted LS initializer."""
    from scipy.optimize import lsq_linear
    square_root_weight = np.sqrt(problem.weights)
    design = square_root_weight[:, None] * problem.design
    response = square_root_weight * problem.y
    lower, upper = problem.strict_bounds
    result = lsq_linear(design, response, bounds=(lower, upper), method='trf', tol=1e-13, max_iter=2000, lsmr_tol='auto')
    if not result.success or not np.isfinite(result.x).all():
        raise InnerSolveError(f'constrained least-squares initialization failed: {result.message}')
    return np.asarray(result.x, dtype=np.float64)

def _solution_from_result(*, problem: LanguageProblem, values: np.ndarray, delta: float | None, success: bool, status: int, message: str, iterations: int, function_evaluations: int, language: str | None) -> LanguageSolution:
    prediction = np.asarray(problem.design @ values, dtype=np.float64)
    residual = np.asarray(prediction - problem.y, dtype=np.float64)
    objective, gradient = language_objective_and_gradient(values, problem, delta)
    interval = problem.training_minimum - problem.physical_floor
    fraction = float((values[0] - problem.physical_floor) / interval)
    zero_tolerance = 1e-12
    return LanguageSolution(language=language, L_inf=float(values[0]), A=float(values[1]), B=float(values[2]), objective=objective, prediction=prediction, residual=residual, success=bool(success), status=int(status), message=str(message), iterations=int(iterations), function_evaluations=int(function_evaluations), gradient_infinity_norm=float(np.max(np.abs(gradient))), floor_fraction=fraction, floor_saturated=not FLOOR_IDENTIFIED_LOWER <= fraction <= FLOOR_IDENTIFIED_UPPER, A_zero_contact=bool(values[1] <= zero_tolerance), B_zero_contact=bool(values[2] <= zero_tolerance))

def solve_language(problem: LanguageProblem, delta: float | None, *, language: str | None=None, initial: Any | None=None, ftol: float=INNER_FTOL, gtol: float=INNER_GTOL, maximum_iterations: int=INNER_MAX_ITERATIONS, require_success: bool=True) -> LanguageSolution:
    """Solve one direct-coefficient bounded OLS/Huber inner problem."""
    from scipy.optimize import minimize
    if delta is not None and (not math.isfinite(delta) or delta <= 0.0):
        raise ValueError('delta must be None (OLS) or finite and positive')
    if not (math.isfinite(ftol) and ftol > 0.0):
        raise ValueError('ftol must be finite and positive')
    if not (math.isfinite(gtol) and gtol > 0.0):
        raise ValueError('gtol must be finite and positive')
    if maximum_iterations < 1:
        raise ValueError('maximum_iterations must be positive')
    lower, upper = problem.strict_bounds
    if initial is None:
        start = constrained_least_squares_initialization(problem)
    else:
        start = np.asarray(initial, dtype=np.float64)
        if start.shape != (3,) or not np.isfinite(start).all():
            raise ValueError('inner initial value must contain three finite values')
        start = np.minimum(np.maximum(start, lower), upper)
    result = minimize(lambda values: language_objective_and_gradient(values, problem, delta), start, method='L-BFGS-B', jac=True, bounds=list(zip(lower.tolist(), upper.tolist())), options={'ftol': float(ftol), 'gtol': float(gtol), 'maxiter': int(maximum_iterations), 'maxls': 50})
    solution = _solution_from_result(problem=problem, values=np.asarray(result.x, dtype=np.float64), delta=delta, success=bool(result.success), status=int(result.status), message=str(result.message), iterations=int(result.nit), function_evaluations=int(result.nfev), language=language)
    if require_success and (not solution.success):
        label = '' if language is None else f' for {language}'
        raise InnerSolveError(f'inner solve failed{label}: {solution.message}')
    return solution

def solve_language_reference(problem: LanguageProblem, delta: float | None, *, maximum_iterations: int=5000) -> LanguageSolution:
    """Independent SLSQP/numerical-gradient reference for small audit cases."""
    from scipy.optimize import minimize
    start = constrained_least_squares_initialization(problem)
    lower, upper = problem.strict_bounds
    result = minimize(lambda values: weighted_objective(problem.design @ values - problem.y, problem.weights, delta), start, method='SLSQP', jac=None, bounds=list(zip(lower.tolist(), upper.tolist())), options={'ftol': 1e-13, 'maxiter': int(maximum_iterations)})
    return _solution_from_result(problem=problem, values=np.asarray(result.x, dtype=np.float64), delta=delta, success=bool(result.success), status=int(result.status), message=str(result.message), iterations=int(result.nit), function_evaluations=int(result.nfev), language='reference')

def solve_all_languages(*, y: Any, capacity: Any, data: Any, weights: Any, language_ids: Sequence[str], physical_floors: Any, delta: float | None, require_success: bool=True) -> ProfileSolution:
    """Profile all languages independently and restore original row order."""
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in (('y', y), ('capacity', capacity), ('data', data), ('weights', weights), ('physical_floors', physical_floors))}
    size = arrays['y'].size
    if any((value.ndim != 1 or value.size != size for value in arrays.values())):
        raise ValueError('profile arrays must be aligned one-dimensional vectors')
    if size == 0 or not all((np.isfinite(value).all() for value in arrays.values())):
        raise ValueError('profile arrays must be non-empty and finite')
    identifiers = np.asarray(tuple(language_ids), dtype=str)
    if identifiers.ndim != 1 or identifiers.size != size or np.any(identifiers == ''):
        raise ValueError('language_ids must contain one non-empty identifier per row')
    prediction = np.empty(size, dtype=np.float64)
    solutions: list[LanguageSolution] = []
    for language in sorted(set(identifiers.tolist())):
        indices = np.flatnonzero(identifiers == language)
        floors = arrays['physical_floors'][indices]
        if not np.all(floors == floors[0]):
            raise ValueError(f'physical adjusted-loss floor changed within {language}')
        problem = LanguageProblem(y=arrays['y'][indices], capacity=arrays['capacity'][indices], data=arrays['data'][indices], weights=arrays['weights'][indices], physical_floor=float(floors[0]), training_minimum=float(np.min(arrays['y'][indices])))
        solution = solve_language(problem, delta, language=language, require_success=require_success)
        prediction[indices] = solution.prediction
        solutions.append(solution)
    residual = np.asarray(prediction - arrays['y'], dtype=np.float64)
    objective = weighted_objective(residual, arrays['weights'], delta)
    success = all((item.success for item in solutions))
    if require_success and (not success):
        raise InnerSolveError('one or more language solves failed')
    return ProfileSolution(objective=objective, prediction=prediction, residual=residual, coefficients=tuple(solutions), success=success)
POSITIVE_START_VALUES = (0.05, 0.25, 0.5)
SIGNED_START_VALUES = (-0.5, 0.0, 0.5)
NONNEGATIVE_START_VALUES = (0.0, 0.1, 1.0)
START_GRID_ORDERING_VERSION = 'stage1_fit_lexicographic_endpoints/v1'
EXPECTED_PRIMARY_START_COUNTS = {'C0': 9, 'J0': 729, 'JH': 729, 'M0': 81, 'M1': 6561, 'M2': 59049}
OUTER_FTOL = 1e-12
OUTER_GTOL = 1e-06
OUTER_MAX_ITERATIONS = 10000
BOUNDARY_RELATIVE_WIDTH = 0.0001
MAXIMUM_BOUND_EXPANSIONS = 3
OBJECTIVE_RELATIVE_TOLERANCE = 1e-06
PREDICTION_ABSOLUTE_TOLERANCE = 0.0001
MINIMUM_BEST_BASIN_STARTS = 3
DIRECT_PENALTY_OBJECTIVE = 1e+100
DIRECT_PENALTY_PREDICTION = 1e+50

class ProjectedOptimizationError(RuntimeError):
    """Raised for a rejected outer evaluation or malformed search problem."""

@dataclass(frozen=True)
class ProjectedProblem:
    """Immutable array-only input for one model/fold/replicate fit."""
    y: np.ndarray
    weights: np.ndarray
    N: np.ndarray
    D: np.ndarray
    E: np.ndarray
    K: np.ndarray
    language_ids: tuple[str, ...]
    physical_floors: np.ndarray
    D_transfer: np.ndarray | None = None
    D_other: np.ndarray | None = None
    D_family: np.ndarray | None = None
    D_script: np.ndarray | None = None
    regularization: M5M6Regularization | None = None

    def __post_init__(self) -> None:
        arrays = {name: np.asarray(value, dtype=np.float64) for name, value in (('y', self.y), ('weights', self.weights), ('N', self.N), ('D', self.D), ('E', self.E), ('K', self.K), ('physical_floors', self.physical_floors))}
        if self.D_transfer is not None:
            if self.D_other is None or self.D_family is not None or self.D_script is not None:
                raise ValueError('provide exactly the M3/M4 or M6 exposure component set')
            arrays['D_transfer'] = np.asarray(self.D_transfer, dtype=np.float64)
            arrays['D_other'] = np.asarray(self.D_other, dtype=np.float64)
        elif any(item is not None for item in (self.D_family, self.D_script, self.D_other)):
            if any(item is None for item in (self.D_family, self.D_script, self.D_other)):
                raise ValueError('provide family, script, and other exposure together')
            arrays['D_family'] = np.asarray(self.D_family, dtype=np.float64)
            arrays['D_script'] = np.asarray(self.D_script, dtype=np.float64)
            arrays['D_other'] = np.asarray(self.D_other, dtype=np.float64)
        size = arrays['y'].size
        if size == 0 or any((value.ndim != 1 or value.size != size for value in arrays.values())):
            raise ValueError('projected problem arrays must be aligned vectors')
        if not all((np.isfinite(value).all() for value in arrays.values())):
            raise ValueError('projected problem arrays must be finite')
        if any((np.any(arrays[name] <= 0.0) for name in ('N', 'D', 'E', 'K'))):
            raise ValueError('N, D, E, and K must be strictly positive')
        if any(np.any(arrays[name] < 0.0)
                for name in ('D_transfer', 'D_family', 'D_script', 'D_other')
                if name in arrays):
            raise ValueError('non-target exposure components must be non-negative')
        if np.any(arrays['weights'] < 0.0) or not arrays['weights'].sum() > 0.0:
            raise ValueError('weights must be non-negative with positive total')
        identifiers = tuple((str(value) for value in self.language_ids))
        if len(identifiers) != size or any((not value for value in identifiers)):
            raise ValueError('language_ids must contain one non-empty value per row')
        for language in sorted(set(identifiers)):
            indices = [index for index, value in enumerate(identifiers) if value == language]
            floors = arrays['physical_floors'][indices]
            if not np.all(floors == floors[0]):
                raise ValueError(f'physical floor changed within language {language}')
            if not float(floors[0]) < float(np.min(arrays['y'][indices])):
                raise ValueError(f'physical floor is not below training losses for {language}')
        for name, value in arrays.items():
            object.__setattr__(self, name, np.asarray(value, dtype=np.float64))
        object.__setattr__(self, 'language_ids', identifiers)
        if self.regularization is not None and not isinstance(
                self.regularization, M5M6Regularization):
            if not isinstance(self.regularization, Mapping):
                raise ValueError('regularization must be an M5M6Regularization contract')
            object.__setattr__(self, 'regularization',
                M5M6Regularization.from_mapping(self.regularization))

    @classmethod
    def from_arrays(cls, *, y: Any, weights: Any, N: Any, D: Any, E: Any, K: Any, language_ids: Sequence[str], physical_floors: Any | None=None, unigram_nll: Any | None=None, D_transfer: Any | None=None, D_other: Any | None=None, D_family: Any | None=None, D_script: Any | None=None, regularization: M5M6Regularization | Mapping[str, Any] | None=None) -> 'ProjectedProblem':
        if (physical_floors is None) == (unigram_nll is None):
            raise ValueError('provide exactly one of physical_floors or unigram_nll')
        floors = np.asarray(physical_floors, dtype=np.float64) if physical_floors is not None else -np.asarray(unigram_nll, dtype=np.float64)
        return cls(y=np.asarray(y, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64), N=np.asarray(N, dtype=np.float64), D=np.asarray(D, dtype=np.float64), E=np.asarray(E, dtype=np.float64), K=np.asarray(K, dtype=np.float64), language_ids=tuple(language_ids), physical_floors=floors, D_transfer=None if D_transfer is None else np.asarray(D_transfer, dtype=np.float64), D_other=None if D_other is None else np.asarray(D_other, dtype=np.float64), D_family=None if D_family is None else np.asarray(D_family, dtype=np.float64), D_script=None if D_script is None else np.asarray(D_script, dtype=np.float64), regularization=regularization)

@dataclass(frozen=True)
class StartPoint:
    start_id: str
    start_index: int | None
    values: np.ndarray
    kind: str = 'grid'

@dataclass(frozen=True)
class OuterBound:
    name: str
    lower: float
    upper: float
    kind: str
    expansions: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {'positive', 'signed', 'nonnegative'}:
            raise ValueError(f'unknown bound kind: {self.kind}')
        if not (math.isfinite(self.lower) and math.isfinite(self.upper) and (self.lower < self.upper)):
            raise ValueError('bounds must form a finite non-empty interval')
        if self.kind == 'positive' and self.lower <= 0.0:
            raise ValueError('positive-decay lower bounds must exceed zero')
        if self.kind == 'nonnegative' and self.lower < 0.0:
            raise ValueError('non-negative lower bounds cannot be negative')
        if self.expansions < 0:
            raise ValueError('bound expansions cannot be negative')

@dataclass(frozen=True)
class BoundaryContact:
    name: str
    side: str
    value: float
    distance: float
    interval_width: float

@dataclass(frozen=True)
class BoundExpansion:
    bounds: tuple[OuterBound, ...]
    contacts: tuple[BoundaryContact, ...]
    nonidentified: tuple[str, ...]

@dataclass(frozen=True)
class ProjectedEvaluation:
    coordinates: np.ndarray
    shared_parameters: SharedParameters
    objective: float
    prediction: np.ndarray
    profile: ProfileSolution
    objective_components: ObjectiveComponents | None = None

@dataclass(frozen=True)
class ProjectedStartResult:
    model: str
    start_id: str
    start_index: int | None
    success: bool
    status: int
    message: str
    objective: float | None
    coordinates: np.ndarray
    shared_parameters: SharedParameters | None
    prediction: np.ndarray | None
    profile: ProfileSolution | None
    iterations: int
    function_evaluations: int
    boundary_contacts: tuple[BoundaryContact, ...]
    objective_components: ObjectiveComponents | None = None

@dataclass(frozen=True)
class EquivalenceResult:
    equivalent: bool
    relative_objective_difference: float
    maximum_absolute_prediction_difference: float

@dataclass(frozen=True)
class BasinResult:
    best: ProjectedStartResult
    members: tuple[ProjectedStartResult, ...]
    resolved: bool

@dataclass(frozen=True)
class DirectFitResult:
    success: bool
    status: int
    message: str
    objective: float
    prediction: np.ndarray
    outer_coordinates: np.ndarray
    shared_parameters: SharedParameters
    language_coefficients: dict[str, dict[str, float]]
    iterations: int
    function_evaluations: int
    objective_components: ObjectiveComponents | None = None

@dataclass(frozen=True)
class DirectCheckResult:
    direct: DirectFitResult
    equivalence: EquivalenceResult
    retried: bool
    resolved: bool

def start_count(model: str) -> int:
    """Return a count without materializing the Cartesian product."""
    count = 3 ** len(outer_parameter_names(model))
    expected = EXPECTED_PRIMARY_START_COUNTS.get(model)
    if expected is not None and count != expected:
        raise AssertionError(f'start-grid count changed for {model}')
    return count

def start_at(model: str, start_index: int) -> StartPoint:
    """Index the lexicographic Cartesian grid; the last coordinate varies fastest."""
    count = start_count(model)
    if isinstance(start_index, bool) or not isinstance(start_index, int):
        raise TypeError('start_index must be an integer')
    if not 0 <= start_index < count:
        raise IndexError(f'start_index {start_index} is outside [0,{count})')
    kinds = outer_parameter_kinds(model)
    digits = [0] * len(kinds)
    remainder = start_index
    for position in range(len(kinds) - 1, -1, -1):
        remainder, digits[position] = divmod(remainder, 3)
    starts_by_kind = {'positive': POSITIVE_START_VALUES, 'signed': SIGNED_START_VALUES,
        'nonnegative': NONNEGATIVE_START_VALUES}
    values = np.asarray([
        ((0.0, 0.03, 0.10)
            if name in {'eta_family', 'eta_script', 'gamma_other'}
            else starts_by_kind[kind])[digit]
        for name, kind, digit in zip(outer_parameter_names(model), kinds, digits)
    ], dtype=np.float64)
    return StartPoint(start_id=f'grid:{start_index:06d}', start_index=start_index, values=values)

def d16_adaptive_tiers(model: str) -> tuple[int, ...]:
    """Return the frozen cumulative D16 start targets and full-grid fallback."""
    full = start_count(model)
    return tuple(target for target in D16_ADAPTIVE_START_TARGETS if target < full) + (full,)

def d16_adaptive_starts(model: str, center: Any, basin_coordinates: Sequence[Any],
    bounds: Sequence[OuterBound], *, target: int) -> tuple[StartPoint, ...]:
    """Build a deterministic, nested D16 start set around the all-data basin.

    The full-data optimum and up to three distinct basin representatives come
    first.  The remaining starts are unscrambled Halton perturbations with
    progressively wider radii.  A failed tier ultimately falls back to the
    registered Cartesian grid; this helper never approximates that fallback.
    """
    checked_model = _validate_model(model)
    tiers = d16_adaptive_tiers(checked_model)
    if target not in tiers[:-1]:
        raise ValueError(f'adaptive target must be one of {tiers[:-1]} for {model}')
    active = tuple(bounds)
    if tuple(bound.name for bound in active) != outer_parameter_names(checked_model):
        raise ValueError('D16 bounds do not match the model coordinate order')
    lower = np.asarray([bound.lower for bound in active], dtype=np.float64)
    upper = np.asarray([bound.upper for bound in active], dtype=np.float64)
    widths = upper - lower
    origin = np.clip(_coordinates(checked_model, center), lower, upper)
    starts: list[StartPoint] = []
    seen: set[bytes] = set()

    def append(start_id: str, values: Any, kind: str) -> None:
        candidate = np.clip(_coordinates(checked_model, values), lower, upper)
        key = np.asarray(candidate, dtype=np.float64).tobytes()
        if key not in seen and len(starts) < target:
            seen.add(key)
            starts.append(StartPoint(start_id, None, candidate, kind))

    append('d16:full-data-optimum', origin, 'd16_anchor')
    for index, values in enumerate(basin_coordinates[:3]):
        append(f'd16:full-data-basin:{index}', values, 'd16_anchor')

    from scipy.stats import qmc
    engine = qmc.Halton(d=len(active), scramble=False)
    source_index = 0
    while len(starts) < target:
        point = engine.random(1)[0]
        radius = 0.02 if source_index < 64 else (0.08 if source_index < 256 else 0.25)
        append(f'd16:halton:{source_index:06d}', origin + (2.0 * point - 1.0) * radius * widths, 'd16_perturbation')
        source_index += 1
        if source_index > 100_000:
            raise RuntimeError('could not construct the frozen D16 start set')
    return tuple(starts)

def iter_starts(model: str, begin: int=0, end_exclusive: int | None=None) -> Iterator[StartPoint]:
    count = start_count(model)
    end = count if end_exclusive is None else end_exclusive
    if not 0 <= begin <= end <= count:
        raise ValueError('invalid contiguous start-index interval')
    for index in range(begin, end):
        yield start_at(model, index)

def nested_warm_start(target_model: str, source: SharedParameters) -> StartPoint:
    if target_model not in {'M1', 'M2'}:
        raise ValueError('nested warm starts are defined only for M0->M1 and M1->M2')
    return StartPoint(start_id=f"warm:{('M0' if target_model == 'M1' else 'M1')}", start_index=start_count(target_model), values=outer_from_shared(target_model, source), kind='warm')

def initial_bounds(model: str) -> tuple[OuterBound, ...]:
    limits = {'positive': (0.01, 2.0), 'signed': (-1.0, 1.0), 'nonnegative': (0.0, 1.0)}
    coordinate_limits = {'eta_family': (0.0, 1.0), 'eta_script': (0.0, 1.0),
        'gamma_other': (0.0, 0.5)}
    return tuple(OuterBound(name=name,
        lower=coordinate_limits.get(name, limits[kind])[0],
        upper=coordinate_limits.get(name, limits[kind])[1], kind=kind)
        for name, kind in zip(outer_parameter_names(model), outer_parameter_kinds(model)))

def boundary_contacts(values: Any, bounds: Sequence[OuterBound], *, relative_width: float=BOUNDARY_RELATIVE_WIDTH) -> tuple[BoundaryContact, ...]:
    coordinates = np.asarray(values, dtype=np.float64)
    if coordinates.shape != (len(bounds),) or not np.isfinite(coordinates).all():
        raise ValueError('one finite coordinate is required for each bound')
    if not (math.isfinite(relative_width) and relative_width >= 0.0):
        raise ValueError('relative_width must be finite and non-negative')
    result: list[BoundaryContact] = []
    for value, bound in zip(coordinates, bounds):
        width = bound.upper - bound.lower
        if value < bound.lower - 1e-12 * width or value > bound.upper + 1e-12 * width:
            raise ValueError(f'coordinate {bound.name} lies outside its bounds')
        lower_distance = max(0.0, float(value - bound.lower))
        upper_distance = max(0.0, float(bound.upper - value))
        threshold = relative_width * width
        if lower_distance <= threshold:
            result.append(BoundaryContact(bound.name, 'lower', float(value), lower_distance, width))
        if upper_distance <= threshold:
            result.append(BoundaryContact(bound.name, 'upper', float(value), upper_distance, width))
    return tuple(result)

def expand_contacted_bounds(bounds: Sequence[OuterBound], contacts: Sequence[BoundaryContact], *, maximum_expansions: int=MAXIMUM_BOUND_EXPANSIONS) -> BoundExpansion:
    """Expand only contacted coordinates and report exhausted coordinates."""
    if maximum_expansions < 0:
        raise ValueError('maximum_expansions cannot be negative')
    by_name: dict[str, set[str]] = {}
    for contact in contacts:
        by_name.setdefault(contact.name, set()).add(contact.side)
    unknown = sorted(set(by_name) - {bound.name for bound in bounds})
    if unknown:
        raise ValueError(f'contacts reference unknown bounds: {unknown}')
    expanded: list[OuterBound] = []
    nonidentified: list[str] = []
    for bound in bounds:
        sides = by_name.get(bound.name)
        if not sides:
            expanded.append(bound)
            continue
        if bound.expansions >= maximum_expansions:
            expanded.append(bound)
            nonidentified.append(bound.name)
            continue
        if bound.kind == 'signed':
            radius = 2.0 * max(abs(bound.lower), abs(bound.upper))
            updated = OuterBound(bound.name, -radius, radius, bound.kind, bound.expansions + 1)
        elif bound.name in {'eta_family', 'eta_script', 'gamma_other'}:
            # These scientific coordinates have frozen physical intervals;
            # a contacted upper cap is non-identification, not an invitation
            # to reuse the M3/M4 tau expansion rule.
            expanded.append(bound)
            if 'upper' in sides:
                nonidentified.append(bound.name)
            continue
        elif bound.kind == 'nonnegative':
            # Zero is the meaningful nested null and is never expanded.  The
            # only allowed upper expansion maps tau=0.03 to the frozen 0.1 cap.
            if 'upper' not in sides or bound.expansions >= 1:
                expanded.append(bound)
                if 'upper' in sides and bound.expansions >= 1:
                    nonidentified.append(bound.name)
                continue
            updated = OuterBound(bound.name, 0.0,
                ATLAS_TRANSFER_TAU_EXPANDED_MAXIMUM / ATLAS_TRANSFER_TAU_SCALE,
                bound.kind, bound.expansions + 1)
        else:
            lower = bound.lower / 10.0 if 'lower' in sides else bound.lower
            upper = bound.upper * 2.0 if 'upper' in sides else bound.upper
            updated = OuterBound(bound.name, lower, upper, bound.kind, bound.expansions + 1)
        expanded.append(updated)
    return BoundExpansion(tuple(expanded), tuple(contacts), tuple(nonidentified))

def deterministic_local_perturbations(center: Any, bounds: Sequence[OuterBound], *, relative_step: float=0.02) -> tuple[StartPoint, ...]:
    """Generate deterministic +/- coordinate moves for a failed basin audit."""
    values = np.asarray(center, dtype=np.float64)
    if values.shape != (len(bounds),) or not np.isfinite(values).all():
        raise ValueError('center must align with bounds')
    if not (math.isfinite(relative_step) and relative_step > 0.0):
        raise ValueError('relative_step must be finite and positive')
    result: list[StartPoint] = []
    for index, bound in enumerate(bounds):
        width = bound.upper - bound.lower
        margin = np.finfo(np.float64).eps * width
        for sign, label in ((-1.0, 'minus'), (1.0, 'plus')):
            candidate = values.copy()
            candidate[index] = np.clip(candidate[index] + sign * relative_step * width, bound.lower + margin, bound.upper - margin)
            result.append(StartPoint(start_id=f'local:{index}:{label}', start_index=None, values=np.asarray(candidate, dtype=np.float64), kind='local'))
    return tuple(result)

def evaluate_projected(model: str, coordinates: Any, problem: ProjectedProblem, delta: float | None) -> ProjectedEvaluation:
    values = np.asarray(coordinates, dtype=np.float64)
    shared = shared_from_outer(model, values)
    if model in EXPLORATORY_COMPOSITION_MODELS:
        regularization_objective, _ = regularization_objective_and_gradient(
            model, values, problem.regularization)
        if model == 'M6':
            capacity, data = predictor_terms(model, shared, problem.N, problem.D,
                problem.E, problem.K, D_family=problem.D_family,
                D_script=problem.D_script, D_other=problem.D_other)
        else:
            capacity, data = predictor_terms(model, shared, problem.N, problem.D,
                problem.E, problem.K)
    else:
        if problem.regularization is not None:
            raise ValueError(f'{model} does not accept M5/M6 regularization')
        effective_d = effective_exposure_data(model, shared, problem.D, problem.D_transfer, problem.D_other)
        capacity, data = predictor_terms(model, shared, problem.N, effective_d, problem.E, problem.K)
    profile = solve_all_languages(y=problem.y, capacity=capacity, data=data, weights=problem.weights, language_ids=problem.language_ids, physical_floors=problem.physical_floors, delta=delta, require_success=True)
    if not math.isfinite(profile.objective) or not np.isfinite(profile.prediction).all():
        raise ProjectedOptimizationError('profiled evaluation is non-finite')
    if model in EXPLORATORY_COMPOSITION_MODELS:
        total = float(profile.objective + regularization_objective)
        components = ObjectiveComponents(data_fit=profile.objective,
            regularization=regularization_objective, total=total)
        return ProjectedEvaluation(coordinates=values.copy(), shared_parameters=shared,
            objective=total, prediction=profile.prediction, profile=profile,
            objective_components=components)
    return ProjectedEvaluation(coordinates=values.copy(), shared_parameters=shared,
        objective=profile.objective, prediction=profile.prediction, profile=profile)

def _coerce_start(model: str, start: StartPoint | Sequence[float] | np.ndarray) -> StartPoint:
    if isinstance(start, StartPoint):
        point = start
    else:
        point = StartPoint('explicit', None, np.asarray(start, dtype=np.float64), 'explicit')
    values = np.asarray(point.values, dtype=np.float64)
    if values.shape != (len(outer_parameter_names(model)),) or not np.isfinite(values).all():
        raise ValueError("start does not match the model's outer coordinates")
    return StartPoint(point.start_id, point.start_index, values, point.kind)

def run_projected_start(model: str, start: StartPoint | Sequence[float] | np.ndarray, problem: ProjectedProblem, *, delta: float | None, bounds: Sequence[OuterBound] | None=None, ftol: float=OUTER_FTOL, gtol: float=OUTER_GTOL, maximum_iterations: int=OUTER_MAX_ITERATIONS) -> ProjectedStartResult:
    """Run exactly one projected L-BFGS-B start and retain failures explicitly."""
    from scipy.optimize import minimize
    point = _coerce_start(model, start)
    active_bounds = tuple(initial_bounds(model) if bounds is None else bounds)
    names = outer_parameter_names(model)
    if tuple((bound.name for bound in active_bounds)) != names:
        raise ValueError('outer bounds do not match the model coordinate order')
    scipy_bounds = [(bound.lower, bound.upper) for bound in active_bounds]
    for value, bound in zip(point.values, active_bounds):
        if not bound.lower <= value <= bound.upper:
            raise ValueError(f'start coordinate {bound.name} lies outside its bounds')
    cache: dict[bytes, float] = {}

    def objective(values: np.ndarray) -> float:
        key = np.asarray(values, dtype=np.float64).tobytes()
        if key not in cache:
            cache[key] = evaluate_projected(model, values, problem, delta).objective
        return cache[key]
    try:
        result = minimize(objective, point.values, method='L-BFGS-B', jac=None, bounds=scipy_bounds, options={'ftol': float(ftol), 'gtol': float(gtol), 'maxiter': int(maximum_iterations), 'maxls': 50})
        final = evaluate_projected(model, result.x, problem, delta)
        contacts = boundary_contacts(result.x, active_bounds)
        return ProjectedStartResult(model=model, start_id=point.start_id, start_index=point.start_index, success=bool(result.success), status=int(result.status), message=str(result.message), objective=final.objective, coordinates=np.asarray(result.x, dtype=np.float64), shared_parameters=final.shared_parameters, prediction=final.prediction, profile=final.profile, iterations=int(result.nit), function_evaluations=int(result.nfev), boundary_contacts=contacts, objective_components=final.objective_components)
    except (InnerSolveError, ProjectedOptimizationError, FloatingPointError) as error:
        return ProjectedStartResult(model=model, start_id=point.start_id, start_index=point.start_index, success=False, status=-1, message=f'outer evaluation rejected: {error}', objective=None, coordinates=point.values.copy(), shared_parameters=None, prediction=None, profile=None, iterations=0, function_evaluations=len(cache), boundary_contacts=())

def relative_objective_difference(first: float, second: float) -> float:
    if not (math.isfinite(first) and math.isfinite(second)):
        return float('inf')
    denominator = max(abs(first), abs(second), np.finfo(np.float64).tiny)
    return float(abs(first - second) / denominator)

def equivalence_check(first_objective: float, second_objective: float, first_prediction: Any, second_prediction: Any, *, objective_tolerance: float=OBJECTIVE_RELATIVE_TOLERANCE, prediction_tolerance: float=PREDICTION_ABSOLUTE_TOLERANCE) -> EquivalenceResult:
    first = np.asarray(first_prediction, dtype=np.float64)
    second = np.asarray(second_prediction, dtype=np.float64)
    if first.shape != second.shape or first.size == 0:
        raise ValueError('equivalence predictions must have the same non-empty shape')
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return EquivalenceResult(False, float('inf'), float('inf'))
    relative = relative_objective_difference(first_objective, second_objective)
    maximum = float(np.max(np.abs(first - second)))
    return EquivalenceResult(equivalent=relative <= objective_tolerance and maximum <= prediction_tolerance, relative_objective_difference=relative, maximum_absolute_prediction_difference=maximum)

def select_best_start(results: Sequence[ProjectedStartResult]) -> ProjectedStartResult:
    successful = [result for result in results if result.success and result.objective is not None and math.isfinite(result.objective)]
    if not successful:
        raise ProjectedOptimizationError('no converged finite optimizer start')
    return min(successful, key=lambda result: (float(result.objective), result.start_id))

def best_basin(results: Sequence[ProjectedStartResult], *, minimum_members: int=MINIMUM_BEST_BASIN_STARTS) -> BasinResult:
    if minimum_members < 1:
        raise ValueError('minimum_members must be positive')
    best = select_best_start(results)
    assert best.objective is not None and best.prediction is not None
    members: list[ProjectedStartResult] = []
    for result in results:
        if not result.success or result.objective is None or result.prediction is None:
            continue
        if equivalence_check(best.objective, result.objective, best.prediction, result.prediction).equivalent:
            members.append(result)
    members.sort(key=lambda result: result.start_id)
    return BasinResult(best, tuple(members), len(members) >= minimum_members)

def floor_is_saturated(fraction: float) -> bool:
    if not math.isfinite(fraction):
        raise ValueError('floor fraction must be finite')
    return not 0.01 <= fraction <= 0.99

def _direct_initial_vector(projected: ProjectedStartResult) -> tuple[np.ndarray, tuple[str, ...]]:
    if not projected.success or projected.profile is None:
        raise ProjectedOptimizationError('direct check requires a successful projected fit')
    languages = tuple(sorted(projected.profile.by_language()))
    values = list(np.asarray(projected.coordinates, dtype=np.float64))
    epsilon = np.finfo(np.float64).eps
    for language in languages:
        coefficient = projected.profile.by_language()[language]
        fraction = float(np.clip(coefficient.floor_fraction, epsilon, 1.0 - epsilon))
        c = math.log(fraction / (1.0 - fraction))
        a = math.log(max(coefficient.A, np.finfo(np.float64).eps))
        b = math.log(max(coefficient.B, np.finfo(np.float64).eps))
        values.extend((c, a, b))
    return (np.asarray(values, dtype=np.float64), languages)

def _direct_objective_gradient(values: np.ndarray, *, model: str, problem: ProjectedProblem, languages: tuple[str, ...], delta: float | None) -> tuple[float, np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    from scipy.special import expit
    outer_size = len(outer_parameter_names(model))
    outer = np.asarray(values[:outer_size], dtype=np.float64)
    if values.size != outer_size + 3 * len(languages):
        raise ValueError('direct joint vector has the wrong size')

    def rejected() -> tuple[float, np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
        return (DIRECT_PENALTY_OBJECTIVE, np.zeros_like(values, dtype=np.float64), np.full(problem.y.shape, DIRECT_PENALTY_PREDICTION, dtype=np.float64), {})
    if not np.isfinite(values).all():
        return rejected()
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        capacity, data, capacity_jac, data_jac = predictor_terms_and_jacobian(model, outer,
            problem.N, problem.D, problem.E, problem.K,
            D_transfer=problem.D_transfer, D_other=problem.D_other,
            D_family=problem.D_family, D_script=problem.D_script)
    if not all((np.isfinite(item).all() for item in (capacity, data, capacity_jac, data_jac))):
        return rejected()
    identifiers = np.asarray(problem.language_ids, dtype=str)
    L_inf = np.empty(problem.y.size, dtype=np.float64)
    coefficient_a = np.empty_like(L_inf)
    coefficient_b = np.empty_like(L_inf)
    d_floor_dc = np.empty_like(L_inf)
    coefficient_records: dict[str, dict[str, float]] = {}
    row_groups: list[np.ndarray] = []
    for language_index, language in enumerate(languages):
        indices = np.flatnonzero(identifiers == language)
        row_groups.append(indices)
        base = outer_size + 3 * language_index
        c_value, a_value, b_value = values[base:base + 3]
        sigmoid = float(expit(c_value))
        lower = float(problem.physical_floors[indices[0]])
        upper = float(np.min(problem.y[indices]))
        interval = upper - lower
        floor = lower + interval * sigmoid
        with np.errstate(over='ignore', invalid='ignore'):
            a_coefficient = float(np.exp(np.float64(a_value)))
            b_coefficient = float(np.exp(np.float64(b_value)))
        if not (math.isfinite(a_coefficient) and math.isfinite(b_coefficient)):
            return rejected()
        L_inf[indices] = floor
        coefficient_a[indices] = a_coefficient
        coefficient_b[indices] = b_coefficient
        d_floor_dc[indices] = interval * sigmoid * (1.0 - sigmoid)
        coefficient_records[language] = {'L_inf': floor, 'A': a_coefficient, 'B': b_coefficient, 'floor_fraction': sigmoid}
    with np.errstate(over='ignore', invalid='ignore'):
        prediction = np.asarray(L_inf + coefficient_a * capacity + coefficient_b * data, dtype=np.float64)
    if not np.isfinite(prediction).all():
        return rejected()
    residual = prediction - problem.y
    losses, score = loss_and_score(residual, delta)
    weighted_score = problem.weights * score
    data_fit_objective = float(np.dot(problem.weights, losses))
    flat_capacity_jac = capacity_jac.reshape(problem.y.size, outer_size)
    flat_data_jac = data_jac.reshape(problem.y.size, outer_size)
    gradient = np.empty_like(values)
    gradient[:outer_size] = flat_capacity_jac.T @ (weighted_score * coefficient_a) + flat_data_jac.T @ (weighted_score * coefficient_b)
    for language_index, indices in enumerate(row_groups):
        base = outer_size + 3 * language_index
        gradient[base] = float(np.sum(weighted_score[indices] * d_floor_dc[indices]))
        gradient[base + 1] = float(np.sum(weighted_score[indices] * coefficient_a[indices] * capacity[indices]))
        gradient[base + 2] = float(np.sum(weighted_score[indices] * coefficient_b[indices] * data[indices]))
    if model in EXPLORATORY_COMPOSITION_MODELS:
        regularization_objective, regularization_gradient = regularization_objective_and_gradient(
            model, outer, problem.regularization)
        objective = float(data_fit_objective + regularization_objective)
        gradient[:outer_size] += regularization_gradient
    else:
        if problem.regularization is not None:
            raise ValueError(f'{model} does not accept M5/M6 regularization')
        objective = data_fit_objective
    if not (math.isfinite(objective) and np.isfinite(gradient).all()):
        return rejected()
    return (objective, gradient, prediction, coefficient_records)

def _run_direct_once(projected: ProjectedStartResult, problem: ProjectedProblem, *, delta: float | None, bounds: Sequence[OuterBound], maximum_iterations: int, initial: np.ndarray | None=None) -> tuple[DirectFitResult, np.ndarray]:
    from scipy.optimize import minimize
    model = projected.model
    default_initial, languages = _direct_initial_vector(projected)
    start = default_initial if initial is None else np.asarray(initial, dtype=np.float64)
    outer_size = len(outer_parameter_names(model))
    coefficient_log_bounds = (math.log(float(np.nextafter(np.float64(0.0), np.float64(np.inf)))), math.log(float(np.finfo(np.float64).max)))
    language_bounds = [bound for _ in languages for bound in ((None, None), coefficient_log_bounds, coefficient_log_bounds)]
    scipy_bounds: list[tuple[float | None, float | None]] = [(bound.lower, bound.upper) for bound in bounds] + language_bounds

    def value_and_gradient(values: np.ndarray) -> tuple[float, np.ndarray]:
        objective, gradient, _, _ = _direct_objective_gradient(values, model=model, problem=problem, languages=languages, delta=delta)
        return (objective, gradient)
    result = minimize(value_and_gradient, start, method='L-BFGS-B', jac=True, bounds=scipy_bounds, options={'ftol': OUTER_FTOL, 'gtol': OUTER_GTOL, 'maxiter': int(maximum_iterations), 'maxls': 50})
    final_values = np.asarray(result.x, dtype=np.float64)
    objective, _, prediction, coefficients = _direct_objective_gradient(final_values,
        model=model, problem=problem, languages=languages, delta=delta)
    outer = np.asarray(result.x[:outer_size], dtype=np.float64)
    components = None
    if model in EXPLORATORY_COMPOSITION_MODELS and math.isfinite(objective):
        regularization_objective, _ = regularization_objective_and_gradient(
            model, outer, problem.regularization)
        data_fit_objective = weighted_objective(
            prediction - problem.y, problem.weights, delta)
        components = ObjectiveComponents(data_fit=data_fit_objective,
            regularization=regularization_objective, total=objective)
    direct = DirectFitResult(success=bool(result.success), status=int(result.status), message=str(result.message), objective=objective, prediction=prediction, outer_coordinates=outer, shared_parameters=shared_from_outer(model, outer), language_coefficients=coefficients, iterations=int(result.nit), function_evaluations=int(result.nfev), objective_components=components)
    return (direct, np.asarray(result.x, dtype=np.float64))

def direct_joint_check(projected: ProjectedStartResult, problem: ProjectedProblem, *, delta: float | None, bounds: Sequence[OuterBound] | None=None, maximum_iterations: int=OUTER_MAX_ITERATIONS) -> DirectCheckResult:
    """Run the D10 direct log/sigmoid joint cross-check, retrying once."""
    if projected.objective is None or projected.prediction is None:
        raise ProjectedOptimizationError('projected result has no finite solution')
    active_bounds = tuple(initial_bounds(projected.model) if bounds is None else bounds)
    direct, vector = _run_direct_once(projected, problem, delta=delta, bounds=active_bounds, maximum_iterations=maximum_iterations)
    comparison = equivalence_check(projected.objective, direct.objective, projected.prediction, direct.prediction)
    retried = False
    if not direct.success or not comparison.equivalent:
        retried = True
        direct, _ = _run_direct_once(projected, problem, delta=delta, bounds=active_bounds, maximum_iterations=2 * maximum_iterations, initial=vector)
        comparison = equivalence_check(projected.objective, direct.objective, projected.prediction, direct.prediction)
    return DirectCheckResult(direct=direct, equivalence=comparison, retried=retried, resolved=bool(direct.success and comparison.equivalent))
