# GERMANY: learn a hidden formula by asking the sensor for measurements.
# The published function family has straight-line effects, one product of two
# inputs, and a sudden jump when the inputs cross a flat boundary.
#
# The plan uses at most 148 of the 150 allowed queries:
# 1. Measure all 32 combinations of dials set to 0 or 1.
# 2. Measure 32 points inside the range to distinguish possible formulas.
# 3. Find which two inputs multiply together and separate the two output levels.
# 4. Narrow down the jump location on up to six lines, using 14 queries per line.
# 5. Return the discovered structure and predictions for the challenge points.
#
# Start at solve(). The functions above it are helpers.
# NumPy notes: @ multiplies arrays; .T swaps rows and columns; [-1] means last.
# An interaction means one input's effect depends on another input.
# The oracle is the supplied object that answers queries and counts their cost.

"""Adaptive identification of a linear/bilinear response with a planar step.

Submission entry point: solve(oracle, points). Uses only NumPy and the standard
library. All observations come through the supplied, budgeted oracle.
"""

from itertools import combinations, product
import numpy as np


# Build the numbers used to fit one proposed formula.
# Each row is one measured point. Its seven columns are: a constant,
# five separate dial values, and the product of the proposed pair of dials.
def _design(points, pair):
    x = np.asarray(points, dtype=float).reshape(-1, 5)
    # Center each dial around its middle: 0.5 becomes 0.
    # This makes the separate effects and their shared effect easier to describe.
    z = x - 0.5
    return np.column_stack((np.ones(len(x)), z, z[:, pair[0]] * z[:, pair[1]]))


# A residual is the measured output minus the output predicted so far.
# After removing the correct smooth formula, residuals should form two groups:
# one before the jump and one after it. Find the best dividing point.
def _two_levels(residuals):
    """Best split into two constant residual levels, independently per row."""
    # Sort each candidate's residuals so every possible split is between neighbors.
    ordered = np.sort(residuals, axis=1)
    # Running sums let us calculate group errors quickly for every possible split.
    sums = np.cumsum(ordered, axis=1)
    squares = np.cumsum(ordered * ordered, axis=1)
    count = residuals.shape[1]
    k = np.arange(1, count, dtype=float)
    # For each split, add the squared distances from each group's own average.
    # Smaller total error means two tighter, more convincing groups.
    left = squares[:, :-1] - sums[:, :-1] ** 2 / k
    right = ((squares[:, -1:] - squares[:, :-1])
             - (sums[:, -1:] - sums[:, :-1]) ** 2 / (count - k))
    error = np.maximum(left + right, 0)
    # Keep the split with the smallest error. cut is halfway between its neighbors.
    split = np.argmin(error, axis=1)
    row = np.arange(len(residuals))
    cut = (ordered[row, split] + ordered[row, split + 1]) * 0.5
    return error[row, split], cut


# Try all ten possible pairs of interacting dials.
# For each pair, test many small fits and keep the one that explains the data best.
# These calculations reuse stored answers; they do not spend more oracle queries.
def _identify(points, values, rng):
    """Find the interaction and separate smooth response from constant jumps.

Random minimal fits inside one regime recover the same polynomial (up to its
intercept). The correct pair leaves exactly two residual levels everywhere.
Corners expose rare regimes; interior points remove corner-only ambiguities.
"""
    x = np.asarray(points, dtype=float)
    y = np.asarray(values, dtype=float)
    # Use interior points for well-conditioned minimal fits. Draw without
    # replacement within each sample, and reuse the design across hypotheses.
    # The first 32 measurements are corners. Use the later, interior measurements here.
    # A proposed smooth formula has seven coefficients, so each small fit uses seven points.
    # Try 1,024 sets: some should fall entirely on one side of the jump.
    indices = np.array([rng.choice(np.arange(32, len(x)), 7, replace=False)
                        for _ in range(1024)])
    best = None
    for pair in combinations(range(5), 2):
        design = _design(x, pair)
        minimal = design[indices]
        # Fit all the small sets in one NumPy operation.
        # pinv helps solve for coefficients even when a set of equations is troublesome.
        weights = np.einsum('bij,bj->bi', np.linalg.pinv(minimal, rcond=1e-10),
                            y[indices])
        # Use each small fit to predict every measured point, then calculate its errors.
        residuals = y[None, :] - weights @ design.T
        errors, cuts = _two_levels(residuals)
        # Refine the four most promising fits for this pair.
        # A label of 0 means the lower residual group; 1 means the upper group.
        for index in np.argsort(errors)[:4]:
            labels = (residuals[index] > cuts[index]).astype(float)
            # Alternate between fitting coefficients and assigning the two groups.
            # The extra labels column lets the formula learn a separate jump amount.
            for _ in range(4):
                matrix = np.column_stack((design, labels))
                coefficients = np.linalg.lstsq(matrix, y, rcond=None)[0]
                # Swap the group names if needed so the reported jump points upward.
                if coefficients[-1] < 0:
                    labels = 1 - labels
                    continue
                if coefficients[-1] < 1e-9:
                    break
                new_labels = ((y - design @ coefficients[:-1])
                              > coefficients[-1] * 0.5).astype(float)
                # Stop refining once no measurement changes groups.
                if np.array_equal(new_labels, labels):
                    break
                labels = new_labels
            matrix = np.column_stack((design, labels))
            coefficients = np.linalg.lstsq(matrix, y, rcond=None)[0]
            if coefficients[-1] < 0:
                labels = 1 - labels
                matrix = np.column_stack((design, labels))
                coefficients = np.linalg.lstsq(matrix, y, rcond=None)[0]
            # Compare predictions with all observed answers. Keep the smallest average squared error.
            error = float(np.mean((matrix @ coefficients - y) ** 2))
            if best is None or error < best[0]:
                best = error, pair, coefficients, labels
    return best


# Find corner-to-corner lines that cross the jump.
# An edge changes exactly one dial while leaving the other four fixed.
def _crossing_edges(corners, labels):
    edges = []
    for a in range(len(corners)):
        for b in range(a + 1, len(corners)):
            if labels[a] == labels[b]:
                continue
            if np.count_nonzero(corners[a] != corners[b]) != 1:
                continue
            # low and high refer to the output group, not to which dial value is smaller.
            low, high = (a, b) if labels[a] == 0 else (b, a)
            edges.append((corners[low].copy(), corners[high].copy()))
    return edges


# Choose lines in different parts of the input space.
# We need different directions to determine the whole boundary, not just one point.
def _spread_edges(edges, count=6):
    """Select edges with a well-spread, affinely independent intersection."""
    mids = np.array([(a + b) / 2 for a, b in edges])
    selected = [int(np.argmin(np.linalg.norm(mids - mids.mean(axis=0), axis=1)))]
    while len(selected) < min(count, len(edges)):
        relative = mids - mids[selected[0]]
        if len(selected) > 1:
            # SVD separates the directions already covered by the chosen lines.
            # Use it to prefer a new line that adds information in another direction.
            _, singular, vectors = np.linalg.svd(relative[selected[1:]], full_matrices=False)
            basis = vectors[singular > 1e-8]
            remainder = relative - (relative @ basis.T) @ basis
        else:
            remainder = relative
        scores = np.linalg.norm(remainder, axis=1)
        # If all directions are already covered, choose a line far from the selected ones.
        if scores.max() < 1e-8:
            scores = np.min(np.linalg.norm(mids[:, None] - mids[selected], axis=2), axis=1)
        scores[selected] = -1
        selected.append(int(np.argmax(scores)))
    return [edges[index] for index in selected]


# This is the function the platform calls. points contains inputs to predict,
# not their answers. We must learn the formula from our own oracle queries.
def solve(oracle, points):
    # Fix the random seed so the same oracle produces a repeatable experiment.
    rng = np.random.default_rng(831709)
    observations, values, cache = [], [], {}

    # Save every answer locally. Reuse an old answer instead of asking twice.
    # phase is only a readable label for the query log.
    def ask(point, phase):
        # Keep all dial settings within the allowed 0-to-1 range.
        x = np.clip(np.asarray(point, dtype=float), 0, 1)
        key = tuple(round(float(v), 9) for v in x)
        if key not in cache:
            # Check the remaining budget BEFORE sending a new query.
            if oracle.remaining <= 0:
                raise RuntimeError('Query budget exhausted before an intended query.')
            value = float(oracle.ask(x.tolist(), phase=phase))
            cache[key] = value
            observations.append(x.copy())
            values.append(value)
        return cache[key]

    # With five dials and two choices per dial, there are 2**5 = 32 corners.
    # Changing dials together exposes effects that single-dial tests can miss.
    corners = np.array(list(product((0.0, 1.0), repeat=5)))
    for point in corners:
        ask(point, 'full factorial: joint effects and rare regimes')
    # Add 32 points away from the corners to distinguish formulas
    # that happen to produce the same values at the corners.
    for point in rng.uniform(0.04, 0.96, size=(32, 5)):
        ask(point, 'interior: distinguish bilinear response from a step')
    fit_error, pair, coefficients, labels = _identify(observations, values, rng)
    # The first seven coefficients describe the smooth response. The last is the jump.
    smooth, jump = coefficients[:-1], float(coefficients[-1])

    # Subtract the smooth prediction from an answer.
    # More than half the jump means this point belongs to the upper group.
    def regime(point, value):
        expected = float((_design([point], pair) @ smooth)[0])
        return int(value - expected > jump * 0.5)

    edges = _spread_edges(_crossing_edges(corners, labels[:32]))
    boundary_points = []
    # Both ends already have known, opposite groups. Query the middle,
    # then keep the half that still has opposite groups at its ends. Repeat.
    for low, high in edges:
        # Six lines times fourteen queries plus 64 discovery queries <= 148.
        for _ in range(14):
            if oracle.remaining <= 0:
                break
            middle = (low + high) / 2
            value = ask(middle, 'boundary: adaptive binary search')
            if regime(middle, value):
                high = middle
            else:
                low = middle
        # After 14 halvings, the remaining interval is only 1/16384 of the edge.
        # Its midpoint is a close estimate of where the jump happens.
        boundary_points.append((low + high) / 2)

    # Fit a flat boundary through the estimated crossing points.
    boundary_points = np.asarray(boundary_points)
    center = boundary_points.mean(axis=0)
    _, _, vectors = np.linalg.svd(boundary_points - center, full_matrices=True)
    # The normal is a direction perpendicular to the boundary.
    # Its five entries become the five coefficients in the boundary equation.
    normal = vectors[-1]
    direction = corners[labels[:32] == 1].mean(axis=0) - corners[labels[:32] == 0].mean(axis=0)
    # Point the normal toward the upper group, so the final condition uses >=.
    if normal @ direction < 0:
        normal = -normal
    # Bisection error is below 1/16384. Remove only coefficients at that scale.
    normal[np.abs(normal) < 0.001 * np.max(np.abs(normal))] = 0
    normal /= np.linalg.norm(normal)
    # The threshold fixes the boundary's position: normal @ inputs >= threshold.
    threshold = float(normal @ center)

    # Refit using all observed regimes, including the closely spaced boundary
    # measurements. Do not assign their sides from the approximate plane.
    all_x = np.asarray(observations)
    all_y = np.asarray(values)
    design = _design(all_x, pair)
    # Near the boundary, use the measured answers to decide groups.
    # The approximate plane could put a very close point on the wrong side.
    known_regimes = (all_y - design @ smooth > jump * 0.5).astype(float)
    matrix = np.column_stack((design, known_regimes))
    coefficients = np.linalg.lstsq(matrix, all_y, rcond=None)[0]
    smooth, jump = coefficients[:-1], float(coefficients[-1])

    # An input matters if it has its own effect, belongs to the interaction,
    # or helps decide which side of the jump we are on.
    names = ['x1', 'x2', 'x3', 'x4', 'x5']
    gate_indices = np.flatnonzero(normal)
    relevant = set(int(i) for i in gate_indices)
    relevant.update(i for i in range(5) if abs(smooth[i + 1]) > 1e-4)
    interactions = []
    if abs(smooth[-1]) > 1e-4:
        relevant.update(pair)
        interactions.append({
            'variables': [names[i] for i in pair], 'type': 'product',
            'coefficient': float(smooth[-1]),
            'equivalent_centered_form': f'{smooth[-1]:.12g}*({names[pair[0]]}-0.5)*({names[pair[1]]}-0.5)',
        })
    # Build a human-readable version of the learned formula.
    expression = f'{smooth[0]:.12g}'
    for i in range(5):
        if abs(smooth[i + 1]) > 1e-9:
            expression += f' + ({smooth[i + 1]:.12g})*({names[i]}-0.5)'
    expression += f' + ({smooth[-1]:.12g})*({names[pair[0]]}-0.5)*({names[pair[1]]}-0.5)'
    plane = ' + '.join(f'({normal[i]:.12g})*{names[i]}' for i in gate_indices)
    condition = f'{plane} >= {threshold:.12g}'
    expression += f' + ({jump:.12g} if {condition} else 0.0)'
    # Predict all supplied challenge points without spending any more queries.
    challenge = np.asarray(points, dtype=float).reshape(-1, 5)
    predictions = (_design(challenge, pair) @ smooth
                   + jump * (challenge @ normal >= threshold))
    # Return the exact dictionary fields requested by the evaluator.
    # Use ordinary lists and numbers so the harness can send them as JSON.
    return {
        'relevant_variables': [names[i] for i in sorted(relevant)],
        'irrelevant_variables': [names[i] for i in range(5) if i not in relevant],
        'interactions': interactions,
        'discontinuities': [{
            'gate_variables': [names[i] for i in gate_indices],
            'boundary': {'form': 'linear',
                         'coefficients': {names[i]: float(normal[i]) for i in range(5)},
                         'threshold': threshold, 'active_when': 'greater_equal'},
            'jump': jump, 'condition': condition,
        }],
        'model_expression': expression,
        'predictions': predictions.tolist(),
        'notes': (f'{oracle.spent} oracle queries. Full factorial corners and independent '
                  'interior observations separate smooth interaction from two response '
                  'regimes. Six crossing edges locate the switching plane by bisection. '
                  f'Discovery mean squared residual: {fit_error:.3g}.'),
    }
