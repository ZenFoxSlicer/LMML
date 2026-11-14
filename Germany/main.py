# GERMANY: investigate a locked box with five knobs and one screen.
# We can turn the knobs and read the number, but we cannot open the box.
# Some knobs may do nothing. Two may work together. Some settings turn on
# an extra amount, like a switch. We must discover which knobs and numbers.
#
# The trap: testing a brightness dial while a lamp is off tells you very little.
# Likewise, leaving the other knobs in the middle can hide a shared effect.
# So our program tries combinations before it decides what each knob does.
#
# READING ORDER: start at solve() below. It contains the story in seven steps.
# When it calls a helper, come back to that helper to see how the step works.
# Only ask() requests a reading. Everything else is reasoning with saved answers.
#
# Names used in this file:
# - point: one list of five knob settings, each between 0 and 1.
# - prediction: the number we think the screen will show.
# - coefficient / weight: a multiplier, like the 3 in "3 * knob_setting".
# - interaction: two knobs working together.
# - residual: actual reading minus predicted reading; the leftover error.
# - regime / label: our name for a switch state, off (0) or on (1).
# - boundary / threshold: where the switch changes state / the limit in its rule.
#
# NumPy works with whole tables at once. @ multiplies matching entries and adds
# them: [2, 3] @ [4, 5] = 2*4 + 3*5 = 23. .T swaps rows and columns.
# The test readings are limited to 150; our plan uses at most 148.

"""Work out how the locked box responds to its five knobs.

The platform calls solve(oracle, points). The oracle lets us read the screen.
We return our explanation of the box and our answers to the examiner's points.
"""

from itertools import combinations, product
import numpy as np


# HELPER: prepare the ingredients for one possible explanation.
# A point is a list of five knob settings, like [0, 1, 0.2, 0.8, 0.5].
# pair says which two knobs we currently suspect work together.
def _design(points, pair):
    # Put the settings in a table: one experiment per row, five knobs per row.
    # reshape(-1, 5) asks NumPy to work out the number of rows.
    x = np.asarray(points, dtype=float).reshape(-1, 5)
    # Describe each knob relative to its middle position.
    # Setting 0 becomes -0.5; setting 0.5 becomes 0; setting 1 becomes +0.5.
    # If either suspected knob is in the middle, their product below becomes zero.
    # That is how a middle-only experiment can miss their shared effect.
    z = x - 0.5
    # Make seven columns: one constant, five knob values, and one paired product.
    # Later, NumPy will find a multiplier for each column. For example, a
    # multiplier of 3 on a knob column means "add 3 times this knob's value".
    return np.column_stack((np.ones(len(x)), z, z[:, pair[0]] * z[:, pair[1]]))


# HELPER: see whether an explanation has left out just a switch.
# residual = actual screen reading - our current prediction.
# Example: leftover errors [0, 0, 0, 2, 2] suggest an extra +2 in some cases.
# Each row here belongs to a different possible explanation of the box.
def _two_levels(residuals):
    """Look for two groups of leftover errors: switch off and switch on."""
    # Sort the errors so we can try dividing them into a lower and an upper group.
    ordered = np.sort(residuals, axis=1)
    # Keep running totals and totals of squares. These let us compare every
    # possible dividing place quickly, without writing a loop for each group.
    sums = np.cumsum(ordered, axis=1)
    squares = np.cumsum(ordered * ordered, axis=1)
    count = residuals.shape[1]
    # Try a lower group of size 1, then size 2, and so on.
    # Neither group is allowed to be empty.
    k = np.arange(1, count, dtype=float)
    # Measure how spread out each group is around its own average.
    # Squaring the differences makes both positive and negative errors count.
    # The formulas below calculate that total for every possible division.
    left = squares[:, :-1] - sums[:, :-1] ** 2 / k
    right = ((squares[:, -1:] - squares[:, :-1])
             - (sums[:, -1:] - sums[:, :-1]) ** 2 / (count - k))
    # Add the two groups' errors. Small means two clear groups.
    # maximum(..., 0) removes tiny negative values caused by computer rounding.
    error = np.maximum(left + right, 0)
    # Choose the best division for each explanation.
    # cut is the value halfway between the two groups at that division.
    split = np.argmin(error, axis=1)
    row = np.arange(len(residuals))
    cut = (ordered[row, split] + ordered[row, split + 1]) * 0.5
    return error[row, split], cut


# HELPER: try explanations of what is inside the locked box.
# We know the task's formula family, but must discover its knobs and numbers.
# points are experiments we have done; values are the readings we received.
def _identify(points, values, rng):
    """Try possible explanations using the readings we already have.

We look for the pair of knobs that works together and the extra amount added
by the switch. Trying an explanation is just arithmetic, not another reading.
"""
    x = np.asarray(points, dtype=float)
    y = np.asarray(values, dtype=float)
    # Take 1,024 small groups of seven readings from the 32 interior experiments.
    # Why seven? Our possible explanation has seven unknown multipliers.
    # Why many groups? Some groups should all have the switch in the same state,
    # which lets us learn the ordinary knob effects without crossing the jump.
    # We are reusing saved readings here. This costs ZERO extra queries.
    indices = np.array([rng.choice(np.arange(32, len(x)), 7, replace=False)
                        for _ in range(1024)])
    best = None
    # There are ten possible pairs among five knobs. Try each pair in turn.
    for pair in combinations(range(5), 2):
        # Build the seven ingredients for this suspected pair, then select the
        # small groups of experiments that we picked above.
        design = _design(x, pair)
        minimal = design[indices]
        # Ask NumPy to work out the multipliers for every small group.
        # These are guesses based on seven readings, not our final answer.
        # pinv and einsum do this arithmetic for all the groups at once.
        weights = np.einsum('bij,bj->bi', np.linalg.pinv(minimal, rcond=1e-10),
                            y[indices])
        # Test every guess against ALL our readings, not just its seven examples.
        # The @ operation multiplies ingredients by their multipliers and adds them.
        # The correct explanation should leave two clear groups of errors.
        residuals = y[None, :] - weights @ design.T
        errors, cuts = _two_levels(residuals)
        # Keep the four most promising guesses for this pair and improve them.
        # labels is our current guess about the switch: 0 = off, 1 = on.
        # This is based on the leftover error, not the size of the raw screen reading.
        for index in np.argsort(errors)[:4]:
            labels = (residuals[index] > cuts[index]).astype(float)
            for _ in range(4):
                # Add an eighth column for the switch state. Its multiplier is the jump.
                # lstsq chooses multipliers that make the predictions as close as possible
                # to the saved readings. This is what "fitting a model" means here.
                matrix = np.column_stack((design, labels))
                coefficients = np.linalg.lstsq(matrix, y, rcond=None)[0]
                # If we named the two groups backwards, swap them.
                # We describe the switch as adding a positive extra amount.
                if coefficients[-1] < 0:
                    labels = 1 - labels
                    continue
                # Stop this refinement if its supposed jump is effectively zero.
                if coefficients[-1] < 1e-9:
                    break
                # Use the updated explanation to reconsider which readings had the switch on.
                # More than half the estimated jump counts as on.
                new_labels = ((y - design @ coefficients[:-1])
                              > coefficients[-1] * 0.5).astype(float)
                # If no switch guesses changed, there is no need to repeat this improvement.
                if np.array_equal(new_labels, labels):
                    break
                labels = new_labels
            # Work out the multipliers once more using the final switch guesses.
            matrix = np.column_stack((design, labels))
            coefficients = np.linalg.lstsq(matrix, y, rcond=None)[0]
            if coefficients[-1] < 0:
                labels = 1 - labels
                matrix = np.column_stack((design, labels))
                coefficients = np.linalg.lstsq(matrix, y, rcond=None)[0]
            # Score this explanation by its average squared prediction error.
            # Keep the best pair, its multipliers, and its switch guesses.
            error = float(np.mean((matrix @ coefficients - y) ** 2))
            if best is None or error < best[0]:
                best = error, pair, coefficients, labels
    return best


# HELPER: find places where we already know the switch changes state.
# A "corner" is a setting with every knob at either 0 or 1.
# An "edge" joins two corners that differ in just one knob.
def _crossing_edges(corners, labels):
    edges = []
    for a in range(len(corners)):
        for b in range(a + 1, len(corners)):
            # Skip endpoints that appear to have the same switch state.
            if labels[a] == labels[b]:
                continue
            # Keep only paths where we can move a single knob between the endpoints.
            if np.count_nonzero(corners[a] != corners[b]) != 1:
                continue
            # Name the endpoints by switch state: low = off, high = on.
            # These names do not mean the actual knob setting is smaller or larger.
            low, high = (a, b) if labels[a] == 0 else (b, a)
            edges.append((corners[low].copy(), corners[high].copy()))
    return edges


# HELPER: choose up to six paths to investigate closely.
# We want experiments spread across different settings, not six repeats
# of almost the same experiment.
def _spread_edges(edges, count=6):
    """Choose different places to investigate where the switch turns on."""
    # Use each path's midpoint as a rough description of where it is.
    mids = np.array([(a + b) / 2 for a, b in edges])
    # Start with a path near the middle of the available paths.
    selected = [int(np.argmin(np.linalg.norm(mids - mids.mean(axis=0), axis=1)))]
    while len(selected) < min(count, len(edges)):
        # Describe the other paths by how far they are from our first choice.
        relative = mids - mids[selected[0]]
        if len(selected) > 1:
            # SVD is a NumPy tool for finding directions in a collection of points.
            # Here it tells us what directions our chosen experiments already cover.
            _, singular, vectors = np.linalg.svd(relative[selected[1:]], full_matrices=False)
            basis = vectors[singular > 1e-8]
            # Remove the parts already covered. What remains tells us how much
            # new information a different path could add.
            remainder = relative - (relative @ basis.T) @ basis
        else:
            remainder = relative
        # Prefer the path that adds the most new information.
        scores = np.linalg.norm(remainder, axis=1)
        # Once all directions are covered, simply choose a path far from the others.
        if scores.max() < 1e-8:
            scores = np.min(np.linalg.norm(mids[:, None] - mids[selected], axis=2), axis=1)
        # Do not choose the same path twice.
        scores[selected] = -1
        selected.append(int(np.argmax(scores)))
    return [edges[index] for index in selected]


# START READING HERE: this is our automatic investigator.
# The platform gives us the oracle, which can read the box's screen.
# It also gives us points: the 300 exam questions we must answer later.
# The helper functions above are tools used by the steps below.
def solve(oracle, points):
    # Use repeatable random choices, so rerunning the same experiment is easier to debug.
    rng = np.random.default_rng(831709)
    # observations stores knob settings; values stores the matching screen readings.
    # cache remembers answers by setting, so we do not need to ask the box twice.
    observations, values, cache = [], [], {}

    # OUR WAY TO READ THE SCREEN. Every new reading uses one of our 150 chances.
    # phase is just a note explaining why we asked; it is saved in the query log.
    def ask(point, phase):
        # Keep every knob in its allowed range. Use a rounded tuple as the cache key.
        x = np.clip(np.asarray(point, dtype=float), 0, 1)
        key = tuple(round(float(v), 9) for v in x)
        # Only send a request if we have not already saved this answer.
        if key not in cache:
            # Check before asking. A 151st reading would invalidate the whole submission.
            if oracle.remaining <= 0:
                raise RuntimeError('Query budget exhausted before an intended query.')
            # This is where we actually ask the box for a reading. Save what it says.
            value = float(oracle.ask(x.tolist(), phase=phase))
            cache[key] = value
            observations.append(x.copy())
            values.append(value)
        return cache[key]

    # STEP 1: test combinations, not just one knob with all the others in the middle.
    # Every knob is either 0 or 1. Five knobs give 2**5 = 32 combinations.
    # Trying them together helps expose hidden shared effects and switch states.
    corners = np.array(list(product((0.0, 1.0), repeat=5)))
    for point in corners:
        ask(point, 'full factorial: joint effects and rare regimes')
    # STEP 2: take 32 readings between the extremes as well.
    # Different explanations can agree at 0 and 1 but disagree in between.
    # We have now spent 64 queries.
    for point in rng.uniform(0.04, 0.96, size=(32, 5)):
        ask(point, 'interior: distinguish bilinear response from a step')
    # STEP 3: work out which explanation matches the saved readings.
    # This searches possible paired knobs and switch states without new queries.
    fit_error, pair, coefficients, labels = _identify(observations, values, rng)
    # smooth holds the ordinary knob effects, including the paired effect.
    # jump is the extra amount added when the switch is on.
    smooth, jump = coefficients[:-1], float(coefficients[-1])

    # Given a reading, decide whether the switch was off or on.
    # Example: if the ordinary formula predicts 10 and the jump is 2,
    # a reading near 10 means off; a reading near 12 means on.
    def regime(point, value):
        expected = float((_design([point], pair) @ smooth)[0])
        return int(value - expected > jump * 0.5)

    # STEP 4: narrow down where the switch changes state.
    # We can now move one knob at a time: we already know our two endpoints
    # have different switch states. The other knobs stay at useful settings.
    edges = _spread_edges(_crossing_edges(corners, labels[:32]))
    boundary_points = []
    # Investigate up to six of these paths.
    for low, high in edges:
        # Try the middle, then keep the half containing the change of state.
        # This is binary search, like narrowing down a guessed number.
        # At most 6 * 14 extra readings: 64 + 84 = 148, safely below 150.
        for _ in range(14):
            if oracle.remaining <= 0:
                break
            middle = (low + high) / 2
            value = ask(middle, 'boundary: adaptive binary search')
            # If the midpoint is on, move the on endpoint there. Otherwise move the off endpoint.
            # The remaining endpoints still surround the place where the switch changes.
            if regime(middle, value):
                high = middle
            else:
                low = middle
        # After 14 halvings, only 1/16384 of the original path remains.
        # Save its midpoint as our estimate of where the switch flips.
        boundary_points.append((low + high) / 2)

    # STEP 5: turn those switch locations into one rule.
    # For example, a rule could be "knob 1 + 2 * knob 4 >= some limit".
    # That is an illustration; the actual knobs and numbers come from the readings.
    boundary_points = np.asarray(boundary_points)
    center = boundary_points.mean(axis=0)
    # Use the measured switch locations to find the numbers in the rule.
    # SVD supplies the direction perpendicular to their shared flat boundary.
    _, _, vectors = np.linalg.svd(boundary_points - center, full_matrices=True)
    # normal stores the five multipliers in the switch rule.
    # A zero entry means that knob does not help decide when the switch turns on.
    normal = vectors[-1]
    direction = corners[labels[:32] == 1].mean(axis=0) - corners[labels[:32] == 0].mean(axis=0)
    # The same dividing surface has two sides. Choose the direction
    # that makes >= mean switch on, not switch off.
    if normal @ direction < 0:
        normal = -normal
    # Treat very tiny estimated influences as zero: they can be search/rounding error.
    normal[np.abs(normal) < 0.001 * np.max(np.abs(normal))] = 0
    # Put the multipliers on a common scale. Scaling the whole rule together
    # does not change it, just as x >= 1 and 2*x >= 2 mean the same thing.
    normal /= np.linalg.norm(normal)
    # threshold is the limit in "weighted knob settings >= limit".
    threshold = float(normal @ center)

    # STEP 6: improve the final numbers using every reading we collected.
    all_x = np.asarray(observations)
    all_y = np.asarray(values)
    design = _design(all_x, pair)
    # Use the actual readings to decide the switch state for these experiments.
    # Some search points are so close to the boundary that our approximate rule
    # could put them on the wrong side.
    known_regimes = (all_y - design @ smooth > jump * 0.5).astype(float)
    matrix = np.column_stack((design, known_regimes))
    # Let NumPy choose the ordinary multipliers and jump amount one last time.
    coefficients = np.linalg.lstsq(matrix, all_y, rcond=None)[0]
    smooth, jump = coefficients[:-1], float(coefficients[-1])

    # STEP 7: write our explanation for the examiner.
    # A knob matters if it changes the reading on its own, works with another knob,
    # or controls the switch. Other knobs are reported as irrelevant (decorative).
    names = ['x1', 'x2', 'x3', 'x4', 'x5']
    gate_indices = np.flatnonzero(normal)
    relevant = set(int(i) for i in gate_indices)
    relevant.update(i for i in range(5) if abs(smooth[i + 1]) > 1e-4)
    interactions = []
    # If the paired effect is large enough to count, report its knobs and multiplier.
    if abs(smooth[-1]) > 1e-4:
        relevant.update(pair)
        interactions.append({
            'variables': [names[i] for i in pair], 'type': 'product',
            'coefficient': float(smooth[-1]),
            'equivalent_centered_form': f'{smooth[-1]:.12g}*({names[pair[0]]}-0.5)*({names[pair[1]]}-0.5)',
        })
    # Write the learned formula as a readable Python expression.
    # This is a description of what we discovered, not a formula copied from practice.
    expression = f'{smooth[0]:.12g}'
    for i in range(5):
        if abs(smooth[i + 1]) > 1e-9:
            expression += f' + ({smooth[i + 1]:.12g})*({names[i]}-0.5)'
    expression += f' + ({smooth[-1]:.12g})*({names[pair[0]]}-0.5)*({names[pair[1]]}-0.5)'
    plane = ' + '.join(f'({normal[i]:.12g})*{names[i]}' for i in gate_indices)
    condition = f'{plane} >= {threshold:.12g}'
    expression += f' + ({jump:.12g} if {condition} else 0.0)'
    # Answer the 300 exam questions with our learned formula.
    # We calculate these predictions ourselves; we do NOT ask the box 300 more times.
    challenge = np.asarray(points, dtype=float).reshape(-1, 5)
    predictions = (_design(challenge, pair) @ smooth
                   + jump * (challenge @ normal >= threshold))
    # Return the examiner's required fields as ordinary Python data.
    # The platform checks both our explanation and our predicted readings.
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
