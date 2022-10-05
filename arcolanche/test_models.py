from .models import *


def test_NActivationIsing():
    np.seed(0)

    # matching the correlations with total neighborhood activation
    n = 10
    params = np.random.normal(size=2)

    model = NActivationIsing(n)
    model.set_params(params)

    obs = model.calc_observables()

    assert np.linalg.norm(model.solve(obs, np.random.normal(size=2))['x'] - params) < 1e-6
