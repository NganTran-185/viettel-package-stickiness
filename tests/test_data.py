from src.data import load, get_Xy, validate, LEAKAGE_COLS


def test_load_shape():
    df = load()
    assert df.shape[1] == 51            
    assert len(df) > 50_000            


def test_no_leakage_in_features():
    X, _ = get_Xy()
    assert not set(X.columns) & set(LEAKAGE_COLS)


def test_validate_report():
    rep = validate(load())
    assert rep["missing_target"] == 0