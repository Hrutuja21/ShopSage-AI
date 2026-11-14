def test_project_package_imports():
    from src.shopsage import preprocessing
    assert callable(preprocessing.add_late_delivery_target)
