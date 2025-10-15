"""
Compatibility shim for legacy pickles that reference
`src.price_model.multicollinearity_pruner.MulticollinearityPruner`.

We alias to the current `AdvancedFeatureSelector` so previously saved
artifacts unpickle without errors.
"""

from .advanced_feature_selector import AdvancedFeatureSelector as MulticollinearityPruner


