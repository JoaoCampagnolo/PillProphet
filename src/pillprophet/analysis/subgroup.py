"""Subgroup analysis by phase, indication, sponsor, etc."""


def subgroup_metrics(y_true, y_prob, groups, group_column: str) -> dict:
    """Compute metrics stratified by a grouping variable."""
    raise NotImplementedError
