from __future__ import annotations

from typing import override

import equinox as eqx


class Labeler(eqx.Module):
    def create_labels(self, indices: list[tuple[int, ...]]) -> list[str]:
        raise NotImplementedError


class ListLabeler(Labeler):
    label_list: list[str]

    @override
    def create_labels(self, indices: list[tuple[int, ...]]) -> list[str]:
        if len(indices) != len(self.label_list):
            msg = f"Got {len(self.label_list)} labels for {len(indices)} plots"
            raise ValueError(msg)
        return self.label_list


class SimpleLabeler(Labeler):
    simple_label: str

    @override
    def create_labels(self, indices: list[tuple[int, ...]]) -> list[str]:
        if not indices:
            return []
        if len(indices) == 1:
            return [self.simple_label]
        return [f"{self.simple_label} {x}" for x in indices]


class DefaultLabeler(Labeler):
    @override
    def create_labels(self, indices: list[tuple[int, ...]]) -> list[str]:
        return [str(x) for x in indices]
