"""Shared utility helpers for calibration nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qualibrate import QualibrationNode


def get_node_id_label(node: "QualibrationNode") -> str:
    """Return a human-readable node ID string suitable for figure suptitles.

    - If the node was loaded from an existing run, returns that run's ID.
    - Otherwise queries the data-handler to predict the next ID (last saved ID + 1).
    - Falls back to a descriptive string if the storage manager is unavailable.
    """
    if node.parameters.load_data_id is not None:
        return str(node.parameters.load_data_id)
    try:
        from qualang_tools.results.data_handler.data_folder_tools import get_latest_data_folder
        dh = node._get_storage_manager().data_handler
        latest = get_latest_data_folder(dh.root_data_folder, folder_pattern=dh.folder_pattern)
        next_id = latest["idx"] + 1 if latest is not None else 1
        return f"{next_id}"
    except Exception:
        return "new run (id assigned after save)"
