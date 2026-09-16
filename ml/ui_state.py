"""Pure state rules for the ML page; intentionally independent from Tk."""


def ml_controls_editable(*, knn_ready: bool, compute_locked: bool,
                         ml_locked: bool) -> bool:
    """Return whether the researcher may edit and start ML parameters.

    Worker ``is_alive()`` is deliberately absent: a completion callback may
    already have ended the UI operation while its worker is returning from the
    final Python frames.
    """
    return bool(knn_ready and not compute_locked and not ml_locked)
