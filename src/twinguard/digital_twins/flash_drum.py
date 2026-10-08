"""Independent nominal digital twin for the E_FLASH_RESS flash drum."""

from __future__ import annotations

from twinguard.plants.flash_drum import DynamicFlashModel, FlashParameters, FlashState


class FlashDigitalTwin(DynamicFlashModel):
    """Nominal DT with its own copied parameters and mutable simulation state."""

    def __init__(
        self,
        parameters: FlashParameters | None = None,
        initial_state: FlashState | None = None,
    ) -> None:
        super().__init__(parameters=parameters, initial_state=initial_state)

