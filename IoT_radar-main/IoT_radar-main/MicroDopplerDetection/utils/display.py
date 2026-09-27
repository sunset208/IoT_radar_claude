"""Real-time matplotlib dashboard for the micro-Doppler radar pipeline."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Generator

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)


class DashboardRadar:
    """Matplotlib dashboard for micro-Doppler monitoring.

    Layout with GridSpec — default ``show_presence_score=True`` (temps réel)::

        Row 0 : TX spectrum (full width)
        Row 1 : RX spectrum (full width)
        Row 2 : Presence score curve (left 3/5) | Info box (right 2/5)

    With ``show_presence_score=False`` (relecture ``record_visualization``),
    row 2 is only the info box spanning the full width — no score curve.

    When the score panel is shown, the presence score is the fused output of
    the Fisher band-power F-test and the phase-autocorrelation peak.  The
    dashboard status is driven by the fused score vs. ``affichage.seuil_proba``.

    Architecture
    ------------
    The frame generator is consumed in a daemon producer thread.  This
    decouples the (sometimes long) acquisition + processing path from the
    GUI event loop, which is essential on real hardware where
    ``sdr.rx()`` blocks ~8 ms per buffer at 2 MHz / 16384 samples.

    Two specific WSLg + TkAgg pitfalls are explicitly avoided here:

    * **Grey / unopenable window.**  The producer thread is *not* started
      until the dashboard fires its first ``draw_event``.  This
      guarantees Tk has fully painted the window before any GIL pressure
      from the producer kicks in.
    * **Generator-already-executing race.**  ``generator.close()`` is
      called *only* from inside the producer thread's ``finally`` block,
      never from the GUI thread.  The GUI just sets a stop flag and
      joins.
    """

    def __init__(
        self,
        config: dict,
        context: dict,
        *,
        show_presence_score: bool = True,
    ) -> None:
        aff = config["affichage"]
        det_cfg = config.get("detection", {})

        self._N_hist: int = aff["N_historique"]
        self._plein_ecran: bool = aff["plein_ecran"]
        self._seuil_score: float = aff.get("seuil_proba", 0.6)
        self._alpha: float = det_cfg.get("alpha", 0.01)

        self._f_hz: np.ndarray = context["f_hz"]
        self._f_hz_tx: np.ndarray = context["f_hz_tx"]
        self._spectre_tx_db: np.ndarray = context["spectre_tx_db"]
        self._R_min_m: float = context.get("R_min_m", 0.0)
        self._R_max_m: float = context.get("R_max_m", 0.0)
        self._df_hz: float = context.get("df_hz", 0.0)
        self._dv_mps: float = context.get("dv_mps", 0.0)
        self._n_fft: int = context.get("n_fft", 0)
        self._clutter_mode: str = context.get("clutter_mode", "?")
        self._bande_resp: list = context.get("bande_resp", [0.1, 1.0])
        self._B_eff_hz: float = context.get("B_eff_hz", 0.0)

        self._show_presence_score: bool = show_presence_score

        self._score_history: list[float] = []
        self._frame_count: int = 0
        self._last_score: float = 0.0
        self._last_p_value_f: float = 1.0
        self._last_acf_peak: float = 0.0
        self._last_fv_estimated: float | None = None
        self._rx_limits_initialised: bool = False

        self._fig = plt.figure(figsize=(14, 9), constrained_layout=True)
        gs = GridSpec(3, 5, figure=self._fig)

        self._ax_tx = self._fig.add_subplot(gs[0, :])
        self._ax_rx = self._fig.add_subplot(gs[1, :])
        if self._show_presence_score:
            self._ax_score = self._fig.add_subplot(gs[2, :3])
            self._ax_info = self._fig.add_subplot(gs[2, 3:])
        else:
            self._ax_score = None
            self._ax_info = self._fig.add_subplot(gs[2, :])

        self._fig.suptitle(
            "Radar Micro-Doppler — Détection de survivants",
            fontsize=13,
            fontweight="bold",
        )

        self._init_panels()

        if self._plein_ecran:
            mng = plt.get_current_fig_manager()
            if mng is not None:
                try:
                    mng.full_screen_toggle()
                except Exception:
                    pass

        logger.info(
            "Dashboard initialisé — backend=%s, score=%s",
            matplotlib.get_backend(),
            "oui" if self._show_presence_score else "non",
        )

    # ------------------------------------------------------------------
    # Panel initialisation
    # ------------------------------------------------------------------

    def _init_panels(self) -> None:
        ax_tx = self._ax_tx
        ax_rx = self._ax_rx
        ax_info = self._ax_info

        # Panel 1 — TX spectrum (static)
        ax_tx.set_title("Signal émis (domaine fréquentiel)")
        ax_tx.set_xlabel("Fréquence (Hz)")
        ax_tx.set_ylabel("Puissance (dB)")
        ax_tx.plot(
            self._f_hz_tx,
            self._spectre_tx_db,
            linewidth=0.8,
            color="tab:blue",
        )
        _auto_ylim(ax_tx, self._spectre_tx_db)

        # Panel 2 — RX spectrum (live)
        ax_rx.set_title("Signal reçu (domaine fréquentiel)")
        ax_rx.set_xlabel("Fréquence Doppler (Hz)")
        ax_rx.set_ylabel("Puissance (dB)")
        (self._line_rx,) = ax_rx.plot(
            self._f_hz,
            np.zeros(len(self._f_hz)),
            linewidth=0.8,
            color="tab:orange",
        )
        ax_rx.set_xlim(self._f_hz[0], self._f_hz[-1])
        ax_rx.set_ylim(-120, 0)
        # Status text drawn on the RX axes (blit-friendly, unlike a
        # figure-level suptitle which cannot be blitted).
        self._status_text = ax_rx.text(
            0.99, 0.98, "En attente de la première trame…",
            transform=ax_rx.transAxes,
            ha="right", va="top",
            fontsize=11, fontweight="bold", color="grey",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="grey", alpha=0.8),
        )

        if self._show_presence_score and self._ax_score is not None:
            ax_score = self._ax_score
            ax_score.set_title("Score de présence")
            ax_score.set_xlabel("Trame")
            ax_score.set_ylabel("Score (Fisher × ACF)")
            ax_score.set_ylim(-0.05, 1.05)
            ax_score.set_xlim(0, max(self._N_hist, 1))
            (self._line_score,) = ax_score.plot(
                [], [], linewidth=1.2, color="tab:purple",
            )
            ax_score.axhline(
                self._seuil_score,
                color="grey",
                linestyle="--",
                linewidth=1.0,
                label=f"Seuil score ({self._seuil_score})",
            )
            ax_score.legend(loc="upper left", fontsize=8)
        else:
            self._line_score = None

        # Panel 3b — Info box (static structure, updated text)
        ax_info.set_axis_off()
        self._info_text = ax_info.text(
            0.05, 0.95, "",
            transform=ax_info.transAxes,
            fontsize=9,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#f0f0f0",
                edgecolor="#888888",
                linewidth=1.2,
            ),
        )
        self._update_info_box()

    # ------------------------------------------------------------------
    # Info box
    # ------------------------------------------------------------------

    def _update_info_box(self) -> None:
        if self._show_presence_score:
            if self._last_fv_estimated is None:
                fv_str = "  fv     = —"
            else:
                fv_str = f"  fv     = {self._last_fv_estimated:.3f} Hz"

            lines = [
                "╔══════════════════════╗",
                "║   PARAMÈTRES RADAR   ║",
                "╚══════════════════════╝",
                "",
                f"  δf     = {self._df_hz:.3f} Hz",
                f"  δv     = {self._dv_mps * 100:.2f} cm/s",
                f"  N_FFT  = {self._n_fft}",
                f"  B_eff  = {self._B_eff_hz:.1f} Hz",
                f"  Clutter: {self._clutter_mode}",
                f"  Bande  : ±{self._bande_resp[0]}–{self._bande_resp[1]} Hz",
                "",
                f"  Portée : {self._R_min_m:.1f}–{self._R_max_m:.1f} m",
                "",
                "─────── LIVE ───────",
                f"  p_F    = {self._last_p_value_f:.2e}  (α={self._alpha:.0e})",
                f"  ACF    = {self._last_acf_peak:+.2f}",
                fv_str,
                f"  Score  = {self._last_score:.2f}",
            ]
        else:
            lines = [
                "╔══════════════════════╗",
                "║   PARAMÈTRES RADAR   ║",
                "╚══════════════════════╝",
                "",
                f"  δf     = {self._df_hz:.3f} Hz",
                f"  δv     = {self._dv_mps * 100:.2f} cm/s",
                f"  N_FFT  = {self._n_fft}",
                f"  B_eff  = {self._B_eff_hz:.1f} Hz",
                f"  Clutter: {self._clutter_mode}",
                f"  Bande  : ±{self._bande_resp[0]}–{self._bande_resp[1]} Hz",
                "",
                f"  Portée : {self._R_min_m:.1f}–{self._R_max_m:.1f} m",
            ]
        self._info_text.set_text("\n".join(lines))

    # ------------------------------------------------------------------
    # Frame update (called by FuncAnimation in the GUI thread)
    # ------------------------------------------------------------------

    def update_frame(self, frame_data: dict[str, Any] | None) -> tuple:
        """Refresh the live panels with a new frame, or no-op if ``None``.

        Returns the tuple of artists that may have changed.  This is
        what ``FuncAnimation`` redraws when ``blit=True``.
        """
        if not self._show_presence_score:
            artists_bs = (
                self._line_rx,
                self._info_text,
                self._status_text,
            )
        else:
            assert self._line_score is not None
            artists_bs = (
                self._line_rx,
                self._line_score,
                self._info_text,
                self._status_text,
            )

        if frame_data is None:
            return artists_bs

        self._frame_count += 1

        col_db = frame_data["spectre_colonne"]
        n_trame = frame_data.get("n_trame", self._frame_count)

        # Panel 2 — RX spectrum
        self._line_rx.set_ydata(col_db)
        if not self._rx_limits_initialised:
            _auto_ylim(self._ax_rx, col_db)
            self._rx_limits_initialised = True

        if not self._show_presence_score:
            self._status_text.set_text(f"Trame {n_trame}")
            self._status_text.set_color("dimgray")
            return artists_bs

        assert self._line_score is not None
        score = frame_data["score_presence"]
        p_value_f = frame_data["p_value_f"]
        acf_peak = frame_data["acf_peak"]
        fv_estimated = frame_data["fv_estimated"]
        score_detected = score >= self._seuil_score

        # Panel 3a — Presence score history
        self._score_history.append(score)
        if len(self._score_history) > self._N_hist:
            self._score_history = self._score_history[-self._N_hist:]
        x_score = np.arange(len(self._score_history))
        self._line_score.set_data(x_score, self._score_history)

        # Panel 3b — Info box live values
        self._last_score = score
        self._last_p_value_f = p_value_f
        self._last_acf_peak = acf_peak
        self._last_fv_estimated = fv_estimated
        self._update_info_box()

        # Status feedback follows the same fused-score threshold drawn on the
        # score panel.  The raw Fisher alert remains available in frame_data for
        # diagnostics, but it should not override the user-visible score.
        if score_detected:
            self._status_text.set_text(f"RESPIRATION DÉTECTÉE — Trame {n_trame}")
            self._status_text.set_color("green")
        else:
            self._status_text.set_text(f"Aucune détection — Trame {n_trame}")
            self._status_text.set_color("red")

        return artists_bs

    # ------------------------------------------------------------------
    # Animation loop
    # ------------------------------------------------------------------

    def run(self, generator: Generator[dict[str, Any], None, None]) -> None:
        """Start the live animation driven by a frame generator.

        Lifecycle:

        1. Build the figure (already done in ``__init__``).
        2. Schedule ``FuncAnimation`` ticks (drives ``update_frame``).
        3. The first ``draw_event`` (i.e. when the OS window is actually
           painted) starts a daemon producer thread.  The thread iterates
           the generator and stores the latest frame in ``_latest_frame``.
        4. Every animation tick, the GUI thread reads ``_latest_frame``
           (newest frame, dropping any older ones) and forwards it to
           ``update_frame``.
        5. On window close, ``_stop_event`` is set and the producer
           thread finishes (which calls ``generator.close()`` → SDR
           cleanup).
        """
        logger.info("Lancement du dashboard temps réel")

        self._generator = generator
        self._stop_event = threading.Event()
        self._latest_frame: dict[str, Any] | None = None
        self._producer_thread: threading.Thread | None = None
        self._producer_started = False

        def _producer() -> None:
            try:
                for frame in generator:
                    if self._stop_event.is_set():
                        break
                    self._latest_frame = frame
                    # Yield the GIL so Tk's event loop on the main thread
                    # can keep processing redraws and user input.  Most
                    # NumPy / SciPy primitives already release the GIL,
                    # but the surrounding Python loop does not, and on
                    # WSLg + TkAgg even small bursts of GIL pressure can
                    # delay the very first window paint.
                    time.sleep(0)
            except Exception:
                logger.exception("Erreur dans le thread de production")
            finally:
                try:
                    generator.close()
                except Exception:
                    pass
                logger.info("Producteur de trames arrêté")

        def _start_producer_once(_event=None) -> None:
            if self._producer_started:
                return
            self._producer_started = True
            self._producer_thread = threading.Thread(
                target=_producer, name="radar-frame-producer", daemon=True,
            )
            self._producer_thread.start()
            logger.info("Producteur de trames démarré (fenêtre visible)")

        def _on_close(_event) -> None:
            self._stop_event.set()

        # Pull-from-shared-variable frame source for FuncAnimation.
        def _frames():
            # On the very first call, ensure the producer is running.
            # (``draw_event`` is the cleanest signal but is backend-
            # dependent; this is a defensive fallback.)
            if not self._producer_started:
                _start_producer_once()
            while not self._stop_event.is_set():
                # Atomic read — last writer wins (single producer).
                frame = self._latest_frame
                self._latest_frame = None  # don't redraw the same frame twice
                yield frame

        # The producer is started by the first draw event, which fires
        # *after* Tk has painted the window for the first time.  This
        # avoids the WSLg "grey window" issue.
        self._fig.canvas.mpl_connect("draw_event", _start_producer_once)
        self._fig.canvas.mpl_connect("close_event", _on_close)

        # ``interval=30`` gives the GUI a refresh ceiling of ~33 Hz.
        # ``blit=True`` makes redraws cheap by only re-blitting the
        # artists returned by :meth:`update_frame`.
        self._anim = FuncAnimation(
            self._fig,
            self.update_frame,
            frames=_frames(),
            interval=30,
            blit=True,
            cache_frame_data=False,
            repeat=False,
        )

        try:
            plt.show()
        finally:
            self._stop_event.set()
            if self._producer_thread is not None and self._producer_thread.is_alive():
                self._producer_thread.join(timeout=2.0)


# ----------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------

def _auto_ylim(
    ax: matplotlib.axes.Axes,
    data: np.ndarray,
    margin_ratio: float = 0.1,
) -> None:
    """Set y-axis limits with a small margin around the data range."""
    if len(data) == 0:
        return
    lo, hi = float(np.min(data)), float(np.max(data))
    span = hi - lo if hi > lo else 1.0
    margin = span * margin_ratio
    ax.set_ylim(lo - margin, hi + margin)
