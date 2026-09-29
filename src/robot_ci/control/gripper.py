"""Gripper control: turning a requested aperture into finger actuator commands.

The problem this module exists for
----------------------------------

The Panda's fingers are driven by *position* actuators with a finite
stiffness. Commanding a separation smaller than the object does not ask for
contact, it asks for a **position error**, and a position error multiplied by a
stiffness is a force:

    F = k * e

Nothing in that relationship refers to the object. A request to close to
0.2 mm on a 16 mm cube is a request for ~7.9 mm of position error *per finger*,
which on this model (k = 800 N/m) is over 6 N of contact force -- while the cube
weighs 0.049 N and needs about 0.025 N per pad to be held. The gripper was
applying roughly two hundred times the force required, and the object left
without the force ever being computed, checked, or bounded.

The fix is not a force controller. Since the actuator is a position actuator,
**a force limit is an engagement limit**:

    e_max = F_max / k

so bounding the force means flooring the requested aperture at
``contact_width - 2 * e_max``. The fingers remain position-actuated, the
collision model is untouched, and the bound is derived from the actuator's
measured stiffness rather than chosen.

`GripperController` is deliberately free of any simulator dependency: it takes
the stiffness, a control period and two measurements, and returns an aperture.
That keeps its behaviour testable without MuJoCo and keeps the unit conversion
in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

from robot_ci.models.robot import GripperControlMode


@dataclass(frozen=True)
class GripperControllerState:
    """A snapshot of what the controller is doing and what it last saw."""

    mode: GripperControlMode
    requested_width_m: float
    """The aperture a policy asked for."""
    commanded_width_m: float
    """The aperture actually sent to the actuators, after flooring and slewing."""
    target_width_m: float
    """Where the slew is heading."""
    measured_width_m: float
    """The aperture read back from the finger joints."""
    contact_width_m: float | None
    """Width of the object being closed on, when known."""
    contact_detected: bool
    engagement_limit_m: float
    """``F_max / k``: the largest position error the force bound permits."""
    max_commanded_engagement_m: float
    """Deepest the command ever went past the object width."""
    max_measured_engagement_m: float
    """Deepest the fingers ever actually went past the object width."""
    steps: int


class GripperController:
    """Converts a requested aperture into a bounded, slewed actuator command.

    Args:
        mode: Which strategy to apply.
        stiffness_n_per_m: Force per metre of position error, measured from the
            compiled finger actuator. This is the entire reason a force bound is
            expressible as a position bound, so it is a required argument rather
            than a default: a hard-coded stiffness would silently mean the wrong
            force on any other model.
        target_contact_force_n: Force the bounded mode may apply, in newtons,
            **per finger**.
        max_closure_speed_m_s: Slew rate for the command, in metres per second.
        control_dt_s: Control period, used with the slew rate.
        open_width_m: Fully open aperture of the mechanism.
        closed_width_m: Hard mechanical minimum aperture. Measured, not
            assumed: the fingers meet their stops here even with nothing
            between them.
        closure_ramp_enabled: Slew the bounded command at `max_closure_speed`
            instead of stepping it. Off by default because ramping (limits
            impulse) and bounding (limits steady-state force) are two different
            mechanisms; the bounded-contact experiment holds the ramp off so
            the force bound is measured on its own.
    """

    def __init__(
        self,
        *,
        mode: GripperControlMode = GripperControlMode.POSITION,
        stiffness_n_per_m: float,
        target_contact_force_n: float = 1.0,
        max_closure_speed_m_s: float = 0.05,
        control_dt_s: float = 0.05,
        open_width_m: float = 0.08,
        closed_width_m: float = 0.011,
        closure_ramp_enabled: bool = False,
    ) -> None:
        if stiffness_n_per_m <= 0.0:
            raise ValueError(f"stiffness must be positive, got {stiffness_n_per_m}")
        if target_contact_force_n <= 0.0:
            raise ValueError(f"target force must be positive, got {target_contact_force_n}")
        if max_closure_speed_m_s <= 0.0:
            raise ValueError(f"closure speed must be positive, got {max_closure_speed_m_s}")
        if control_dt_s <= 0.0:
            raise ValueError(f"control period must be positive, got {control_dt_s}")
        if closed_width_m > open_width_m:
            raise ValueError(
                f"closed aperture {closed_width_m} exceeds open aperture {open_width_m}"
            )

        self.mode = mode
        self.stiffness_n_per_m = float(stiffness_n_per_m)
        self.target_contact_force_n = float(target_contact_force_n)
        self.max_closure_speed_m_s = float(max_closure_speed_m_s)
        self.control_dt_s = float(control_dt_s)
        self.open_width_m = float(open_width_m)
        self.closed_width_m = float(closed_width_m)
        self.closure_ramp_enabled = bool(closure_ramp_enabled)

        self._commanded_width_m = self.open_width_m
        self._requested_width_m = self.open_width_m
        self._measured_width_m = self.open_width_m
        self._contact_width_m: float | None = None
        self._steps = 0
        self._max_commanded_engagement_m = 0.0
        self._max_measured_engagement_m = 0.0

    # -- derived quantities -------------------------------------------------

    @property
    def engagement_limit_m(self) -> float:
        """Largest position error the force bound permits: ``F_max / k``."""
        return self.target_contact_force_n / self.stiffness_n_per_m

    @property
    def contact_band_m(self) -> float:
        """How close counts as contact: the full bounded engagement span.

        Derived rather than chosen. The command deliberately sits
        ``2 * engagement_limit`` above the object width, so the fingers are
        expected to settle within that band once they touch; a narrower band
        would fail to register the contact the controller itself created.
        """
        return 2.0 * self.engagement_limit_m

    def _target_width(self, requested_width_m: float) -> float:
        """Return the aperture to slew toward, applying the force bound."""
        if self.mode is GripperControlMode.POSITION:
            return float(requested_width_m)
        if self._contact_width_m is None:
            # No known object: nothing to derive a remainder from, so the
            # request stands. This is the case for opening and for free-space
            # motion, neither of which can generate contact force.
            return float(requested_width_m)
        # Leave `2 * e_max` of the travel unused. Once the pads meet the object
        # the remaining position error is at most `e_max` per finger, so the
        # contact force is at most `F_max`.
        floor = self._contact_width_m - self.contact_band_m
        return max(float(requested_width_m), floor, self.closed_width_m)

    # -- public interface ---------------------------------------------------

    def reset(
        self,
        *,
        commanded_width_m: float | None = None,
        contact_width_m: float | None = None,
    ) -> None:
        """Begin a new episode.

        Args:
            commanded_width_m: Aperture to start from; defaults to fully open.
            contact_width_m: Width of the object this episode may grip, if any.
                Supplying it up front means the force bound is available from
                the first closing command rather than only after an explicit
                ``command_close``.
        """
        start = self.open_width_m if commanded_width_m is None else float(commanded_width_m)
        if contact_width_m is not None and contact_width_m <= 0.0:
            raise ValueError(f"contact width must be positive, got {contact_width_m}")
        self._commanded_width_m = start
        self._requested_width_m = start
        self._measured_width_m = start
        self._contact_width_m = None if contact_width_m is None else float(contact_width_m)
        self._steps = 0
        self._max_commanded_engagement_m = 0.0
        self._max_measured_engagement_m = 0.0

    def command_open(self) -> None:
        """Request a fully open aperture and forget any object being gripped."""
        self._contact_width_m = None
        self._requested_width_m = self.open_width_m

    def command_close(self, contact_width_m: float | None = None) -> None:
        """Request closure onto an object of ``contact_width_m`` metres.

        The width is what makes the force bound possible: it is the point at
        which the fingers stop travelling and start pressing.
        """
        if contact_width_m is not None:
            if contact_width_m <= 0.0:
                raise ValueError(f"contact width must be positive, got {contact_width_m}")
            self._contact_width_m = float(contact_width_m)
        self._requested_width_m = self.closed_width_m

    def step(
        self, *, requested_width_m: float, measured_width_m: float | None = None
    ) -> float:
        """Advance one control step and return the aperture to command.

        Args:
            requested_width_m: The aperture the policy asked for this step.
            measured_width_m: The aperture read back from the joints, if the
                caller can provide it. Used for contact detection and
                diagnostics; the force bound itself does not depend on it.

        Returns:
            The aperture to write to the actuators, in metres.
        """
        self._requested_width_m = float(requested_width_m)
        if measured_width_m is not None:
            self._measured_width_m = float(measured_width_m)

        target = self._target_width(self._requested_width_m)

        if self.mode is GripperControlMode.POSITION or not self.closure_ramp_enabled:
            # Pass-through, exactly as an unmediated write would be. POSITION
            # mode reproduces the previous behaviour exactly; BOUNDED with the
            # ramp off does the same, so that the force bound is measured on
            # its own (bounding limits steady-state force, ramping would limit
            # impulse -- two different mechanisms).
            self._commanded_width_m = target
        else:
            # Slew, so contact is approached over several steps. A step change
            # from open to closed delivers the whole force in one control
            # period, which is a separate experiment from the force bound.
            span = self.max_closure_speed_m_s * self.control_dt_s
            delta = target - self._commanded_width_m
            if abs(delta) <= span:
                self._commanded_width_m = target
            else:
                self._commanded_width_m += span if delta > 0.0 else -span

        self._steps += 1

        if self._contact_width_m is not None:
            self._max_commanded_engagement_m = max(
                self._max_commanded_engagement_m,
                self._contact_width_m - self._commanded_width_m,
            )
            self._max_measured_engagement_m = max(
                self._max_measured_engagement_m,
                self._contact_width_m - self._measured_width_m,
            )

        return self._commanded_width_m
    @property
    def contact_detected(self) -> bool:
        """Whether the fingers have closed onto the object.

        Geometric, not a tuned threshold: the fingers are inside the bounded
        contact band around the object's width, so they can only be there
        because the object stopped them.
        """
        if self._contact_width_m is None:
            return False
        return self._measured_width_m <= self._contact_width_m + self.contact_band_m

    def get_state(self) -> GripperControllerState:
        """Return a snapshot of the controller's inputs, outputs and extrema."""
        return GripperControllerState(
            mode=self.mode,
            requested_width_m=self._requested_width_m,
            commanded_width_m=self._commanded_width_m,
            target_width_m=self._target_width(self._requested_width_m),
            measured_width_m=self._measured_width_m,
            contact_width_m=self._contact_width_m,
            contact_detected=self.contact_detected,
            engagement_limit_m=self.engagement_limit_m,
            max_commanded_engagement_m=self._max_commanded_engagement_m,
            max_measured_engagement_m=self._max_measured_engagement_m,
            steps=self._steps,
        )

    def get_diagnostics(self) -> dict[str, float | str | bool | None]:
        """Return the metrics this controller can honestly report.

        Deliberately limited to what the controller knows. Contact force and
        impulse are *not* included: the backend reads those from MuJoCo's
        solver, and a controller that invented them from a commanded position
        would be reporting an intention as a measurement.
        """
        state = self.get_state()
        return {
            "gripper_mode": state.mode.value,
            "commanded_finger_position_m": state.commanded_width_m / 2.0,
            "requested_width_m": state.requested_width_m,
            "commanded_width_m": state.commanded_width_m,
            "measured_width_m": state.measured_width_m,
            "target_width_m": state.target_width_m,
            "contact_detected": state.contact_detected,
            "engagement_limit_m": state.engagement_limit_m,
            "max_commanded_engagement_m": state.max_commanded_engagement_m,
            "max_measured_engagement_m": state.max_measured_engagement_m,
            "target_contact_force_n": self.target_contact_force_n,
            "gripper_steps": float(state.steps),
        }

