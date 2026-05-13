# Robot State Machine

This is the first implementation target for autonomous behavior. Each state should have a timeout and should command zero motor output when it fails.

## Signals

- `front_pole`: pole detection from the forward/normal Pi camera.
- `front_bell_or_rod`: bell or rod detection from the forward/normal Pi camera.
- `upward_rod`: rod detection from the upward AI camera.
- `upward_bell`: bell detection from the upward AI camera.
- `pole_center_error_px`: pole center x-coordinate minus image center.
- `bell_pole_error_px`: bell/rod x-coordinate minus pole x-coordinate.
- `rod_angle_deg`: detected rod angle in the upward camera frame.
- `contact_or_near_pole`: physical contact, distance threshold, or vision area threshold.
- `climb_height`: encoder estimate, timed estimate, or other climb progress estimate.

## States

| State | Purpose | Inputs | Outputs | Exit Condition | Failure / Timeout |
| --- | --- | --- | --- | --- | --- |
| `IDLE` | Robot is safe and waiting. | Start command. | Motors off, striker off. | Start command received. | Stay idle. |
| `SEARCH_TOP_STRUCTURE` | Find the pole and bell/rod area. | `front_pole`, `front_bell_or_rod`. | Slow rotate. | Pole and bell/rod seen for N consecutive frames. | Stop and report no target. |
| `COARSE_ALIGN` | Put bell/rod and pole into the desired horizontal relationship. | `bell_pole_error_px`. | Slow lateral/turn correction. | `abs(bell_pole_error_px) < threshold` for N frames. | Return to search or stop. |
| `APPROACH_POLE` | Drive toward pole while keeping pole centered. | `front_pole`, `pole_center_error_px`, `contact_or_near_pole`. | Forward drive plus steering correction. | `contact_or_near_pole == true`. | Stop if pole lost or timeout. |
| `NEAR_POLE_FINE_ALIGN` | Align robot orientation using upward rod angle. | `upward_rod`, `rod_angle_deg`. | Slow rotate in place. | `abs(rod_angle_deg - reference_angle_deg) < threshold`. | Stop or return to coarse align. |
| `ATTACH` | Engage climbing contact with pole. | `contact_or_near_pole`, motor current if available. | Short controlled forward/grip command. | Attachment confirmed or fixed attach time elapsed. | Back off and stop. |
| `CLIMB` | Climb to bell height. | `climb_height`, `upward_bell`. | Climb motor command. | Bell visible at strike height or target climb height reached. | Stop and hold if possible. |
| `DETECT_BELL_TOP` | Locate bell after climbing. | `upward_bell`. | Small scan/rotate command. | Bell centered for N frames. | Stop or descend strategy. |
| `STRIKE` | Hit the bell. | Bell centered, actuator health. | Fire striker once. | Strike complete. | Stop actuator and report failure. |
| `DONE` | Mission complete. | Strike complete. | Motors off. | Manual reset. | Stay done. |
| `FAULT` | Safe failure state. | Any unsafe condition. | Motors off, striker off. | Manual reset. | Stay fault. |

## Safety Rules

- Any state that loses required vision for more than a short grace period must command zero velocity.
- Any actuator command should have a maximum duration.
- Autonomous motion should require an external enable switch or launch-time arming flag.
- Manual stop must override the state machine.
- The first implementation should log state transitions and the sensor values that caused them.
