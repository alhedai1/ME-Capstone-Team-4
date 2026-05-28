# Software Design Presentation Notes

Target time: about 4 minutes

## Slide 25: Software Design

Slide content:
- Vision-driven autonomous control
- Two cameras, one state machine
- YOLO for pole approach
- OpenCV circle tracking for bell feedback
- Software sequence: drive -> climb -> detect bell -> strike

Speaker notes:
My part focuses on how the robot makes decisions from camera input. The final software sequence is simpler than the full early plan: first the robot drives on the ground toward the pole, then it attaches and climbs using the magnetic wheels, then it tracks the bell circle and strikes. The software is still built around a state machine, so each phase has one clear perception goal and one clear control output.

## Slide 26: State Machine

Slide content:
- Search: find and center the pole
- Approach: drive toward pole using YOLO
- Attach/Climb: contact pole and climb with magnetic wheels
- Track Bell: use circle detection while climbing
- Strike: confirm bell, actuate striker, repeat after 3 seconds

Speaker notes:
The main structure is a finite state machine. The robot starts by searching for the pole, then approaches it, attaches to the pole, climbs, tracks the bell, and strikes. The reason we used this structure is reliability. If one part fails, like losing the pole for a few frames, the robot only has to recover inside that state instead of breaking the whole mission. The state machine also makes the software easier to tune, because the ground-driving logic and the bell-tracking logic are separated.

## Slide 27: Ground Navigation: AI Camera + YOLO

Slide content:
- AI camera runs YOLO pole detection
- Bounding box center gives steering error:
  `error_x = pole_center - frame_center`
- Bounding box width estimates distance to pole
- Control:
  left/right error -> steer
  centered -> drive forward
  wide box -> stop near pole

Speaker notes:
For ground navigation, the front AI camera detects the pole using YOLO. From the YOLO bounding box, we use two values. First, the horizontal center of the box gives the steering error. If the pole center is left of the image center, the robot turns left; if it is right, the robot turns right; and if it is inside the deadband, the robot drives forward. Second, the width of the bounding box is used as a simple distance estimate. As the robot gets closer, the pole appears wider, so once the width passes a threshold for several frames, the robot stops and starts the attach-and-climb phase.

## Slide 28: Attach and Climb Control

Slide content:
- Stop near pole after YOLO approach
- Drive forward to engage magnetic wheels
- Ramp motor speed before full climbing power
- Use camera feedback to decide when bell is in range
- No separate pole/bell orbit alignment in final sequence

Speaker notes:
After the ground approach, the software switches from navigation to climbing. At this point we do not perform a separate orbit alignment step. Instead, the robot is already placed facing the pole, so the control goal is to make contact, engage the magnetic wheels, and climb. The motors first drive forward to attach to the pole, then the climbing speed is ramped up instead of jumping instantly to full power. This helps reduce slipping and makes the transition from ground driving to vertical climbing more controlled.

## Slide 29: Bell Circle Tracking and Strike Timing

Slide content:
- Track bell circle while climbing
- Bell visible -> climb
- Bell lost -> stop motors and allow controlled slip-down
- Bell reacquired -> wait at least 3 seconds
- Confirm bell -> strike again

Speaker notes:
During climbing, the software uses visual feedback from the bell. The bell is close to the camera at this point, so OpenCV circle detection is enough. When the bell circle is visible, the robot keeps climbing and the striker can hit when the bell is confirmed. When the bell disappears from the camera view, that means the robot has passed the useful strike area, so the motors stop and the robot slips down. When the bell appears again, the robot waits at least three seconds to satisfy the challenge requirement, then it can strike again. This gives us a feedback loop for the two required hits without needing an exact height measurement.

## Slide 30: Fallbacks and Tuning

Slide content:
- Box and circle smoothing reduce noisy motion
- Stable-frame checks prevent false transitions
- Lost-frame counters handle brief detection loss
- Recovery search when target is lost too long
- MJPEG preview stream used for live tuning

Speaker notes:
The main challenge was not just detecting the pole or bell once. It was making the robot behave reliably with noisy camera data and real mechanical motion. So the software includes smoothing for bounding boxes and circle detections, stable-frame requirements before important transitions, and missed-frame counters so the robot does not immediately fail when the target disappears for a moment. We also used a live preview stream from the Raspberry Pi while testing, which made it much easier to tune thresholds and control parameters.

## Closing Transition

Speaker notes:
Overall, the software converts camera measurements into small control decisions: center the pole, approach it, climb the pole, track the bell circle, and strike with timing control. That is how the robot connects the operational workflow to real autonomous behavior.
