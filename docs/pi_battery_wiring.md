# Raspberry Pi Battery Wiring

Draft wiring plan for powering a Raspberry Pi 5 from a 3S LiPo pack. Verify component ratings before connecting hardware.

## Logic Power Path

```text
3S LiPo main + -> BMS B+
3S LiPo main - -> BMS B-
LiPo balance wires -> B-, B1, B2, B+ in the correct voltage order
BMS P+ -> fuse -> buck converter IN+
BMS P- -> buck converter IN-
buck USB-C output -> Raspberry Pi 5 USB-C power input
```

## Motor Power Path

Keep motor power separate from Pi logic power:

```text
3S LiPo / BMS output -> motor fuse or switch -> motor driver VM / motor power
motor driver GND -> battery/BMS P-
Pi GND -> motor driver logic GND
Pi GPIO -> motor driver logic inputs only
```

Do not power motors from the Pi 5V rail.

## Pre-Power Checks

Use a multimeter before connecting the Pi:

- Battery main voltage: about 9.0-12.6 V for a 3S LiPo.
- BMS `P+` to `P-`: same as battery output when the BMS is enabled.
- Buck output with no load: about 5.0-5.1 V.
- Buck output under expected load: still about 5.0-5.1 V.
- Buck current rating: enough for Pi 5 peak load and connected USB/camera devices.
- Fuse rating: above normal Pi load but below wiring/component damage current.
- Polarity: USB-C output polarity and BMS balance connector order are correct.

## Practical Notes

- Add a physical power switch or removable link for the robot.
- Add a separate emergency motor cutoff before autonomous testing.
- Use wire gauge appropriate for motor current, not just Pi current.
- Keep high-current motor wiring physically away from camera ribbon cables where possible.
- If the Pi reports undervoltage, measure buck output at the Pi under load, not only at the converter.
