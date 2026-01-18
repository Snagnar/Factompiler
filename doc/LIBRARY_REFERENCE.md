# Standard Library Reference

Facto includes a standard library of commonly-used functions for mathematical operations, memory patterns, and signal processing. This reference documents all available library functions.

## Table of Contents

- [Importing Libraries](#importing-libraries)
- [math.facto — Mathematical Functions](#mathfacto--mathematical-functions)
  - [abs](#abs)
  - [sign](#sign)
  - [min](#min)
  - [max](#max)
  - [clamp](#clamp)
  - [lerp](#lerp)
  - [between](#between)
  - [get_bit](#get_bit)
  - [set_bit](#set_bit)
  - [clear_bit](#clear_bit)
  - [toggle_bit](#toggle_bit)
  - [div_floor](#div_floor)
  - [mod_positive](#mod_positive)
- [memory_patterns.facto — Memory Functions](#memory_patternsfacto--memory-functions)
  - [edge_rising](#edge_rising)
  - [edge_falling](#edge_falling)
  - [clock](#clock)
  - [pulse_stretch](#pulse_stretch)
  - [debounce](#debounce)
  - [rate_limit](#rate_limit)
  - [ema](#ema)
  - [delay](#delay)
- [signal_processing.facto — Signal Processing](#signal_processingfacto--signal-processing)
  - [rgb_pack](#rgb_pack)
  - [rgb_unpack_r / rgb_unpack_g / rgb_unpack_b](#rgb_unpack_r--rgb_unpack_g--rgb_unpack_b)
  - [color_lerp](#color_lerp)
  - [brightness](#brightness)
  - [hsv_to_rgb](#hsv_to_rgb)
  - [deadband](#deadband)
  - [hysteresis](#hysteresis)
  - [quantize](#quantize)
  - [wrap](#wrap)
  - [rescale](#rescale)
  - [mux4](#mux4)
  - [priority_encode](#priority_encode)
  - [saturating_add / saturating_sub](#saturating_add--saturating_sub)
- [Best Practices](#best-practices)
- [See Also](#see-also)

## Importing Libraries

Use the `import` statement to include library functions in your program:

```facto
import "lib/math.facto";
import "lib/memory_patterns.facto";
import "lib/signal_processing.facto";
```

After importing, all functions from the library are available in your program's global namespace.

---

## math.facto — Mathematical Functions

Common mathematical functions optimized for Factorio circuit networks. All functions use the efficient `condition : value` syntax where applicable.

### abs

```facto
func abs(Signal x) → Signal
```

Returns the absolute value of `x`.

**Parameters:**
- `x` (Signal): The input value

**Returns:** Signal containing |x|

**Example:**
```facto
import "lib/math.facto";

Signal position = ("signal-P", -42);
Signal distance = abs(position);  # Result: 42
```

**Combinator Count:** 2 decider combinators

---

### sign

```facto
func sign(Signal x) → Signal
```

Returns the sign of `x`: -1 if negative, 0 if zero, 1 if positive.

**Parameters:**
- `x` (Signal): The input value

**Returns:** Signal containing -1, 0, or 1

**Example:**
```facto
import "lib/math.facto";

Signal velocity = ("signal-V", -100);
Signal direction = sign(velocity);  # Result: -1

Signal zero = ("signal-Z", 0);
Signal zero_sign = sign(zero);  # Result: 0
```

**Combinator Count:** 2 decider combinators

---

### min

```facto
func min(Signal a, Signal b) → Signal
```

Returns the smaller of two values.

**Parameters:**
- `a` (Signal): First value
- `b` (Signal): Second value

**Returns:** Signal containing the minimum of a and b

**Example:**
```facto
import "lib/math.facto";

Signal temp1 = ("signal-1", 75);
Signal temp2 = ("signal-2", 82);
Signal coldest = min(temp1, temp2);  # Result: 75
```

**Combinator Count:** 2 decider combinators

---

### max

```facto
func max(Signal a, Signal b) → Signal
```

Returns the larger of two values.

**Parameters:**
- `a` (Signal): First value
- `b` (Signal): Second value

**Returns:** Signal containing the maximum of a and b

**Example:**
```facto
import "lib/math.facto";

Signal bid1 = ("signal-1", 100);
Signal bid2 = ("signal-2", 150);
Signal highest = max(bid1, bid2);  # Result: 150
```

**Combinator Count:** 2 decider combinators

---

### clamp

```facto
func clamp(Signal x, int low, int high) → Signal
```

Constrains a value to be within a specified range [low, high].

**Parameters:**
- `x` (Signal): The value to clamp
- `low` (int): Minimum allowed value
- `high` (int): Maximum allowed value

**Returns:** Signal containing x clamped to [low, high]

**Example:**
```facto
import "lib/math.facto";

Signal raw_speed = ("signal-S", 150);
Signal safe_speed = clamp(raw_speed, 0, 100);  # Result: 100

Signal low_value = ("signal-L", -50);
Signal clamped_low = clamp(low_value, 0, 100);  # Result: 0

Signal normal = ("signal-N", 42);
Signal unchanged = clamp(normal, 0, 100);  # Result: 42
```

**Combinator Count:** 4 decider combinators

---

### lerp

```facto
func lerp(int a, int b, Signal t) → Signal
```

Linear interpolation between two values. The factor `t` ranges from 0 to 100, representing 0% to 100%.

**Parameters:**
- `a` (int): Start value (when t = 0)
- `b` (int): End value (when t = 100)
- `t` (Signal): Interpolation factor (0-100)

**Returns:** Signal containing the interpolated value

**Example:**
```facto
import "lib/math.facto";

# Fade between brightness levels based on time of day
Signal time_percent = ("signal-T", 50);  # 50% through the day
Signal brightness = lerp(0, 255, time_percent);  # Result: 127
```

**Combinator Count:** 2 arithmetic combinators

**Note:** Due to integer arithmetic, results are truncated. For precise interpolation, consider scaling your values.

---

### between

```facto
func between(Signal x, int low, int high) → Signal
```

Checks if a value is within a range (inclusive).

**Parameters:**
- `x` (Signal): The value to check
- `low` (int): Lower bound (inclusive)
- `high` (int): Upper bound (inclusive)

**Returns:** Signal containing 1 if low ≤ x ≤ high, 0 otherwise

**Example:**
```facto
import "lib/math.facto";

Signal temperature = ("signal-T", 75);
Signal in_comfort_zone = between(temperature, 65, 80);  # Result: 1

Signal too_hot = ("signal-H", 95);
Signal comfortable = between(too_hot, 65, 80);  # Result: 0
```

**Combinator Count:** 1 multi-condition decider combinator (condition folding)

---

### get_bit

```facto
func get_bit(Signal value, int pos) → Signal
```

Extracts a single bit from a value. Bit 0 is the least significant bit.

**Parameters:**
- `value` (Signal): The value to extract from
- `pos` (int): Bit position (0 = LSB)

**Returns:** Signal containing 0 or 1

**Example:**
```facto
import "lib/math.facto";

Signal flags = ("signal-F", 0b1010);  # Binary: 1010
Signal bit0 = get_bit(flags, 0);  # Result: 0
Signal bit1 = get_bit(flags, 1);  # Result: 1
Signal bit2 = get_bit(flags, 2);  # Result: 0
Signal bit3 = get_bit(flags, 3);  # Result: 1
```

**Combinator Count:** 2 arithmetic combinators (shift + AND)

---

### set_bit

```facto
func set_bit(Signal value, int pos) → Signal
```

Sets a bit to 1 at the specified position.

**Parameters:**
- `value` (Signal): The value to modify
- `pos` (int): Bit position to set (0 = LSB)

**Returns:** Signal with the specified bit set to 1

**Example:**
```facto
import "lib/math.facto";

Signal flags = ("signal-F", 0b0000);
Signal with_bit2 = set_bit(flags, 2);  # Result: 0b0100 = 4
```

**Combinator Count:** 1 arithmetic combinator (OR)

---

### clear_bit

```facto
func clear_bit(Signal value, int pos) → Signal
```

Clears a bit to 0 at the specified position.

**Parameters:**
- `value` (Signal): The value to modify
- `pos` (int): Bit position to clear (0 = LSB)

**Returns:** Signal with the specified bit set to 0

**Example:**
```facto
import "lib/math.facto";

Signal flags = ("signal-F", 0b1111);
Signal without_bit2 = clear_bit(flags, 2);  # Result: 0b1011 = 11
```

**Combinator Count:** 2 arithmetic combinators (XOR + AND)

---

### toggle_bit

```facto
func toggle_bit(Signal value, int pos) → Signal
```

Flips a bit at the specified position.

**Parameters:**
- `value` (Signal): The value to modify
- `pos` (int): Bit position to toggle (0 = LSB)

**Returns:** Signal with the specified bit flipped

**Example:**
```facto
import "lib/math.facto";

Signal flags = ("signal-F", 0b1010);
Signal toggled = toggle_bit(flags, 1);  # Result: 0b1000 = 8
```

**Combinator Count:** 1 arithmetic combinator (XOR)

---

### div_floor

```facto
func div_floor(Signal a, Signal b) → Signal
```

Integer division that always rounds toward negative infinity (like Python's `//` operator).

**Parameters:**
- `a` (Signal): Dividend
- `b` (Signal): Divisor

**Returns:** Signal containing floor(a/b)

**Example:**
```facto
import "lib/math.facto";

# Standard division truncates toward zero
Signal std = ("signal-A", -7) / 3;  # Result: -2

# div_floor rounds toward negative infinity
Signal a = ("signal-A", -7);
Signal b = ("signal-B", 3);
Signal floored = div_floor(a, b);  # Result: -3
```

**Combinator Count:** ~5 combinators

**Use Case:** Useful for coordinate calculations where you need consistent rounding behavior regardless of sign.

---

### mod_positive

```facto
func mod_positive(Signal a, Signal b) → Signal
```

Modulo operation that always returns a positive result (like Python's `%` operator).

**Parameters:**
- `a` (Signal): Dividend
- `b` (Signal): Divisor

**Returns:** Signal containing the positive remainder

**Example:**
```facto
import "lib/math.facto";

# Standard modulo can return negative
Signal std = ("signal-A", -7) % 3;  # Result: -1

# mod_positive always returns positive
Signal a = ("signal-A", -7);
Signal b = ("signal-B", 3);
Signal positive = mod_positive(a, b);  # Result: 2
```

**Combinator Count:** ~6 combinators

**Use Case:** Useful for wrapping values (like array indices or angles) where negative results would be incorrect.

---

## memory_patterns.facto — Memory Patterns

Common memory patterns for state management. These patterns handle edge cases and provide reliable behavior.

### edge_rising

```facto
func edge_rising(Signal input) → Signal
```

Detects rising edges: outputs 1 for exactly one tick when input transitions from 0 to non-zero.

**Parameters:**
- `input` (Signal): The signal to monitor

**Returns:** Signal that is 1 for one tick on rising edge, 0 otherwise

**Example:**
```facto
import "lib/memory_patterns.facto";

Signal button = ("signal-button", 0);  # Wire from your circuit
Signal pressed = edge_rising(button);

# pressed is 1 only on the tick when button goes from 0 → non-zero
```

**Combinator Count:** Memory cell + 2 decider combinators

**Use Case:** Button press detection, trigger-once logic, event counting.

---

### edge_falling

```facto
func edge_falling(Signal input) → Signal
```

Detects falling edges: outputs 1 for exactly one tick when input transitions from non-zero to 0.

**Parameters:**
- `input` (Signal): The signal to monitor

**Returns:** Signal that is 1 for one tick on falling edge, 0 otherwise

**Example:**
```facto
import "lib/memory_patterns.facto";

Signal motion_sensor = ("signal-M", 0);  # Wire from detector
Signal motion_stopped = edge_falling(motion_sensor);

# motion_stopped is 1 only when motion_sensor goes from non-zero → 0
```

**Combinator Count:** Memory cell + 2 decider combinators

**Use Case:** Detecting when something stops, release detection, cleanup triggers.

---

### clock

```facto
func clock(int period) → Signal
```

Creates a modulo counter that cycles from 0 to period-1.

**Parameters:**
- `period` (int): The period of the clock (cycles 0 to period-1)

**Returns:** Signal containing the current tick count

**Example:**
```facto
import "lib/memory_patterns.facto";

Signal tick = clock(60);  # Counts 0, 1, 2, ..., 59, 0, 1, ...

# Use for blinking lights
Entity lamp = place("small-lamp", 0, 0);
lamp.enable = tick < 30;  # On for first half, off for second half
```

**Combinator Count:** 1 arithmetic combinator with feedback

**Use Case:** Animation timing, periodic events, phase control.

---

### pulse_stretch

```facto
func pulse_stretch(Signal input, int duration) → Signal
```

Extends a brief pulse to last for a specified number of ticks.

**Parameters:**
- `input` (Signal): The trigger signal (pulse to extend)
- `duration` (int): How many ticks the output should remain high

**Returns:** Signal that stays high for `duration` ticks after trigger

**Example:**
```facto
import "lib/memory_patterns.facto";

Signal trigger = ("signal-T", 0);  # Brief 1-tick pulse
Signal extended = pulse_stretch(trigger, 60);  # Stays high for 60 ticks

Entity alarm_lamp = place("small-lamp", 0, 0);
alarm_lamp.enable = extended > 0;  # Lamp stays on for 1 second (60 ticks)
```

**Combinator Count:** Memory cell + ~3 combinators

**Use Case:** Alert indicators, minimum display time, debounce output.

---

### debounce

```facto
func debounce(Signal input, int ticks) → Signal
```

Filters out rapid changes by requiring input to be stable for a number of ticks before the output changes.

**Parameters:**
- `input` (Signal): The noisy input signal
- `ticks` (int): Required stability time before output changes

**Returns:** Debounced signal

**Example:**
```facto
import "lib/memory_patterns.facto";

Signal raw_button = ("signal-B", 0);  # Noisy button signal
Signal clean_button = debounce(raw_button, 5);  # Must be stable for 5 ticks

# clean_button only changes after raw_button has been stable for 5 ticks
```

**Combinator Count:** 2 memory cells + ~3 combinators

**Use Case:** Noisy sensor filtering, button debouncing, glitch rejection.

---

### rate_limit

```facto
func rate_limit(Signal input, int max_change) → Signal
```

Smoothly tracks an input value but limits how fast it can change per tick.

**Parameters:**
- `input` (Signal): The target value to track
- `max_change` (int): Maximum change allowed per tick

**Returns:** Smoothed signal that approaches input at limited rate

**Example:**
```facto
import "lib/memory_patterns.facto";

Signal target_speed = ("signal-T", 100);  # Desired speed
Signal actual_speed = rate_limit(target_speed, 5);  # Max 5 units/tick change

# actual_speed smoothly approaches target_speed
# If target jumps from 0 to 100, actual_speed takes 20 ticks to reach it
```

**Combinator Count:** Memory cell + ~8 combinators

**Use Case:** Smooth motor control, gradual transitions, acceleration limiting.

---

### ema

```facto
func ema(Signal input, int alpha_num, int alpha_denom) → Signal
```

Exponential Moving Average for signal smoothing. New values are weighted by alpha = alpha_num / alpha_denom.

**Parameters:**
- `input` (Signal): The raw input signal
- `alpha_num` (int): Numerator of the smoothing factor
- `alpha_denom` (int): Denominator of the smoothing factor

**Returns:** Smoothed signal using EMA formula

**Example:**
```facto
import "lib/memory_patterns.facto";

Signal noisy_sensor = ("signal-S", 0);  # Wire from sensor

# 25% weight on new values (alpha = 1/4)
Signal smoothed = ema(noisy_sensor, 1, 4);

# Higher alpha = faster response but less smoothing
# Lower alpha = slower response but smoother output
```

**Combinator Count:** Memory cell + 3 arithmetic combinators

**Use Case:** Sensor smoothing, noise reduction, trend following.

**Note:** The formula is: `output = previous + (input - previous) * alpha_num / alpha_denom`

---

### delay

```facto
func delay(Signal input) → Signal
```

Delays a signal by exactly 1 tick.

**Parameters:**
- `input` (Signal): The signal to delay

**Returns:** Signal containing the previous tick's input value

**Example:**
```facto
import "lib/memory_patterns.facto";

Signal current = ("signal-C", 0);
Signal previous = delay(current);

# Calculate change per tick
Signal delta = current - previous;
```

**Combinator Count:** Memory cell (optimized to feedback loop)

**Use Case:** Pipeline synchronization, derivative calculation, state comparison.

---

## signal_processing.facto — Signal Processing

Utility functions for signal processing, color manipulation, and common circuit patterns. Complements math.facto and memory_patterns.facto.

```facto
import "lib/signal_processing.facto";
```

---

### rgb_pack

```facto
func rgb_pack(Signal r, Signal g, Signal b) → Signal
```

Pack three 8-bit RGB values (0-255) into a single 32-bit integer.

**Format:** `0x00RRGGBB` (bits 16-23: R, bits 8-15: G, bits 0-7: B)

**Parameters:**
- `r` (Signal): Red channel (0-255)
- `g` (Signal): Green channel (0-255)
- `b` (Signal): Blue channel (0-255)

**Returns:** Signal containing packed RGB value

**Example:**
```facto
import "lib/signal_processing.facto";

Signal packed = rgb_pack(255 | "signal-R", 128 | "signal-G", 64 | "signal-B");
# Result: 0xFF8040 = 16744512
```

---

### rgb_unpack_r / rgb_unpack_g / rgb_unpack_b

```facto
func rgb_unpack_r(Signal packed) → Signal
func rgb_unpack_g(Signal packed) → Signal
func rgb_unpack_b(Signal packed) → Signal
```

Unpack individual RGB channels from a packed color value.

**Parameters:**
- `packed` (Signal): Packed RGB value from `rgb_pack`

**Returns:** Signal containing the extracted channel (0-255)

**Example:**
```facto
import "lib/signal_processing.facto";

Signal color = 0xFF8040 | "signal-C";
Signal r = rgb_unpack_r(color);  # 255
Signal g = rgb_unpack_g(color);  # 128
Signal b = rgb_unpack_b(color);  # 64
```

---

### color_lerp

```facto
func color_lerp(Signal c1, Signal c2, Signal t) → Signal
```

Linearly interpolate between two packed RGB colors.

**Parameters:**
- `c1` (Signal): First color (packed RGB)
- `c2` (Signal): Second color (packed RGB)
- `t` (Signal): Interpolation factor 0-100 (0 = c1, 100 = c2)

**Returns:** Signal containing interpolated packed RGB

**Example:**
```facto
import "lib/signal_processing.facto";

Signal red = rgb_pack(255, 0, 0);
Signal blue = rgb_pack(0, 0, 255);
Signal purple = color_lerp(red, blue, 50);  # Halfway between red and blue
```

---

### brightness

```facto
func brightness(Signal r, Signal g, Signal b) → Signal
```

Calculate perceived brightness of an RGB color using standard luminance formula.

**Formula:** `(77*R + 150*G + 29*B) / 256` (approximates 0.299R + 0.587G + 0.114B)

**Parameters:**
- `r` (Signal): Red channel (0-255)
- `g` (Signal): Green channel (0-255)
- `b` (Signal): Blue channel (0-255)

**Returns:** Signal containing brightness (0-255)

---

### hsv_to_rgb

```facto
func hsv_to_rgb(Signal h, Signal s, Signal v) → Signal
```

Convert HSV color to packed RGB.

**Parameters:**
- `h` (Signal): Hue (0-359 degrees)
- `s` (Signal): Saturation (0-100 percent)
- `v` (Signal): Value/brightness (0-100 percent)

**Returns:** Signal containing packed RGB (0x00RRGGBB)

**Example:**
```facto
import "lib/signal_processing.facto";

# Cycle through hues for a rainbow effect
Memory tick: "signal-T";
tick.write((tick.read() + 1) % 360);

Signal rainbow = hsv_to_rgb(tick.read(), 100, 100);
lamp.rgb = rainbow | "signal-white";
```

---

### deadband

```facto
func deadband(Signal input, int center, int threshold) → Signal
```

Filter out small fluctuations around a setpoint. Returns center when input is within threshold of center.

**Parameters:**
- `input` (Signal): Input value to filter
- `center` (int): Center/setpoint value
- `threshold` (int): Deadband radius

**Returns:** Signal containing center if within threshold, input otherwise

**Example:**
```facto
import "lib/signal_processing.facto";

Signal sensor = ("signal-S", 0);
Signal filtered = deadband(sensor, 100, 5);
# 95-105 → 100, outside that range → actual value
```

---

### hysteresis

```facto
func hysteresis(Signal input, int low_threshold, int high_threshold) → Signal
```

Schmitt trigger behavior with memory. Output goes high when input exceeds high_threshold, low when below low_threshold.

**Parameters:**
- `input` (Signal): Input signal
- `low_threshold` (int): Turn off when input falls below this
- `high_threshold` (int): Turn on when input rises above this

**Returns:** Signal containing 0 or 1

**Example:**
```facto
import "lib/signal_processing.facto";

Signal temperature = ("signal-T", 0);
Signal heater_on = hysteresis(temperature, 18, 22);
# Turns on at 22°, stays on until drops to 18°
```

---

### quantize

```facto
func quantize(Signal value, int step) → Signal
```

Round value to nearest multiple of step.

**Parameters:**
- `value` (Signal): Value to quantize
- `step` (int): Step size

**Returns:** Signal containing quantized value

**Example:**
```facto
import "lib/signal_processing.facto";

Signal raw = 47 | "signal-R";
Signal stepped = quantize(raw, 10);  # Result: 50
```

---

### wrap

```facto
func wrap(Signal value, int min_val, int max_val) → Signal
```

Wrap value to be within range [min_val, max_val). Like modulo but for any range.

**Parameters:**
- `value` (Signal): Value to wrap
- `min_val` (int): Minimum of range
- `max_val` (int): Maximum of range (exclusive)

**Returns:** Signal containing wrapped value

**Example:**
```facto
import "lib/signal_processing.facto";

Signal angle = 370 | "signal-A";
Signal wrapped = wrap(angle, 0, 360);  # Result: 10
```

---

### rescale

```facto
func rescale(Signal value, int in_min, int in_max, int out_min, int out_max) → Signal
```

Linearly rescale a value from one range to another with clamping.

**Parameters:**
- `value` (Signal): Input value
- `in_min`, `in_max` (int): Input range
- `out_min`, `out_max` (int): Output range

**Returns:** Signal containing rescaled value

**Example:**
```facto
import "lib/signal_processing.facto";

Signal percentage = 50 | "signal-P";
Signal byte_value = rescale(percentage, 0, 100, 0, 255);  # Result: 127
```

---

### mux4

```facto
func mux4(Signal sel, Signal a, Signal b, Signal c, Signal d) → Signal
```

4-way multiplexer. Select one of four values based on selector (0-3).

**Parameters:**
- `sel` (Signal): Selector (0-3)
- `a`, `b`, `c`, `d` (Signal): Values to select from

**Returns:** Signal containing selected value

**Example:**
```facto
import "lib/signal_processing.facto";

Signal mode = 2 | "signal-M";
Signal speed = mux4(mode, 10, 25, 50, 100);  # Result: 50
```

---

### priority_encode

```facto
func priority_encode(Signal b0, Signal b1, Signal b2, Signal b3) → Signal
```

Returns index (0-3) of highest-priority set input. Returns -1 if no input set.

**Parameters:**
- `b0`, `b1`, `b2`, `b3` (Signal): Input bits (b3 has highest priority)

**Returns:** Signal containing 0-3 or -1

**Example:**
```facto
import "lib/signal_processing.facto";

Signal idx = priority_encode(1, 0, 1, 0);  # Result: 2 (b2 is highest set)
```

---

### saturating_add / saturating_sub

```facto
func saturating_add(Signal a, Signal b, int max_val) → Signal
func saturating_sub(Signal a, Signal b, int min_val) → Signal
```

Add or subtract with saturation (no overflow/underflow).

**Parameters:**
- `a`, `b` (Signal): Operands
- `max_val` or `min_val` (int): Saturation limit

**Returns:** Signal containing saturated result

**Example:**
```facto
import "lib/signal_processing.facto";

Signal x = saturating_add(200, 100, 255);  # Result: 255 (not 300)
Signal y = saturating_sub(10, 50, 0);      # Result: 0 (not -40)
```

---

## Best Practices

### Choosing Between Functions

**For clamping values:**
- Use `clamp()` when you have both min and max bounds
- Use `max(value, min_bound)` or `min(value, max_bound)` for single-sided bounds

**For conditional selection:**
- Use the `condition : value` syntax directly for simple cases
- Use library functions for complex multi-way selection

**For edge detection:**
- `edge_rising()` for button presses and trigger events
- `edge_falling()` for release detection and cleanup

### Performance Considerations

1. **Inlining:** All library functions are inlined at each call site. Calling the same function twice creates two copies of the combinators.

2. **Memory instances:** Functions with `Memory` declarations create separate memory cells for each call. This is usually desired (independent counters) but can be wasteful if shared state is needed.

3. **Combinator counts:** The documented combinator counts are approximate and may vary based on context and compiler optimizations.

### Common Patterns

**Hysteresis with min/max:**
```facto
import "lib/math.facto";

Memory state: "signal-S";
Signal input = ("signal-I", 0);

# Turn on when input > 80, off when input < 20
Signal turn_on = (input > 80) : 1;
Signal turn_off = (input < 20) : 0;
Signal keep = (input >= 20 && input <= 80) : state.read();
state.write(turn_on + keep);  # turn_off contributes 0
```

**Smooth value tracking:**
```facto
import "lib/memory_patterns.facto";

Signal target = ("signal-T", 0);  # From circuit
Signal smooth = rate_limit(target, 10);  # Max 10 units/tick
Signal extra_smooth = ema(smooth, 1, 8);  # Additional smoothing
```

---

## See Also

- [Language Specification](../LANGUAGE_SPEC.md) — Complete language reference
- [Functions and Modules](06_functions.md) — How to write and use functions
- [Advanced Concepts](07_advanced_concepts.md) — Optimization patterns
