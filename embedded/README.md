# Portable float32 inference

`gru.c` implements the model's input normalization, input affine layer, reset-after GRU and linear output. `scripts/verify_c.py` exports the selected trained checkpoint as `weights.h`, including preprocessing statistics and the optional feature mask, then compiles and checks the implementation.

```text
uv run python scripts/verify_c.py --run runs/source-v3 --output results/c-inference-v3
```

For a minimal single-translation-unit C99 application, put the selected model's `weights.h` on the compiler include path and include `gru.c` once. `BatteryState` owns the persistent hidden state. Call `battery_reset(&state)` once at the start of a recording; call `battery_step(&state, raw)` for each 1 Hz sample, where `raw` is a contiguous three-float array containing voltage (V), discharge-positive current (A), and battery surface temperature (C). Return values are SoC ratios and are not clipped. Independent recordings require separate/reset states.

```c
#include "gru.c"
int main(void) {
    BatteryState state;
    float raw[3] = {3.7f, 1.0f, 25.0f};
    battery_reset(&state);
    float soc = battery_step(&state, raw);
    (void)soc;
    return 0;
}
```

Compile this application as a single source file (do not also compile a second copy of gru.c). Standard C math support is required. GCC/Clang builds normally link with `-lm`; verified Windows builds use MSVC. The generated `HIDDEN` macro defines state size, so headers from different models must not be mixed.

The kernel performs no heap allocations. The reported explicit workspace excludes compiler-generated stack overhead. This is a floating-point kernel; dynamic int8 PyTorch checkpoints cannot be substituted into its weights. Host latency, DLL size and array sizes are not measurements of MCU RAM/Flash, power, or timing. A real board port still needs compiler, sampling, memory-map and timing validation.

Streaming inference retains full history. Training/evaluation windows reset their initial state; compare C and PyTorch only using the same state lifetime and input history.

Full-recording accuracy is now evaluated separately in [the modular results](../results/modular-v2/README.md). Across 15 test files and three seeds (577,041 processed samples), continuous and 300-second reset-window predictions differ by up to 26.674 percentage points. Numerical C/PyTorch parity under matching history does not imply equivalence to sliding-window accuracy.
