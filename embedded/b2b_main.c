/* Back-to-back test driver for gru.c.
 * Input file: repeated records of [uint32 n][n x 3 float32 raw samples]; the state resets per record.
 * Output file: one float32 SoC estimate per sample, in input order. Little-endian on both sides. */
#include <stdint.h>
#include <stdio.h>
#include "gru.c"

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s inputs.bin outputs.bin\n", argv[0]);
        return 2;
    }
    FILE *in = fopen(argv[1], "rb");
    FILE *out = fopen(argv[2], "wb");
    if (!in || !out) {
        perror("open");
        return 2;
    }
    BatteryState state;
    uint32_t n;
    float raw[3];
    unsigned long streams = 0, steps = 0;
    while (fread(&n, sizeof n, 1, in) == 1) {
        battery_reset(&state);
        for (uint32_t i = 0; i < n; i++) {
            if (fread(raw, sizeof(float), 3, in) != 3) {
                fprintf(stderr, "truncated input in stream %lu\n", streams);
                return 1;
            }
            float soc = battery_step(&state, raw);
            if (fwrite(&soc, sizeof soc, 1, out) != 1) {
                perror("write");
                return 1;
            }
        }
        streams++;
        steps += n;
    }
    fclose(in);
    if (fclose(out) != 0) {
        perror("close");
        return 1;
    }
    printf("params=%s streams=%lu steps=%lu state_bytes=%lu\n", battery_params_id(), streams, steps,
           (unsigned long)battery_state_bytes());
    return 0;
}
