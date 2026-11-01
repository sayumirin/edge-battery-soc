/* Portable C99, PyTorch reset-after GRU semantics; no heap allocation. */
#include <math.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include "weights.h"
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif
typedef struct { float h[HIDDEN]; } BatteryState;
static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
API void battery_reset(BatteryState *state) { memset(state, 0, sizeof(*state)); }
API float battery_step(BatteryState *state, const float raw[3]) {
    float x[3], input[3], next[HIDDEN];
    int j,k,g;
    for(j=0;j<3;j++) x[j]=(raw[j]-feature_mean[j])/feature_std[j]*feature_mask[j];
    for(j=0;j<3;j++) {
        input[j]=input_bias[j];
        for(k=0;k<3;k++) input[j]+=input_weight[j*3+k]*x[k];
    }
    for(j=0;j<HIDDEN;j++) {
        float a[3],b[3];
        for(g=0;g<3;g++) {
            int row=g*HIDDEN+j;
            a[g]=gru_bias_ih_l0[row]; b[g]=gru_bias_hh_l0[row];
            for(k=0;k<3;k++) a[g]+=gru_weight_ih_l0[row*3+k]*input[k];
            for(k=0;k<HIDDEN;k++) b[g]+=gru_weight_hh_l0[row*HIDDEN+k]*state->h[k];
        }
        float r=sigmoid(a[0]+b[0]),z=sigmoid(a[1]+b[1]);
        float n=tanhf(a[2]+r*b[2]);
        next[j]=(1.0f-z)*n+z*state->h[j];
    }
    memcpy(state->h,next,sizeof(next));
    float y=head_bias[0];
    for(j=0;j<HIDDEN;j++) y+=head_weight[j]*next[j];
    return y;
}
API size_t battery_state_bytes(void) { return sizeof(BatteryState); }
/* C runtime clock is platform-dependent; repeat enough steps for timer resolution. */
API double battery_benchmark(const float raw[3], int steps) {
    BatteryState state; volatile float sink=0;
    battery_reset(&state);
    clock_t start=clock();
    for(int i=0;i<steps;i++) sink+=battery_step(&state,raw);
    (void)sink;
    return (double)(clock()-start)/CLOCKS_PER_SEC/steps;
}
