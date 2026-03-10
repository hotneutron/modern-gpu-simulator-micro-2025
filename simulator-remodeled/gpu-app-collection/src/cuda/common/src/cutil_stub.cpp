/*
 * Stub implementation of deprecated CUDA SDK cutil timer functions
 * These provide basic timing functionality for legacy benchmarks
 */

#include <sys/time.h>
#include <stdlib.h>
#include <string.h>

// Maximum number of timers
#define MAX_TIMERS 100

// Timer structure
typedef struct {
    int active;
    int running;
    double start_time;
    double accumulated_time;
    int count;
} CutTimer;

static CutTimer timers[MAX_TIMERS];
static int initialized = 0;

static void init_timers() {
    if (!initialized) {
        memset(timers, 0, sizeof(timers));
        initialized = 1;
    }
}

static double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

extern "C" {

// CUTBoolean type from cutil.h
typedef int CUTBoolean;
#define CUTTrue 1
#define CUTFalse 0

CUTBoolean cutCreateTimer(unsigned int* timer) {
    init_timers();
    for (int i = 0; i < MAX_TIMERS; i++) {
        if (!timers[i].active) {
            timers[i].active = 1;
            timers[i].running = 0;
            timers[i].start_time = 0;
            timers[i].accumulated_time = 0;
            timers[i].count = 0;
            *timer = i;
            return CUTTrue;
        }
    }
    return CUTFalse;
}

CUTBoolean cutDeleteTimer(unsigned int timer) {
    if (timer < MAX_TIMERS && timers[timer].active) {
        timers[timer].active = 0;
        return CUTTrue;
    }
    return CUTFalse;
}

CUTBoolean cutStartTimer(unsigned int timer) {
    if (timer < MAX_TIMERS && timers[timer].active) {
        timers[timer].running = 1;
        timers[timer].start_time = get_time_ms();
        return CUTTrue;
    }
    return CUTFalse;
}

CUTBoolean cutStopTimer(unsigned int timer) {
    if (timer < MAX_TIMERS && timers[timer].active && timers[timer].running) {
        double end_time = get_time_ms();
        timers[timer].accumulated_time += (end_time - timers[timer].start_time);
        timers[timer].count++;
        timers[timer].running = 0;
        return CUTTrue;
    }
    return CUTFalse;
}

CUTBoolean cutResetTimer(unsigned int timer) {
    if (timer < MAX_TIMERS && timers[timer].active) {
        timers[timer].running = 0;
        timers[timer].start_time = 0;
        timers[timer].accumulated_time = 0;
        timers[timer].count = 0;
        return CUTTrue;
    }
    return CUTFalse;
}

float cutGetTimerValue(unsigned int timer) {
    if (timer < MAX_TIMERS && timers[timer].active) {
        if (timers[timer].running) {
            double current = get_time_ms();
            return (float)(timers[timer].accumulated_time + (current - timers[timer].start_time));
        }
        return (float)timers[timer].accumulated_time;
    }
    return 0.0f;
}

float cutGetAverageTimerValue(unsigned int timer) {
    if (timer < MAX_TIMERS && timers[timer].active && timers[timer].count > 0) {
        return (float)(timers[timer].accumulated_time / timers[timer].count);
    }
    return 0.0f;
}

// Stub for cutFree - just wraps free()
void cutFree(void* ptr) {
    free(ptr);
}

// Compare functions for verification
CUTBoolean cutCompareub(const unsigned char* reference, const unsigned char* data,
                        unsigned int len) {
    for (unsigned int i = 0; i < len; i++) {
        if (reference[i] != data[i]) {
            return CUTFalse;
        }
    }
    return CUTTrue;
}

CUTBoolean cutComparef(const float* reference, const float* data,
                       unsigned int len) {
    for (unsigned int i = 0; i < len; i++) {
        if (reference[i] != data[i]) {
            return CUTFalse;
        }
    }
    return CUTTrue;
}

CUTBoolean cutComparei(const int* reference, const int* data,
                       unsigned int len) {
    for (unsigned int i = 0; i < len; i++) {
        if (reference[i] != data[i]) {
            return CUTFalse;
        }
    }
    return CUTTrue;
}

} // extern "C"
