/* Win32 shadow of <sys/random.h>; see gufo_posix.h. */
#pragma once
#include "gufo_posix.h"

#define GRND_NONBLOCK 0x01
#define GRND_RANDOM 0x02

#ifdef __cplusplus
extern "C"
#endif
/* Fills the buffer from the system CSPRNG (BCryptGenRandom); flags are
 * accepted and ignored because it never blocks. */
ssize_t getrandom(void* buffer, size_t length, unsigned int flags);
