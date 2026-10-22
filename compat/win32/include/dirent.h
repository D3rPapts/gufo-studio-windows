/* Win32 shadow of <dirent.h>; see gufo_posix.h. Only the descriptor-based
 * scan (fdopendir/readdir/closedir) is provided. */
#pragma once
#include "gufo_posix.h"

#define DT_UNKNOWN 0
#define DT_DIR 4
#define DT_REG 8
#define DT_LNK 10

struct dirent {
  unsigned char d_type;
  char d_name[1024];
};

typedef struct DIR DIR;

#ifdef __cplusplus
extern "C" {
#endif
/* Takes ownership of fd; closedir closes it. */
DIR* fdopendir(int fd);
struct dirent* readdir(DIR* dir);
int closedir(DIR* dir);
#ifdef __cplusplus
}
#endif
