#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <ctype.h>

#define MAX_DIM        4096
#define MAX_SEQ_LEN    2048
#define MAX_TOKEN_LEN  256
#define MAX_PROMPT_LEN 1024
#define EPS            1e-8f

typedef float          f32;
typedef unsigned char  u8;
typedef unsigned short u16;
typedef unsigned int   u32;
typedef unsigned long long u64;

#define TOKEN_BOS  1
#define TOKEN_EOS  2

#endif