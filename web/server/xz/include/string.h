// Minimal string.h for the freestanding liblzma wasm build (see ../xz.c for definitions)
#pragma once
#include <stddef.h>

void* memcpy(void* dst, const void* src, size_t n);
void* memmove(void* dst, const void* src, size_t n);
void* memset(void* s, int c, size_t n);
int memcmp(const void* a, const void* b, size_t n);
