// Minimal stdlib.h for the freestanding liblzma wasm build (see ../xz.c for definitions)
#pragma once
#include <stddef.h>

void* malloc(size_t size);
void* calloc(size_t n, size_t size);
void free(void* p);
