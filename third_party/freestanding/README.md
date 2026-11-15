# Freestanding strip of libc++ for wasm use

Borrowed from

    /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1

then stripped to remove all streams, malloc, etc.

The strip keeps only what the wasm sources need and nothing that allocates: `std::allocator`, the
temporary buffer helpers, and the `allocator<void>` specializations are excluded under `__wasm__`
so the headers compile under any standard (they are built with `-std=c++2c`); the C++20 branches of
the originals would otherwise instantiate the stripped allocation internals.
