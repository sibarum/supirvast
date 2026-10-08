package dev.supirvast.vastir.pass;

import dev.supirvast.vastir.type.Type;

/** A buffer every pass refers to by name: its element type and length. */
public record BufferSpec(String name, Type element, int length) {}
