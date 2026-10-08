/**
 * A multi-pass program as data: {@link dev.supirvast.vastir.pass.Pass passes}, each a kernel over named buffers,
 * and the {@link dev.supirvast.vastir.pass.BufferSpec buffers} they name. A sort's count, scans and permute; a
 * simulation's step. Nothing here runs anything; {@code PassRunner} in vastir-tools does, on either backend.
 */
package dev.supirvast.vastir.pass;
