package dev.supirvast.vastir.pass;

import dev.supirvast.vastir.build.Body;
import dev.supirvast.vastir.core.AtomicOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.SharedArray;
import dev.supirvast.vastir.type.Type;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import static dev.supirvast.vastir.build.Body.F32;
import static dev.supirvast.vastir.build.Body.I32;
import static dev.supirvast.vastir.build.Body.add;
import static dev.supirvast.vastir.build.Body.div;
import static dev.supirvast.vastir.build.Body.eq;
import static dev.supirvast.vastir.build.Body.i;
import static dev.supirvast.vastir.build.Body.load;
import static dev.supirvast.vastir.build.Body.lt;
import static dev.supirvast.vastir.build.Body.not;
import static dev.supirvast.vastir.build.Body.sharedLoad;
import static dev.supirvast.vastir.build.Body.sub;
import static dev.supirvast.vastir.build.Body.v;

/**
 * A counting sort on the device, over resident buffers: items sorted by a key with a known bound, such as the
 * cell of a grid each particle or body is in. The key is the caller's; this is the rest.
 *
 * <ol>
 *   <li>{@link #count}: a fragment of the caller's own kernel, one invocation per item. Given the item's key, it
 *       keeps the key and takes an integer atomic add on the key's count. The add returns the count before it,
 *       which is the item's rank among the items with its key — so counting and ranking are one pass, and the
 *       caller's pass that computes the key is that pass.</li>
 *   <li>{@link #scan}: three passes that turn the counts into starts, an exclusive prefix sum —
 *       {@link #scanBlocks} within blocks of {@link #BLOCK} in workgroup memory, {@link #scanSums} over the block
 *       totals in one workgroup, {@link #addOffsets} to finish. The last also zeroes the counts, so the next sort
 *       needs no clearing pass of its own.</li>
 *   <li>Then either {@link #permute}, each item's fields copied to its key's start plus its rank, or
 *       {@link #order}, each item's index written there instead: a list of items by key, leaving the items where
 *       they are.</li>
 * </ol>
 *
 * <p>The counts are {@code keys + 1} long, with the last never counted into, so its exclusive sum — the last start
 * — is the item count, and key {@code k} holds items {@code [start[k], start[k + 1])}.
 *
 * <p><b>Not stable.</b> Ranks are handed out in whatever order the atomics land, so items with one key come out in
 * a different order each run. The starts are exact and every item is in its key's run; a sum over a run, taken in
 * the order the run holds, is not repeatable bit for bit. A stable sort, ranking by position within a workgroup
 * rather than by atomic, is what would make it so.
 *
 * <p>Every pass here must be registered with a workgroup of {@link #BLOCK}: the scans need it, and the others do
 * not care. Integer atomics and workgroup memory only, so no optional device capability.
 */
public final class CountingSort {

    /** The scan's block, and the workgroup every pass must be registered with. */
    public static final int BLOCK = 256;

    private static final int ROUNDS = Integer.numberOfTrailingZeros(BLOCK);

    private CountingSort() {
    }

    // --- the count, in the caller's kernel -----------------------------------------------------------------

    /**
     * Appends the count to {@code b}: {@code key} kept in {@code keys} at {@code item}, its count taken atomically
     * from {@code counts}, and the count before it kept in {@code ranks} as the item's rank. The counts must be zero
     * when the pass starts, which {@link #addOffsets} leaves them. The buffers are the caller's bindings, all i32.
     */
    public static void count(Body b, Expr item, Expr key, Buffer counts, Buffer keys, Buffer ranks) {
        LocalVar k = b.let("sortKey", key);
        LocalVar rank = b.fetchAtomic("sortRank", AtomicOp.ADD, counts, v(k), i(1));
        b.store(keys, item, v(k));
        b.store(ranks, item, v(rank));
    }

    // --- the exclusive scan --------------------------------------------------------------------------------

    public static final Buffer SCAN_COUNTS = new Buffer("counts", 0, I32);
    public static final Buffer SCAN_STARTS = new Buffer("starts", 1, I32);
    public static final Buffer SCAN_SUMS = new Buffer("blockSums", 2, I32);
    public static final List<Buffer> SCAN_BLOCKS_BUFFERS = List.of(SCAN_COUNTS, SCAN_STARTS, SCAN_SUMS);

    /**
     * One invocation per element of the {@code length} counts, in workgroups of {@link #BLOCK}: each block's
     * exclusive sums into the starts, and its total into the block sums.
     */
    public static Function scanBlocks(int length) {
        SharedArray tile = new SharedArray("tile", I32, BLOCK);
        Body b = new Body();
        LocalVar lid = b.let("lid", new Expr.LocalInvocationId());
        LocalVar k = b.let("k", new Expr.InvocationId());
        LocalVar count = b.let("count", i(0));
        b.when(lt(v(k), i(length)), t -> t.set(count, load(SCAN_COUNTS, v(k))));
        LocalVar inclusive = scanTile(b, tile, lid, count);
        b.when(lt(v(k), i(length)), t -> t.store(SCAN_STARTS, v(k), sub(v(inclusive), v(count))));
        b.when(eq(v(lid), i(BLOCK - 1)), t -> t.store(SCAN_SUMS, new Expr.WorkgroupId(), v(inclusive)));
        return function("sortScanBlocks", b);
    }

    public static final Buffer SUMS_SUMS = new Buffer("blockSums", 0, I32);
    public static final List<Buffer> SCAN_SUMS_BUFFERS = List.of(SUMS_SUMS);

    /**
     * The block sums of {@code length} counts, scanned exclusively in place by a single workgroup: dispatch
     * exactly {@link #BLOCK} invocations. The chunks are unrolled, so every barrier is at the top level.
     */
    public static Function scanSums(int length) {
        int blocks = blocks(length);
        SharedArray tile = new SharedArray("tile", I32, BLOCK);
        Body b = new Body();
        LocalVar lid = b.let("lid", new Expr.LocalInvocationId());
        LocalVar carry = b.let("carry", i(0));
        for (int chunk = 0; chunk * BLOCK < blocks; chunk++) {
            LocalVar k = b.let("k", add(v(lid), i(chunk * BLOCK)));
            LocalVar sum = b.let("sum", i(0));
            b.when(lt(v(k), i(blocks)), t -> t.set(sum, load(SUMS_SUMS, v(k))));
            LocalVar inclusive = scanTile(b, tile, lid, sum);
            b.when(lt(v(k), i(blocks)), t -> t.store(SUMS_SUMS, v(k), add(v(carry), sub(v(inclusive), v(sum)))));
            b.set(carry, add(v(carry), sharedLoad(tile, i(BLOCK - 1))));
            b.barrier();   // everyone has read the chunk's total before the next chunk overwrites the tile
        }
        return function("sortScanSums", b);
    }

    public static final Buffer OFFSET_STARTS = new Buffer("starts", 0, I32);
    public static final Buffer OFFSET_SUMS = new Buffer("blockSums", 1, I32);
    public static final Buffer OFFSET_COUNTS = new Buffer("counts", 2, I32);
    public static final List<Buffer> ADD_OFFSETS_BUFFERS = List.of(OFFSET_STARTS, OFFSET_SUMS, OFFSET_COUNTS);

    /** One invocation per element of the {@code length} counts: the block offsets added, the counts zeroed. */
    public static Function addOffsets(int length) {
        Body b = new Body();
        LocalVar k = b.let("k", new Expr.InvocationId());
        b.when(lt(v(k), i(length)), t -> {
            t.store(OFFSET_STARTS, v(k), add(load(OFFSET_STARTS, v(k)), load(OFFSET_SUMS, div(v(k), i(BLOCK)))));
            t.store(OFFSET_COUNTS, v(k), i(0));
        });
        return function("sortAddOffsets", b);
    }

    /**
     * The three scan passes over named buffers: {@code counts} and {@code starts} of {@code length} words and
     * {@code sums} of {@link #blocks blocks(length)}, all i32. The counts are zero after.
     */
    public static List<Pass> scan(int length, String counts, String starts, String sums) {
        return List.of(
                new Pass("scanBlocks", scanBlocks(length), SCAN_BLOCKS_BUFFERS, List.of(counts, starts, sums), length),
                new Pass("scanSums", scanSums(length), SCAN_SUMS_BUFFERS, List.of(sums), BLOCK),
                new Pass("addOffsets", addOffsets(length), ADD_OFFSETS_BUFFERS, List.of(starts, sums, counts),
                        length));
    }

    // --- the items moved, or listed ------------------------------------------------------------------------

    public static final Buffer PERMUTE_KEYS = new Buffer("keys", 0, I32);
    public static final Buffer PERMUTE_RANKS = new Buffer("ranks", 1, I32);
    public static final Buffer PERMUTE_STARTS = new Buffer("starts", 2, I32);

    /** The keys, ranks and starts, then {@code fields} f32 fields in, then the same fields out. */
    public static List<Buffer> permuteBuffers(int fields) {
        List<Buffer> buffers = new ArrayList<>(List.of(PERMUTE_KEYS, PERMUTE_RANKS, PERMUTE_STARTS));
        buffers.addAll(fields("in", 3, fields));
        buffers.addAll(fields("out", 3 + fields, fields));
        return List.copyOf(buffers);
    }

    /** One invocation per item: its {@code fields} f32 fields copied to {@code starts[key] + rank}. */
    public static Function permute(int fields) {
        List<Buffer> in = fields("in", 3, fields);
        List<Buffer> out = fields("out", 3 + fields, fields);
        Body b = new Body();
        LocalVar p = b.let("p", new Expr.InvocationId());
        b.when(lt(v(p), new Expr.InvocationCount()), t -> {
            LocalVar destination = t.let("destination", slot(v(p)));
            for (int f = 0; f < fields; f++) {
                t.store(out.get(f), v(destination), load(in.get(f), v(p)));
            }
        });
        return function("sortPermute", b);
    }

    public static final Buffer ORDER_OUT = new Buffer("order", 3, I32);
    public static final List<Buffer> ORDER_BUFFERS = List.of(PERMUTE_KEYS, PERMUTE_RANKS, PERMUTE_STARTS, ORDER_OUT);

    /**
     * One invocation per item: its own index written to {@code starts[key] + rank} of the order. The items stay
     * where they are; the order lists them by key, so key {@code k}'s items are
     * {@code order[start[k]] … order[start[k + 1] − 1]}.
     */
    public static Function order() {
        Body b = new Body();
        LocalVar p = b.let("p", new Expr.InvocationId());
        b.when(lt(v(p), new Expr.InvocationCount()), t -> t.store(ORDER_OUT, slot(v(p)), v(p)));
        return function("sortOrder", b);
    }

    // --- sizes, and building blocks ------------------------------------------------------------------------

    /** The counts' length for keys in {@code [0, keys)}: a word per key, and one more for the total. */
    public static int length(int keys) {
        return keys + 1;
    }

    /** How many blocks — and so block sums — a scan of {@code length} counts has. */
    public static int blocks(int length) {
        return (length + BLOCK - 1) / BLOCK;
    }

    private static Expr slot(Expr item) {
        return add(load(PERMUTE_STARTS, load(PERMUTE_KEYS, item)), load(PERMUTE_RANKS, item));
    }

    /**
     * An inclusive scan of {@code value} across the workgroup, Hillis–Steele: {@link #ROUNDS} rounds, each adding
     * the element {@code 2^r} back. Every invocation reads before any writes, a barrier between, so no round reads
     * a value the same round has already moved on. Leaves the tile holding the inclusive sums.
     */
    private static LocalVar scanTile(Body b, SharedArray tile, LocalVar lid, LocalVar value) {
        b.sharedStore(tile, v(lid), v(value));
        b.barrier();
        for (int round = 0; round < ROUNDS; round++) {
            int offset = 1 << round;
            LocalVar acc = b.let("acc", sharedLoad(tile, v(lid)));
            b.when(not(lt(v(lid), i(offset))), t -> t.set(acc, add(v(acc), sharedLoad(tile, sub(v(lid), i(offset))))));
            b.barrier();
            b.sharedStore(tile, v(lid), v(acc));
            b.barrier();
        }
        return b.let("inclusive", sharedLoad(tile, v(lid)));
    }

    private static List<Buffer> fields(String prefix, int firstBinding, int count) {
        return IntStream.range(0, count).mapToObj(f -> new Buffer(prefix + f, firstBinding + f, F32)).toList();
    }

    private static Function function(String name, Body b) {
        return new Function(name, new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }
}
