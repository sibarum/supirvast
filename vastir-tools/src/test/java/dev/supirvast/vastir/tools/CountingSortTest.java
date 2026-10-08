package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.build.Body;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.pass.BufferSpec;
import dev.supirvast.vastir.pass.Buffered;
import dev.supirvast.vastir.pass.CountingSort;
import dev.supirvast.vastir.pass.Pass;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static dev.supirvast.vastir.build.Body.F32;
import static dev.supirvast.vastir.build.Body.I32;
import static dev.supirvast.vastir.build.Body.load;
import static dev.supirvast.vastir.build.Body.lt;
import static dev.supirvast.vastir.build.Body.v;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The counting sort, on each backend, against the host: the starts exactly, every item in its key's run, and a
 * second sort that relies on the first having left the counts at zero. Each backend is judged on its own.
 */
class CountingSortTest {

    /** More keys than one workgroup's worth of blocks, so the block sums take two chunks, with partial ends. */
    private static final int KEYS = 70_001;
    /** Not a multiple of the workgroup, so every per-item pass has a tail. */
    private static final int ITEMS = 5_003;

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void itemsAreSortedByKey(boolean gpu) {
        Program program = new Program();
        try (Accelerator accelerator = gpu ? new Accelerator() : null) {
            if (gpu) {
                assumeTrue(accelerator.capabilities().gpuAvailable(), "no Vulkan device");
            }
            try (PassRunner runner = gpu
                    ? PassRunner.gpu(accelerator, program, CountingSort.BLOCK, PassRunner.NO_SUBGROUP)
                    : PassRunner.cpu(program, CountingSort.BLOCK, PassRunner.NO_SUBGROUP)) {
                runner.clear();
                // Clustered keys, many items sharing one, as cells of a grid have; then spread ones.
                sortAndCheck(runner, program, keys(new Random(1), 300), "first sort");
                sortAndCheck(runner, program, keys(new Random(2), KEYS), "second sort");
            }
        }
    }

    private static int[] keys(Random random, int bound) {
        int[] keys = new int[ITEMS];
        for (int k = 0; k < ITEMS; k++) {
            keys[k] = random.nextInt(bound);
        }
        return keys;
    }

    private static void sortAndCheck(PassRunner runner, Program program, int[] keys, String which) {
        float[] values = new float[ITEMS];
        for (int k = 0; k < ITEMS; k++) {
            values[k] = k + 0.5f;
        }
        runner.write("given", keys);
        runner.write("value", values);
        runner.run(program.sort);

        int[] starts = new int[CountingSort.length(KEYS)];
        for (int key : keys) {
            starts[key + 1]++;
        }
        for (int k = 1; k < starts.length; k++) {
            starts[k] += starts[k - 1];
        }
        assertArrayEquals(starts, runner.read("starts"), which + ": the starts differ from the host's");

        int[] order = runner.read("order");
        float[] sorted = runner.floats("sorted");
        boolean[] seen = new boolean[ITEMS];
        for (int slot = 0; slot < ITEMS; slot++) {
            int item = order[slot];
            assertTrue(item >= 0 && item < ITEMS && !seen[item], which + ": the order is not a permutation at " + slot);
            seen[item] = true;
            int key = keys[item];
            assertTrue(slot >= starts[key] && slot < starts[key + 1], which + ": item " + item + " outside its run");
            assertEquals(values[item], sorted[slot], 0f, which + ": the permute and the order disagree at " + slot);
        }
        assertTrue(Arrays.stream(runner.read("counts")).allMatch(c -> c == 0), which + ": the counts were left");
    }

    /** A key pass that reads each item's key and counts it, the scan, then both the order and a permute. */
    private static final class Program implements Buffered {

        private static final Buffer GIVEN = new Buffer("given", 0, I32);
        private static final Buffer COUNTS = new Buffer("counts", 1, I32);
        private static final Buffer KEYS_OUT = new Buffer("keys", 2, I32);
        private static final Buffer RANKS = new Buffer("ranks", 3, I32);

        final List<Pass> sort;
        private final Map<String, BufferSpec> buffers = new LinkedHashMap<>();

        Program() {
            int length = CountingSort.length(KEYS);
            put("given", I32, ITEMS);
            put("keys", I32, ITEMS);
            put("ranks", I32, ITEMS);
            put("order", I32, ITEMS);
            put("value", F32, ITEMS);
            put("sorted", F32, ITEMS);
            put("counts", I32, length);
            put("starts", I32, length);
            put("sums", I32, CountingSort.blocks(length));

            Body b = new Body();
            LocalVar p = b.let("p", new Expr.InvocationId());
            b.when(lt(v(p), new Expr.InvocationCount()), t ->
                    CountingSort.count(t, v(p), load(GIVEN, v(p)), COUNTS, KEYS_OUT, RANKS));
            Function key = new Function("key", new Type.FunctionType(Type.VOID, List.of()), b.finish());

            List<Pass> passes = new ArrayList<>();
            passes.add(new Pass("key", key, List.of(GIVEN, COUNTS, KEYS_OUT, RANKS),
                    List.of("given", "counts", "keys", "ranks"), ITEMS));
            passes.addAll(CountingSort.scan(length, "counts", "starts", "sums"));
            passes.add(new Pass("order", CountingSort.order(), CountingSort.ORDER_BUFFERS,
                    List.of("keys", "ranks", "starts", "order"), ITEMS));
            passes.add(new Pass("permute", CountingSort.permute(1), CountingSort.permuteBuffers(1),
                    List.of("keys", "ranks", "starts", "value", "sorted"), ITEMS));
            sort = List.copyOf(passes);
        }

        private void put(String name, Type element, int length) {
            buffers.put(name, new BufferSpec(name, element, length));
        }

        @Override
        public Map<String, BufferSpec> buffers() {
            return buffers;
        }
    }
}
