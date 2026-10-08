package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.build.Body;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.type.Type;
import dev.supirvast.vulkan.ComputeSupport;
import dev.supirvast.vulkan.VulkanDevice;
import dev.supirvast.vulkan.VulkanInstance;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.function.Consumer;

import static dev.supirvast.vastir.build.Body.I32;
import static dev.supirvast.vastir.build.Body.add;
import static dev.supirvast.vastir.build.Body.i;
import static dev.supirvast.vastir.build.Body.load;
import static dev.supirvast.vastir.build.Body.lt;
import static dev.supirvast.vastir.build.Body.mod;
import static dev.supirvast.vastir.build.Body.mul;
import static dev.supirvast.vastir.build.Body.v;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * One device, two queue families: work on a compute-only queue, and work on the device's own queue that reads what
 * it wrote, ordered by a timeline semaphore inside the GPU — every submission made up front, none waited for on the
 * host until the end.
 */
class TwoQueuesTest {

    private static final int N = 1 << 16;
    private static final int LOOPS = 2000;
    private static final int ROUNDS = 4;
    private static final int WORKGROUP = 64;

    @Test
    void aQueueWaitsForAnotherInsideTheGpu() {
        try (VulkanInstance instance = new VulkanInstance("two-queues", List.of())) {
            List<VulkanInstance.DeviceInfo> withComputeOnly = instance.deviceInfos().stream()
                    .filter(d -> d.compute() && instance.computeOnlyQueueFamily(d.physicalDevice()) >= 0)
                    .toList();
            assumeTrue(!withComputeOnly.isEmpty(), "no device with a compute-only queue family");
            for (VulkanInstance.DeviceInfo info : withComputeOnly) {
                onBoth(instance, info);
            }
        }
    }

    /** The rounds on {@code info}, with a queue of its own family and one of its compute-only family. */
    private static void onBoth(VulkanInstance instance, VulkanInstance.DeviceInfo info) {
        int computeOnly = instance.computeOnlyQueueFamily(info.physicalDevice());
        ComputeSupport support = ComputeSupport.query(instance, info.physicalDevice());
        VulkanDevice.Request request = VulkanDevice.Request.headlessCompute(support, 1).withTimelineSemaphore()
                .withQueues(computeOnly, 1, 0.5f);
        try (VulkanDevice device = new VulkanDevice(instance.handle(), instance.selectionFor(info), request);
             GpuContext own = GpuContext.on(instance, device);
             GpuContext async = GpuContext.on(instance, device, computeOnly);
             Accelerator drawing = Accelerator.on(own);
             Accelerator computing = Accelerator.on(async)) {
            assertEquals(computeOnly, async.queueFamily());
            System.out.println("[two queues] " + info.name() + ": family " + own.queueFamily()
                    + " and compute-only family " + computeOnly);

            ResidentBuffer data = computing.allocate(I32, N);
            ResidentBuffer seen = drawing.allocate(I32, N);
            computing.clear(List.of(data));
            KernelHandle stir = computing.register(spec(stir(), N, "data")).orElseThrow();
            KernelHandle copy = drawing.register(spec(plusOne(), N, "data", "seen")).orElseThrow();
            try (DispatchSequence stirring = computing.sequence().dispatch(stir, List.of(data), N).build();
                 DispatchSequence copying = drawing.sequence().dispatch(copy, List.of(data, seen), N).build();
                 GpuContext.Timeline timeline = computing.timeline(0)) {
                assertTrue(stirring.recorded() && copying.recorded(), "a sequence fell back to the CPU");
                // Round r: the stir waits for the last copy (2r) and signals 2r + 1; the copy waits for that
                // and signals 2r + 2. Each queue only ever sees its own submissions in order.
                for (int r = 0; r < ROUNDS; r++) {
                    stirring.run(List.of(timeline.at(2L * r)), List.of(timeline.at(2L * r + 1)));
                    copying.run(List.of(timeline.at(2L * r + 1)), List.of(timeline.at(2L * r + 2)));
                }
                assertTrue(timeline.await(2L * ROUNDS, 30_000_000_000L), "the rounds did not finish");
                assertArrayEquals(expected(ROUNDS), seen.read(), "the copy did not see the stir before it");
            }
        }
    }

    /** Every word stirred {@link #LOOPS} times: long enough that a copy not waiting for it reads it unfinished. */
    private static Function stir() {
        Buffer data = new Buffer("data", 0, I32);
        return kernel(b -> {
            LocalVar k = b.let("k", new Expr.InvocationId());
            b.when(lt(v(k), new Expr.InvocationCount()), t -> {
                LocalVar x = t.let("x", load(data, v(k)));
                LocalVar n = t.let("n", i(0));
                t.loop(lt(v(n), i(LOOPS)), body -> {
                    body.set(x, mod(add(mul(v(x), i(31)), i(7)), i(1_000_003)));
                    body.set(n, add(v(n), i(1)));
                });
                t.store(data, v(k), v(x));
            });
        });
    }

    /** {@code seen[k] = data[k] + 1}. */
    private static Function plusOne() {
        Buffer data = new Buffer("data", 0, I32);
        Buffer seen = new Buffer("seen", 1, I32);
        return kernel(b -> {
            LocalVar k = b.let("k", new Expr.InvocationId());
            b.when(lt(v(k), new Expr.InvocationCount()), t -> t.store(seen, v(k), add(load(data, v(k)), i(1))));
        });
    }

    private static KernelSpec spec(Function kernel, int length, String... columns) {
        List<KernelColumn> list = new java.util.ArrayList<>();
        for (int c = 0; c < columns.length; c++) {
            list.add(KernelColumn.output(columns[c], c, I32).withLength(length));
        }
        return new KernelSpec(kernel, list).withWorkgroupSize(WORKGROUP);
    }

    private static int[] expected(int rounds) {
        int[] data = new int[N];
        for (int r = 0; r < rounds; r++) {
            for (int k = 0; k < N; k++) {
                int x = data[k];
                for (int n = 0; n < LOOPS; n++) {
                    x = (x * 31 + 7) % 1_000_003;
                }
                data[k] = x;
            }
        }
        int[] seen = new int[N];
        for (int k = 0; k < N; k++) {
            seen[k] = data[k] + 1;
        }
        return seen;
    }

    private static Function kernel(Consumer<Body> writer) {
        Body b = new Body();
        writer.accept(b);
        return new Function("kernel", new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }
}
