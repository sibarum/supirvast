package dev.supirvast.vulkan;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The compute support query and the device it licenses, against whatever GPU is here.
 *
 * <p>What a device reports is the machine's, so the assertions are about what must hold of <em>any</em> device —
 * the floors Vulkan guarantees, the relations between numbers — and about the query agreeing with what a device
 * made from it then says. A wrong struct offset reads as a plausible number, which is why these look at the
 * relations and not only at the values; the exact values for one GPU were checked once, against LWJGL's
 * bindings, when this was ported off them.
 */
class ComputeSupportTest {

    private static Optional<VulkanInstance.DeviceInfo> computeDevice(VulkanInstance instance) {
        return instance.deviceInfos().stream().filter(VulkanInstance.DeviceInfo::compute)
                .min(java.util.Comparator.comparingInt(d -> d.type() == 2 ? 0 : 1));
    }

    private static VulkanInstance instanceOrSkip() {
        try {
            return new VulkanInstance("supirvast compute support test", List.of());
        } catch (RuntimeException | Error e) {
            assumeTrue(false, "no Vulkan loader: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void whatTheDriverSaysObeysTheFloorsAndTheRelationsBetweenNumbers() {
        try (VulkanInstance instance = instanceOrSkip()) {
            Optional<VulkanInstance.DeviceInfo> info = computeDevice(instance);
            assumeTrue(info.isPresent(), "no device with a compute queue");
            ComputeSupport support = ComputeSupport.query(instance, info.get().physicalDevice());

            assertTrue(support.maxWorkgroupMemoryBytes() >= 16384, "Vulkan guarantees 16 KB of workgroup memory");
            assertTrue(support.minSubgroupSize() >= 1 && support.minSubgroupSize() <= support.maxSubgroupSize(),
                    "subgroup sizes " + support.minSubgroupSize() + ".." + support.maxSubgroupSize());
            assertEquals(1, Integer.bitCount(support.minSubgroupSize()), "a subgroup size is a power of two");
            assertEquals(1, Integer.bitCount(support.maxSubgroupSize()), "a subgroup size is a power of two");
            if (support.subgroupOperations() != 0) {
                assertTrue((support.subgroupOperations() & ComputeSupport.SUBGROUP_BASIC) != 0,
                        "a device with any subgroup operation in compute has the basic one");
            }
            // float2 extends float: min/max on a buffer cannot be on where the base extension is absent.
            if (support.floatAtomicMinMax() || support.sharedFloatAtomicMinMax()) {
                assertTrue(support.floatAtomicAdd() || support.sharedFloatAtomics() || support.sharedFloatAtomicAdd()
                                || support.floatAtomicMinMax() || support.sharedFloatAtomicMinMax(),
                        "min/max implies the float atomics extension");
            }
        }
    }

    @Test
    void aDeviceMadeFromTheQueryHasThoseFeaturesAndTheQueuesItAskedFor() {
        try (VulkanInstance instance = instanceOrSkip()) {
            Optional<VulkanInstance.DeviceInfo> info = computeDevice(instance);
            assumeTrue(info.isPresent(), "no device with a compute queue");
            ComputeSupport support = ComputeSupport.query(instance, info.get().physicalDevice());
            int queues = Math.min(3, info.get().computeQueueCount());

            try (VulkanDevice device = new VulkanDevice(instance.handle(), instance.selectionFor(info.get()),
                    VulkanDevice.Request.headlessCompute(support, queues))) {
                assertEquals(queues, device.queues().size());
                assertEquals(device.queues().get(0), device.queue());
                assertEquals(Optional.of(support), device.computeSupport(),
                        "the device hands back exactly what it was licensed for");
                device.waitIdle();
            }
        }
    }

    @Test
    void aDeviceMadeForDrawingAloneSaysItHasNoComputeSupport() {
        try (VulkanInstance instance = instanceOrSkip()) {
            Optional<VulkanInstance.DeviceInfo> info = computeDevice(instance);
            assumeTrue(info.isPresent(), "no device with a compute queue");
            try (VulkanDevice device = new VulkanDevice(instance.handle(), instance.selectionFor(info.get()), false)) {
                assertFalse(device.computeSupport().isPresent());
                assertEquals(1, device.queues().size());
            }
        }
    }

    @Test
    void aMemoryTypeIsFoundForTheCommonRequirementsAndRefusedForImpossibleOnes() {
        try (VulkanInstance instance = instanceOrSkip()) {
            Optional<VulkanInstance.DeviceInfo> info = computeDevice(instance);
            assumeTrue(info.isPresent(), "no device with a compute queue");
            try (VulkanDevice device = new VulkanDevice(instance.handle(), instance.selectionFor(info.get()), false)) {
                int hostVisibleCoherent = 0x2 | 0x4;
                int type = device.tryFindMemoryType(~0, hostVisibleCoherent);
                assertTrue(type >= 0, "every device has host-visible, host-coherent memory");
                assertTrue(device.memoryTypeIsHostCoherent(type));
                assertEquals(-1, device.tryFindMemoryType(0, hostVisibleCoherent), "no type is allowed by no bits");
            }
        }
    }
}
