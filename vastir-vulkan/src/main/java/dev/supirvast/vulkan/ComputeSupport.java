package dev.supirvast.vulkan;

import dev.supirvast.ffi.Ffi;
import dev.supirvast.ffi.NativeException;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * What a physical device supports among the optional things a compute kernel can need, and the means of
 * switching those on when a device is made.
 *
 * <p>Raw facts only — booleans and numbers as the driver reports them. Which SPIR-V capability a fact licenses
 * is the lowering's business and lives above this module, which names no shader language: whoever compiles
 * kernels maps these onto what it can emit.
 *
 * <p>A device is made with exactly these enabled ({@link VulkanDevice.Request#compute}), because enabling a
 * feature is what licenses the instructions that use it: a kernel emitted with {@code OpCapability Int64} on
 * a device that was not created with {@code shaderInt64} is invalid, and drivers differ in whether they say so.
 * So the same value that was queried is the one that is enabled, and {@link VulkanDevice#computeSupport()}
 * hands it back to whoever shares the device.
 *
 * @param floatAtomicAdd          float32 atomic add in storage buffers
 * @param floatAtomicMinMax       float32 atomic min/max in storage buffers
 * @param sharedFloatAtomics      float32 atomics on workgroup memory
 * @param sharedFloatAtomicAdd    float32 atomic add on workgroup memory
 * @param sharedFloatAtomicMinMax float32 atomic min/max on workgroup memory
 * @param subgroupOperations      the device's {@code VkSubgroupFeatureFlags} for compute; 0 if compute has none
 * @param subgroupSizeControl     size control and full subgroups together, for the compute stage
 * @param maxWorkgroupMemoryBytes the device's {@code maxComputeSharedMemorySize}
 */
public record ComputeSupport(boolean int8, boolean int16, boolean int64, boolean float64,
                             boolean floatAtomicAdd, boolean floatAtomicMinMax,
                             boolean sharedFloatAtomics, boolean sharedFloatAtomicAdd,
                             boolean sharedFloatAtomicMinMax,
                             int subgroupOperations, int minSubgroupSize, int maxSubgroupSize,
                             boolean subgroupSizeControl, long maxWorkgroupMemoryBytes) {

    /** {@code VK_SUBGROUP_FEATURE_*_BIT}, which {@link #subgroupOperations} is a mask of. */
    public static final int SUBGROUP_BASIC = 0x01;
    public static final int SUBGROUP_VOTE = 0x02;
    public static final int SUBGROUP_ARITHMETIC = 0x04;
    public static final int SUBGROUP_SHUFFLE = 0x10;
    public static final int SUBGROUP_SHUFFLE_RELATIVE = 0x20;

    private static final String ATOMIC_FLOAT = "VK_EXT_shader_atomic_float";
    private static final String ATOMIC_FLOAT_2 = "VK_EXT_shader_atomic_float2";

    private static final int STYPE_FEATURES_2 = 1000059000;
    private static final int STYPE_PROPERTIES_2 = 1000059001;
    private static final int STYPE_VULKAN_1_1_PROPERTIES = 50;
    private static final int STYPE_VULKAN_1_2_FEATURES = 51;
    private static final int STYPE_VULKAN_1_3_FEATURES = 53;
    private static final int STYPE_VULKAN_1_3_PROPERTIES = 54;
    private static final int STYPE_ATOMIC_FLOAT_FEATURES = 1000260000;
    private static final int STYPE_ATOMIC_FLOAT_2_FEATURES = 1000273000;

    private static final int SHADER_STAGE_COMPUTE = 0x20;
    private static final int VK_MAX_EXTENSION_NAME_SIZE = 256;

    /**
     * How much to allocate for each struct that goes in a {@code pNext} chain, whatever the layout below it
     * names. The layouts here describe the <em>prefix</em> of each struct that is read or written — the rest
     * is fields this module has no use for — but a driver fills the whole struct when asked and reads the
     * whole struct when told to enable it, so the allocation has to be as big as the real thing, zeroed.
     * The largest of them, {@code VkPhysicalDeviceVulkan13Properties}, is a few hundred bytes.
     */
    private static final long WHOLE_STRUCT = 1024;

    /** {@code sType}, padding, {@code pNext}: the 16 bytes every extensible struct starts with. */
    private static MemoryLayout[] header() {
        return new MemoryLayout[] {JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext")};
    }

    /** A struct header followed by {@code VkBool32}s, in the order the specification declares them. */
    private static GroupLayout bools(String name, String... fields) {
        List<MemoryLayout> members = new ArrayList<>(List.of(header()));
        for (String field : fields) {
            members.add(JAVA_INT.withName(field));
        }
        return MemoryLayout.structLayout(members.toArray(MemoryLayout[]::new)).withName(name);
    }

    /** {@code VkPhysicalDeviceFeatures}, through {@code shaderInt16}: the three fields read are the last of them. */
    private static final GroupLayout FEATURES_2 = MemoryLayout.structLayout(
            header()[0], header()[1], header()[2],
            MemoryLayout.structLayout(
                    JAVA_INT.withName("robustBufferAccess"), JAVA_INT.withName("fullDrawIndexUint32"),
                    JAVA_INT.withName("imageCubeArray"), JAVA_INT.withName("independentBlend"),
                    JAVA_INT.withName("geometryShader"), JAVA_INT.withName("tessellationShader"),
                    JAVA_INT.withName("sampleRateShading"), JAVA_INT.withName("dualSrcBlend"),
                    JAVA_INT.withName("logicOp"), JAVA_INT.withName("multiDrawIndirect"),
                    JAVA_INT.withName("drawIndirectFirstInstance"), JAVA_INT.withName("depthClamp"),
                    JAVA_INT.withName("depthBiasClamp"), JAVA_INT.withName("fillModeNonSolid"),
                    JAVA_INT.withName("depthBounds"), JAVA_INT.withName("wideLines"),
                    JAVA_INT.withName("largePoints"), JAVA_INT.withName("alphaToOne"),
                    JAVA_INT.withName("multiViewport"), JAVA_INT.withName("samplerAnisotropy"),
                    JAVA_INT.withName("textureCompressionETC2"), JAVA_INT.withName("textureCompressionASTC_LDR"),
                    JAVA_INT.withName("textureCompressionBC"), JAVA_INT.withName("occlusionQueryPrecise"),
                    JAVA_INT.withName("pipelineStatisticsQuery"), JAVA_INT.withName("vertexPipelineStoresAndAtomics"),
                    JAVA_INT.withName("fragmentStoresAndAtomics"),
                    JAVA_INT.withName("shaderTessellationAndGeometryPointSize"),
                    JAVA_INT.withName("shaderImageGatherExtended"),
                    JAVA_INT.withName("shaderStorageImageExtendedFormats"),
                    JAVA_INT.withName("shaderStorageImageMultisample"),
                    JAVA_INT.withName("shaderStorageImageReadWithoutFormat"),
                    JAVA_INT.withName("shaderStorageImageWriteWithoutFormat"),
                    JAVA_INT.withName("shaderUniformBufferArrayDynamicIndexing"),
                    JAVA_INT.withName("shaderSampledImageArrayDynamicIndexing"),
                    JAVA_INT.withName("shaderStorageBufferArrayDynamicIndexing"),
                    JAVA_INT.withName("shaderStorageImageArrayDynamicIndexing"),
                    JAVA_INT.withName("shaderClipDistance"), JAVA_INT.withName("shaderCullDistance"),
                    JAVA_INT.withName("shaderFloat64"), JAVA_INT.withName("shaderInt64"),
                    JAVA_INT.withName("shaderInt16")).withName("features")
    ).withName("VkPhysicalDeviceFeatures2");

    /**
     * Through {@code timelineSemaphore}, the 38th of the 1.2 features. Every field up to it is named because the
     * offset is what matters: the two read or written are {@code shaderInt8} and {@code timelineSemaphore}.
     */
    private static final GroupLayout VULKAN_1_2_FEATURES = bools("VkPhysicalDeviceVulkan12Features",
            "samplerMirrorClampToEdge", "drawIndirectCount", "storageBuffer8BitAccess",
            "uniformAndStorageBuffer8BitAccess", "storagePushConstant8", "shaderBufferInt64Atomics",
            "shaderSharedInt64Atomics", "shaderFloat16", "shaderInt8", "descriptorIndexing",
            "shaderInputAttachmentArrayDynamicIndexing", "shaderUniformTexelBufferArrayDynamicIndexing",
            "shaderStorageTexelBufferArrayDynamicIndexing", "shaderUniformBufferArrayNonUniformIndexing",
            "shaderSampledImageArrayNonUniformIndexing", "shaderStorageBufferArrayNonUniformIndexing",
            "shaderStorageImageArrayNonUniformIndexing", "shaderInputAttachmentArrayNonUniformIndexing",
            "shaderUniformTexelBufferArrayNonUniformIndexing", "shaderStorageTexelBufferArrayNonUniformIndexing",
            "descriptorBindingUniformBufferUpdateAfterBind", "descriptorBindingSampledImageUpdateAfterBind",
            "descriptorBindingStorageImageUpdateAfterBind", "descriptorBindingStorageBufferUpdateAfterBind",
            "descriptorBindingUniformTexelBufferUpdateAfterBind", "descriptorBindingStorageTexelBufferUpdateAfterBind",
            "descriptorBindingUpdateUnusedWhilePending", "descriptorBindingPartiallyBound",
            "descriptorBindingVariableDescriptorCount", "runtimeDescriptorArray", "samplerFilterMinmax",
            "scalarBlockLayout", "imagelessFramebuffer", "uniformBufferStandardLayout",
            "shaderSubgroupExtendedTypes", "separateDepthStencilLayouts", "hostQueryReset", "timelineSemaphore");

    /** Through {@code computeFullSubgroups}, the ninth of the 1.3 features. */
    private static final GroupLayout VULKAN_1_3_FEATURES = bools("VkPhysicalDeviceVulkan13Features",
            "robustImageAccess", "inlineUniformBlock", "descriptorBindingInlineUniformBlockUpdateAfterBind",
            "pipelineCreationCacheControl", "privateData", "shaderDemoteToHelperInvocation",
            "shaderTerminateInvocation", "subgroupSizeControl", "computeFullSubgroups");

    private static final GroupLayout ATOMIC_FLOAT_FEATURES = bools("VkPhysicalDeviceShaderAtomicFloatFeaturesEXT",
            "shaderBufferFloat32Atomics", "shaderBufferFloat32AtomicAdd", "shaderBufferFloat64Atomics",
            "shaderBufferFloat64AtomicAdd", "shaderSharedFloat32Atomics", "shaderSharedFloat32AtomicAdd");

    private static final GroupLayout ATOMIC_FLOAT_2_FEATURES = bools("VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT",
            "shaderBufferFloat16Atomics", "shaderBufferFloat16AtomicAdd", "shaderBufferFloat16AtomicMinMax",
            "shaderBufferFloat32AtomicMinMax", "shaderBufferFloat64AtomicMinMax", "shaderSharedFloat16Atomics",
            "shaderSharedFloat16AtomicAdd", "shaderSharedFloat16AtomicMinMax", "shaderSharedFloat32AtomicMinMax");

    /** Through {@code subgroupSupportedOperations}. */
    private static final GroupLayout VULKAN_1_1_PROPERTIES = MemoryLayout.structLayout(
            header()[0], header()[1], header()[2],
            MemoryLayout.sequenceLayout(16, JAVA_BYTE).withName("deviceUUID"),
            MemoryLayout.sequenceLayout(16, JAVA_BYTE).withName("driverUUID"),
            MemoryLayout.sequenceLayout(8, JAVA_BYTE).withName("deviceLUID"),
            JAVA_INT.withName("deviceNodeMask"), JAVA_INT.withName("deviceLUIDValid"),
            JAVA_INT.withName("subgroupSize"), JAVA_INT.withName("subgroupSupportedStages"),
            JAVA_INT.withName("subgroupSupportedOperations")
    ).withName("VkPhysicalDeviceVulkan11Properties");

    /** Through {@code requiredSubgroupSizeStages}. */
    private static final GroupLayout VULKAN_1_3_PROPERTIES = MemoryLayout.structLayout(
            header()[0], header()[1], header()[2],
            JAVA_INT.withName("minSubgroupSize"), JAVA_INT.withName("maxSubgroupSize"),
            JAVA_INT.withName("maxComputeWorkgroupSubgroups"), JAVA_INT.withName("requiredSubgroupSizeStages")
    ).withName("VkPhysicalDeviceVulkan13Properties");

    private static final GroupLayout EXTENSION_PROPERTIES = MemoryLayout.structLayout(
            MemoryLayout.sequenceLayout(VK_MAX_EXTENSION_NAME_SIZE, JAVA_BYTE).withName("extensionName"),
            JAVA_INT.withName("specVersion")
    ).withName("VkExtensionProperties");

    private static final VarHandle F2_pNext = Ffi.field(FEATURES_2, "pNext");
    private static final long F2_FEATURES = FEATURES_2.byteOffset(MemoryLayout.PathElement.groupElement("features"));
    private static final long F_SHADER_FLOAT64 = F2_FEATURES + featureOffset("shaderFloat64");
    private static final long F_SHADER_INT64 = F2_FEATURES + featureOffset("shaderInt64");
    private static final long F_SHADER_INT16 = F2_FEATURES + featureOffset("shaderInt16");

    /** Byte offset of a named {@code VkPhysicalDeviceFeatures} field, from the start of that struct. */
    private static long featureOffset(String field) {
        return ((GroupLayout) FEATURES_2.select(MemoryLayout.PathElement.groupElement("features")))
                .byteOffset(MemoryLayout.PathElement.groupElement(field));
    }

    /**
     * Ask the driver what {@code physicalDevice} supports.
     *
     * <p>A feature struct is chained only when its extension exists: asking a driver about one it does not know
     * is harmless in practice and undefined on paper.
     */
    public static ComputeSupport query(VulkanInstance instance, MemorySegment physicalDevice) {
        MemorySegment root = instance.handle();
        MethodHandle getFeatures2 = VkLoader.instanceCommand(root, "vkGetPhysicalDeviceFeatures2",
                FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));
        MethodHandle getProperties2 = VkLoader.instanceCommand(root, "vkGetPhysicalDeviceProperties2",
                FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));

        Set<String> extensions = deviceExtensions(root, physicalDevice);
        boolean hasFloat = extensions.contains(ATOMIC_FLOAT);
        boolean hasFloat2 = hasFloat && extensions.contains(ATOMIC_FLOAT_2);

        try (Arena temp = Arena.ofConfined()) {
            MemorySegment features12 = whole(temp, STYPE_VULKAN_1_2_FEATURES);
            MemorySegment features13 = whole(temp, STYPE_VULKAN_1_3_FEATURES);
            Ffm.sa(features13, VULKAN_1_3_FEATURES, "pNext", features12);
            MemorySegment chain = features13;
            MemorySegment atomicFloat = MemorySegment.NULL;
            if (hasFloat) {
                atomicFloat = whole(temp, STYPE_ATOMIC_FLOAT_FEATURES);
                Ffm.sa(atomicFloat, ATOMIC_FLOAT_FEATURES, "pNext", chain);
                chain = atomicFloat;
            }
            MemorySegment atomicFloat2 = MemorySegment.NULL;
            if (hasFloat2) {
                atomicFloat2 = whole(temp, STYPE_ATOMIC_FLOAT_2_FEATURES);
                Ffm.sa(atomicFloat2, ATOMIC_FLOAT_2_FEATURES, "pNext", chain);
                chain = atomicFloat2;
            }
            MemorySegment features2 = whole(temp, STYPE_FEATURES_2);
            F2_pNext.set(features2, chain);
            Ffm.invokeVoid(getFeatures2, physicalDevice, features2);

            MemorySegment properties11 = whole(temp, STYPE_VULKAN_1_1_PROPERTIES);
            MemorySegment properties13 = whole(temp, STYPE_VULKAN_1_3_PROPERTIES);
            Ffm.sa(properties13, VULKAN_1_3_PROPERTIES, "pNext", properties11);
            MemorySegment properties2 = whole(temp, STYPE_PROPERTIES_2);
            F2_pNext.set(properties2, properties13);
            Ffm.invokeVoid(getProperties2, physicalDevice, properties2);

            boolean compute = (Ffm.gi(properties11, VULKAN_1_1_PROPERTIES, "subgroupSupportedStages")
                    & SHADER_STAGE_COMPUTE) != 0;
            boolean sizeControl = Ffm.gi(features13, VULKAN_1_3_FEATURES, "subgroupSizeControl") != 0
                    && Ffm.gi(features13, VULKAN_1_3_FEATURES, "computeFullSubgroups") != 0
                    && (Ffm.gi(properties13, VULKAN_1_3_PROPERTIES, "requiredSubgroupSizeStages")
                    & SHADER_STAGE_COMPUTE) != 0;

            return new ComputeSupport(
                    Ffm.gi(features12, VULKAN_1_2_FEATURES, "shaderInt8") != 0,
                    features2.get(JAVA_INT, F_SHADER_INT16) != 0,
                    features2.get(JAVA_INT, F_SHADER_INT64) != 0,
                    features2.get(JAVA_INT, F_SHADER_FLOAT64) != 0,
                    hasFloat && Ffm.gi(atomicFloat, ATOMIC_FLOAT_FEATURES, "shaderBufferFloat32AtomicAdd") != 0,
                    hasFloat2 && Ffm.gi(atomicFloat2, ATOMIC_FLOAT_2_FEATURES, "shaderBufferFloat32AtomicMinMax") != 0,
                    hasFloat && Ffm.gi(atomicFloat, ATOMIC_FLOAT_FEATURES, "shaderSharedFloat32Atomics") != 0,
                    hasFloat && Ffm.gi(atomicFloat, ATOMIC_FLOAT_FEATURES, "shaderSharedFloat32AtomicAdd") != 0,
                    hasFloat2 && Ffm.gi(atomicFloat2, ATOMIC_FLOAT_2_FEATURES, "shaderSharedFloat32AtomicMinMax") != 0,
                    compute ? Ffm.gi(properties11, VULKAN_1_1_PROPERTIES, "subgroupSupportedOperations") : 0,
                    Ffm.gi(properties13, VULKAN_1_3_PROPERTIES, "minSubgroupSize"),
                    Ffm.gi(properties13, VULKAN_1_3_PROPERTIES, "maxSubgroupSize"),
                    sizeControl,
                    instance.maxComputeSharedMemoryBytes(physicalDevice));
        }
    }

    /** What {@link #enable} produces: the head of the {@code pNext} chain, and the device extensions it needs. */
    record Enabled(MemorySegment features2, List<String> extensions) {
    }

    /**
     * Build the {@code VkPhysicalDeviceFeatures2} chain that switches on exactly what this says is supported.
     *
     * <p>The float-atomic extensions are named only when one of their features is on, since enabling a feature
     * licenses what a lowering will then emit. {@code float2} extends {@code float}, so min/max implies the
     * first.
     *
     * @param arena owns the structs; they must outlive the {@code vkCreateDevice} call this is for
     */
    Enabled enable(Arena arena) {
        return enable(arena, false);
    }

    /**
     * As {@link #enable(Arena)}, also switching on {@code timelineSemaphore} when asked. It has to go in this same
     * 1.2 struct: chaining a separate {@code VkPhysicalDeviceTimelineSemaphoreFeatures} beside it is invalid.
     */
    Enabled enable(Arena arena, boolean timelineSemaphore) {
        MemorySegment features12 = whole(arena, STYPE_VULKAN_1_2_FEATURES);
        Ffm.si(features12, VULKAN_1_2_FEATURES, "shaderInt8", int8 ? 1 : 0);
        Ffm.si(features12, VULKAN_1_2_FEATURES, "timelineSemaphore", timelineSemaphore ? 1 : 0);
        // Size control and full subgroups are what let a kernel's subgroups be the ones it was written for.
        MemorySegment features13 = whole(arena, STYPE_VULKAN_1_3_FEATURES);
        Ffm.sa(features13, VULKAN_1_3_FEATURES, "pNext", features12);
        Ffm.si(features13, VULKAN_1_3_FEATURES, "subgroupSizeControl", subgroupSizeControl ? 1 : 0);
        Ffm.si(features13, VULKAN_1_3_FEATURES, "computeFullSubgroups", subgroupSizeControl ? 1 : 0);
        MemorySegment chain = features13;

        List<String> extensions = new ArrayList<>();
        boolean float2 = floatAtomicMinMax || sharedFloatAtomicMinMax;
        if (float2 || floatAtomicAdd || sharedFloatAtomics || sharedFloatAtomicAdd) {
            extensions.add(ATOMIC_FLOAT);
            MemorySegment atomicFloat = whole(arena, STYPE_ATOMIC_FLOAT_FEATURES);
            Ffm.sa(atomicFloat, ATOMIC_FLOAT_FEATURES, "pNext", chain);
            Ffm.si(atomicFloat, ATOMIC_FLOAT_FEATURES, "shaderBufferFloat32AtomicAdd", floatAtomicAdd ? 1 : 0);
            Ffm.si(atomicFloat, ATOMIC_FLOAT_FEATURES, "shaderSharedFloat32Atomics", sharedFloatAtomics ? 1 : 0);
            Ffm.si(atomicFloat, ATOMIC_FLOAT_FEATURES, "shaderSharedFloat32AtomicAdd", sharedFloatAtomicAdd ? 1 : 0);
            chain = atomicFloat;
        }
        if (float2) {
            extensions.add(ATOMIC_FLOAT_2);
            MemorySegment atomicFloat2 = whole(arena, STYPE_ATOMIC_FLOAT_2_FEATURES);
            Ffm.sa(atomicFloat2, ATOMIC_FLOAT_2_FEATURES, "pNext", chain);
            Ffm.si(atomicFloat2, ATOMIC_FLOAT_2_FEATURES, "shaderBufferFloat32AtomicMinMax",
                    floatAtomicMinMax ? 1 : 0);
            Ffm.si(atomicFloat2, ATOMIC_FLOAT_2_FEATURES, "shaderSharedFloat32AtomicMinMax",
                    sharedFloatAtomicMinMax ? 1 : 0);
            chain = atomicFloat2;
        }

        MemorySegment features2 = whole(arena, STYPE_FEATURES_2);
        F2_pNext.set(features2, chain);
        features2.set(JAVA_INT, F_SHADER_INT16, int16 ? 1 : 0);
        features2.set(JAVA_INT, F_SHADER_INT64, int64 ? 1 : 0);
        features2.set(JAVA_INT, F_SHADER_FLOAT64, float64 ? 1 : 0);
        return new Enabled(features2, List.copyOf(extensions));
    }

    /**
     * The features chain for a device that runs no kernels but needs timeline semaphores: a {@code
     * VkPhysicalDeviceFeatures2} with nothing on, over a 1.2 struct with only {@code timelineSemaphore} on.
     */
    static Enabled timelineOnly(Arena arena) {
        MemorySegment features12 = whole(arena, STYPE_VULKAN_1_2_FEATURES);
        Ffm.si(features12, VULKAN_1_2_FEATURES, "timelineSemaphore", 1);
        MemorySegment features2 = whole(arena, STYPE_FEATURES_2);
        F2_pNext.set(features2, features12);
        return new Enabled(features2, List.of());
    }

    /**
     * The adapter LUID of {@code physicalDevice} — the 8 bytes Windows names the adapter by, read as one
     * little-endian long, which is the in-memory form of a Win32 {@code LUID} — or empty when the driver says it is
     * not valid. This is how another API on the same machine (DXGI's {@code EnumAdapterByLuid}) finds the same GPU.
     */
    static java.util.OptionalLong adapterLuid(VulkanInstance instance, MemorySegment physicalDevice) {
        MethodHandle getProperties2 = VkLoader.instanceCommand(instance.handle(), "vkGetPhysicalDeviceProperties2",
                FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment properties11 = whole(temp, STYPE_VULKAN_1_1_PROPERTIES);
            MemorySegment properties2 = whole(temp, STYPE_PROPERTIES_2);
            F2_pNext.set(properties2, properties11);
            Ffm.invokeVoid(getProperties2, physicalDevice, properties2);
            if (Ffm.gi(properties11, VULKAN_1_1_PROPERTIES, "deviceLUIDValid") == 0) {
                return java.util.OptionalLong.empty();
            }
            long offset = VULKAN_1_1_PROPERTIES.byteOffset(MemoryLayout.PathElement.groupElement("deviceLUID"));
            return java.util.OptionalLong.of(properties11.get(java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED,
                    offset));
        }
    }

    /** A zeroed, whole-sized struct with its {@code sType} set. */
    private static MemorySegment whole(Arena arena, int sType) {
        MemorySegment struct = arena.allocate(WHOLE_STRUCT, 8);
        struct.set(JAVA_INT, 0, sType);
        return struct;
    }

    private static Set<String> deviceExtensions(MemorySegment instance, MemorySegment physicalDevice) {
        MethodHandle enumerate = VkLoader.instanceCommand(instance, "vkEnumerateDeviceExtensionProperties",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment pCount = temp.allocate(JAVA_INT);
            Ffm.check(Ffm.invoke(enumerate, physicalDevice, MemorySegment.NULL, pCount, MemorySegment.NULL),
                    "vkEnumerateDeviceExtensionProperties(count)");
            int count = pCount.get(JAVA_INT, 0);
            MemorySegment properties = temp.allocate(EXTENSION_PROPERTIES, Math.max(1, count));
            Ffm.check(Ffm.invoke(enumerate, physicalDevice, MemorySegment.NULL, pCount, properties),
                    "vkEnumerateDeviceExtensionProperties(list)");
            Set<String> names = new HashSet<>();
            long stride = EXTENSION_PROPERTIES.byteSize();
            for (int i = 0; i < pCount.get(JAVA_INT, 0); i++) {
                names.add(properties.getString(i * stride));
            }
            return names;
        }
    }
}
