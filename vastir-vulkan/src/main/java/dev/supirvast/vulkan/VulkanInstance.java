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
import java.util.List;
import java.util.Optional;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * A live {@code VkInstance}, created through the hand-rolled Vulkan loader ({@link VkLoader}). Owns the instance
 * handle and the instance-level commands VexelRay uses — physical-device enumeration, queue-family and
 * surface-support queries used to pick a device, and surface teardown — and destroys the instance on
 * {@link #close()}.
 */
public final class VulkanInstance implements AutoCloseable {

    private static final int VK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
    private static final int VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;
    private static final int VK_SUCCESS = 0;
    private static final int VK_TRUE = 1;
    private static final int VK_API_VERSION_1_3 = (1 << 22) | (3 << 12);
    private static final int VK_MAX_PHYSICAL_DEVICE_NAME_SIZE = 256;
    private static final int VK_MAX_EXTENSION_NAME_SIZE = 256;
    private static final int VK_MAX_DESCRIPTION_SIZE = 256;
    private static final int VK_UUID_SIZE = 16;

    /** What every Vulkan device guarantees for {@code maxPushConstantsSize}; nothing may report less. */
    public static final int MIN_MAX_PUSH_CONSTANTS_SIZE = 128;

    /**
     * The layer that turns a GPU-side lifetime mistake into a line on stderr. Off by default — it costs real
     * frame time — and switched on with {@code -Dvexelray.vulkan.validation} or {@code VEXELRAY_VULKAN_VALIDATION=1}.
     *
     * <p>Worth reaching for the moment anything looks like memory corruption rather than a wrong picture:
     * destroying an object still in use, or still referenced by an in-flight frame, is undefined behaviour that
     * the driver is under no obligation to report. Without the layer the first observable symptom can be a hung
     * GPU — which on Windows stalls the compositor, so it presents as a frozen desktop rather than as a bug here.
     */
    private static final String VALIDATION_LAYER = "VK_LAYER_KHRONOS_validation";
    private static final int VK_QUEUE_GRAPHICS_BIT = 0x0001;
    private static final int VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2;

    private static final GroupLayout APPLICATION_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pNext"),
            ADDRESS.withName("pApplicationName"),
            JAVA_INT.withName("applicationVersion"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pEngineName"),
            JAVA_INT.withName("engineVersion"),
            JAVA_INT.withName("apiVersion")
    ).withName("VkApplicationInfo");

    private static final GroupLayout INSTANCE_CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pApplicationInfo"),
            JAVA_INT.withName("enabledLayerCount"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("ppEnabledLayerNames"),
            JAVA_INT.withName("enabledExtensionCount"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("ppEnabledExtensionNames")
    ).withName("VkInstanceCreateInfo");

    /**
     * Leading fields of VkPhysicalDeviceProperties, padded past the full struct so Vulkan never writes OOB.
     *
     * <p>Declared as far as {@code limits.maxPushConstantsSize} rather than stopping at the name, because
     * that one number decides which road a scene's parameters take. Everything between is <b>named rather
     * than skipped</b>, so the offset is derived by the layout instead of counted by hand: a miscount here
     * does not crash, it reads a neighbouring field, and a neighbouring field is a plausible-looking number.
     *
     * <p>The four bytes of padding before the limits are not decoration. {@code VkPhysicalDeviceLimits}
     * contains {@code VkDeviceSize} members, so it is eight-aligned, and the UUID leaves the cursor at 292.
     */
    private static final GroupLayout PHYSICAL_DEVICE_PROPERTIES = MemoryLayout.structLayout(
            JAVA_INT.withName("apiVersion"),
            JAVA_INT.withName("driverVersion"),
            JAVA_INT.withName("vendorID"),
            JAVA_INT.withName("deviceID"),
            JAVA_INT.withName("deviceType"),
            MemoryLayout.sequenceLayout(VK_MAX_PHYSICAL_DEVICE_NAME_SIZE, JAVA_BYTE).withName("deviceName"),
            MemoryLayout.sequenceLayout(VK_UUID_SIZE, JAVA_BYTE).withName("pipelineCacheUUID"),
            MemoryLayout.paddingLayout(4),
            // VkPhysicalDeviceLimits, as far as the one field read here.
            JAVA_INT.withName("maxImageDimension1D"),
            JAVA_INT.withName("maxImageDimension2D"),
            JAVA_INT.withName("maxImageDimension3D"),
            JAVA_INT.withName("maxImageDimensionCube"),
            JAVA_INT.withName("maxImageArrayLayers"),
            JAVA_INT.withName("maxTexelBufferElements"),
            JAVA_INT.withName("maxUniformBufferRange"),
            JAVA_INT.withName("maxStorageBufferRange"),
            JAVA_INT.withName("maxPushConstantsSize"),
            JAVA_INT.withName("maxMemoryAllocationCount"),
            JAVA_INT.withName("maxSamplerAllocationCount"),
            // The next two are VkDeviceSize, which wants an 8-byte boundary; the limits so far end 4 short of one.
            MemoryLayout.paddingLayout(4),
            JAVA_LONG.withName("bufferImageGranularity"),
            JAVA_LONG.withName("sparseAddressSpaceSize"),
            // maxBoundDescriptorSets through maxFragmentCombinedOutputResources: 38 limits this file has no use
            // for, between the last VkDeviceSize and the one it reads next.
            MemoryLayout.sequenceLayout(38, JAVA_INT).withName("limitsBetween"),
            JAVA_INT.withName("maxComputeSharedMemorySize"),
            MemoryLayout.paddingLayout(
                    1024 - 20 - VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - VK_UUID_SIZE - 4 - 36 - 8 - 4 - 16 - 38 * 4 - 4)
    ).withName("VkPhysicalDeviceProperties");

    /** VkQueueFamilyProperties — 24 bytes; only queueFlags is read here. */
    private static final GroupLayout QUEUE_FAMILY_PROPERTIES = MemoryLayout.structLayout(
            JAVA_INT.withName("queueFlags"),
            JAVA_INT.withName("queueCount"),
            JAVA_INT.withName("timestampValidBits"),
            JAVA_INT.withName("granularity_width"),
            JAVA_INT.withName("granularity_height"),
            JAVA_INT.withName("granularity_depth")
    ).withName("VkQueueFamilyProperties");

    /** {@code VkLayerProperties} — enough of it to read a layer's name back out. */
    private static final GroupLayout LAYER_PROPERTIES = MemoryLayout.structLayout(
            MemoryLayout.sequenceLayout(VK_MAX_EXTENSION_NAME_SIZE, JAVA_BYTE).withName("layerName"),
            JAVA_INT.withName("specVersion"),
            JAVA_INT.withName("implementationVersion"),
            MemoryLayout.sequenceLayout(VK_MAX_DESCRIPTION_SIZE, JAVA_BYTE).withName("description")
    ).withName("VkLayerProperties");

    private static final long DEVICE_NAME_OFFSET =
            PHYSICAL_DEVICE_PROPERTIES.byteOffset(MemoryLayout.PathElement.groupElement("deviceName"));
    private static final long QFP_STRIDE = QUEUE_FAMILY_PROPERTIES.byteSize();
    private static final long LAYER_NAME_OFFSET =
            LAYER_PROPERTIES.byteOffset(MemoryLayout.PathElement.groupElement("layerName"));
    private static final long LAYER_STRIDE = LAYER_PROPERTIES.byteSize();

    private static final VarHandle AI_sType = Ffi.field(APPLICATION_INFO, "sType");
    private static final VarHandle AI_pApplicationName = Ffi.field(APPLICATION_INFO, "pApplicationName");
    private static final VarHandle AI_apiVersion = Ffi.field(APPLICATION_INFO, "apiVersion");
    private static final VarHandle AI_pEngineName = Ffi.field(APPLICATION_INFO, "pEngineName");

    private static final VarHandle CI_sType = Ffi.field(INSTANCE_CREATE_INFO, "sType");
    private static final VarHandle CI_pNext = Ffi.field(INSTANCE_CREATE_INFO, "pNext");
    private static final VarHandle CI_pApplicationInfo = Ffi.field(INSTANCE_CREATE_INFO, "pApplicationInfo");
    private static final VarHandle CI_enabledExtensionCount = Ffi.field(INSTANCE_CREATE_INFO, "enabledExtensionCount");
    private static final VarHandle CI_ppEnabledExtensionNames = Ffi.field(INSTANCE_CREATE_INFO, "ppEnabledExtensionNames");
    private static final VarHandle CI_enabledLayerCount = Ffi.field(INSTANCE_CREATE_INFO, "enabledLayerCount");
    private static final VarHandle CI_ppEnabledLayerNames = Ffi.field(INSTANCE_CREATE_INFO, "ppEnabledLayerNames");

    private static final VarHandle PDP_deviceType = Ffi.field(PHYSICAL_DEVICE_PROPERTIES, "deviceType");
    private static final VarHandle PDP_maxComputeSharedMemorySize =
            Ffi.field(PHYSICAL_DEVICE_PROPERTIES, "maxComputeSharedMemorySize");
    private static final VarHandle PDP_maxPushConstantsSize =
            Ffi.field(PHYSICAL_DEVICE_PROPERTIES, "maxPushConstantsSize");
    private static final VarHandle QFP_queueFlags = Ffi.field(QUEUE_FAMILY_PROPERTIES, "queueFlags");
    private static final VarHandle QFP_queueCount = Ffi.field(QUEUE_FAMILY_PROPERTIES, "queueCount");
    private static final int VK_QUEUE_COMPUTE_BIT = 0x0002;

    /**
     * A chosen physical device plus a queue family that supports both graphics and presentation to a surface.
     *
     * @param maxPushConstantBytes what this device reports for {@code maxPushConstantsSize}. Carried because
     *                             one decision downstream turns on it — whether a scene's parameters ride
     *                             push constants or a storage buffer — and because a host that can see it can
     *                             report headroom instead of discovering the ceiling by hitting it. It is a
     *                             fact about the machine: nothing that decides what a <em>design</em> may
     *                             contain is allowed to read it, or a design would stop opening on a smaller
     *                             device than it was drawn on
     */
    public record DeviceSelection(MemorySegment physicalDevice, int queueFamilyIndex, String deviceName,
                                  int maxPushConstantBytes) {
    }

    private final MemorySegment handle;
    private final VulkanDebugMessenger debugMessenger;
    private final MethodHandle vkEnumeratePhysicalDevices;
    private final MethodHandle vkGetPhysicalDeviceProperties;
    private final MethodHandle vkGetPhysicalDeviceQueueFamilyProperties;
    /**
     * The two {@code VK_KHR_surface} commands, resolved on first use rather than at construction.
     *
     * <p>They exist only when the extension was enabled, and a headless instance does not enable it: there is
     * no window to present to, and asking for {@code VK_KHR_surface} on a machine with no window system is a
     * way for an offscreen run to fail at instance creation for a capability it was never going to use. Eager
     * resolution made that failure unconditional — {@code vkGetInstanceProcAddr} returns NULL for an
     * unenabled extension's commands, so <em>every</em> extension-less instance died in this constructor,
     * naming a surface function to a caller that had not mentioned surfaces.
     *
     * <p>Lazy, so an instance that never touches a surface never asks for them, and one that does gets the
     * same NULL-check failure at the call that actually needs them. Not volatile: an instance is created and
     * used on one thread (see the runtime's threading contract), and the worst a race could do here is
     * resolve the same handle twice to the same address.
     */
    private MethodHandle vkGetPhysicalDeviceSurfaceSupportKHR;
    private MethodHandle vkDestroySurfaceKHR;
    private final MethodHandle vkDestroyInstance;

    public VulkanInstance(String applicationName, List<String> extensions) {
        MethodHandle vkCreateInstance = VkLoader.globalCommand("vkCreateInstance",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));

        try (Arena temp = Arena.ofConfined()) {
            MemorySegment appInfo = temp.allocate(APPLICATION_INFO);
            AI_sType.set(appInfo, VK_STRUCTURE_TYPE_APPLICATION_INFO);
            AI_pApplicationName.set(appInfo, temp.allocateFrom(applicationName));
            AI_pEngineName.set(appInfo, temp.allocateFrom("VexelRay"));
            AI_apiVersion.set(appInfo, VK_API_VERSION_1_3);

            List<String> layers = requestedLayers();

            // Tied to the *request*, not to whether the layer was found. The extension comes from the loader
            // and the layer from the SDK, so they go missing independently: on a machine with no SDK the layer
            // is absent but the messenger still catches what the loader and driver report, which is a strictly
            // better answer than staying silent because the louder source of messages happened to be missing.
            boolean debug = validationRequested() && VulkanDebugMessenger.available();
            if (validationRequested() && !debug) {
                System.err.println("[vexelray] " + VulkanDebugMessenger.EXTENSION + " is unavailable, so Vulkan "
                        + "diagnostics cannot be captured by the application and go to the loader's own output.");
            }
            List<String> allExtensions = new ArrayList<>(extensions);
            if (debug && !allExtensions.contains(VulkanDebugMessenger.EXTENSION)) {
                allExtensions.add(VulkanDebugMessenger.EXTENSION);
            }

            MemorySegment extArray = temp.allocate(ADDRESS, allExtensions.size());
            for (int i = 0; i < allExtensions.size(); i++) {
                extArray.setAtIndex(ADDRESS, i, temp.allocateFrom(allExtensions.get(i)));
            }

            // Not a ternary at the set() below: VarHandle.set is signature-polymorphic and reads the *static*
            // type of its argument, so a conditional expression arrives as Object and fails at runtime.
            MemorySegment layerArray = MemorySegment.NULL;
            if (!layers.isEmpty()) {
                layerArray = temp.allocate(ADDRESS, layers.size());
                for (int i = 0; i < layers.size(); i++) {
                    layerArray.setAtIndex(ADDRESS, i, temp.allocateFrom(layers.get(i)));
                }
            }

            MemorySegment createInfo = temp.allocate(INSTANCE_CREATE_INFO);
            CI_sType.set(createInfo, VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
            // Chaining a messenger create-info here is the only way to hear about vkCreateInstance and
            // vkDestroyInstance themselves: outside this pNext no messenger object exists at the one moment a
            // bad layer or extension list gets reported. Same struct the real messenger is built from. Not a
            // ternary at the set(), for the signature-polymorphic reason noted above.
            MemorySegment debugInfo = MemorySegment.NULL;
            if (debug) {
                debugInfo = VulkanDebugMessenger.createInfo(temp);
            }
            CI_pNext.set(createInfo, debugInfo);
            CI_pApplicationInfo.set(createInfo, appInfo);
            CI_enabledExtensionCount.set(createInfo, allExtensions.size());
            CI_ppEnabledExtensionNames.set(createInfo, extArray);
            CI_enabledLayerCount.set(createInfo, layers.size());
            CI_ppEnabledLayerNames.set(createInfo, layerArray);

            MemorySegment pInstance = temp.allocate(ADDRESS);
            int result;
            try {
                result = (int) vkCreateInstance.invokeExact(createInfo, MemorySegment.NULL, pInstance);
            } catch (Throwable t) {
                throw NativeException.rethrow("vkCreateInstance", t);
            }
            if (result != VK_SUCCESS) {
                throw new NativeException("vkCreateInstance failed: VkResult " + result);
            }
            this.handle = pInstance.get(ADDRESS, 0);
            this.debugMessenger = debug ? VulkanDebugMessenger.attach(this.handle) : null;
        }

        this.vkEnumeratePhysicalDevices = VkLoader.instanceCommand(handle, "vkEnumeratePhysicalDevices",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
        this.vkGetPhysicalDeviceProperties = VkLoader.instanceCommand(handle, "vkGetPhysicalDeviceProperties",
                FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));
        this.vkGetPhysicalDeviceQueueFamilyProperties = VkLoader.instanceCommand(handle,
                "vkGetPhysicalDeviceQueueFamilyProperties", FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS));
        this.vkDestroyInstance = VkLoader.instanceCommand(handle, "vkDestroyInstance",
                FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));
    }

    /** See the field: resolved here so a headless instance never asks for {@code VK_KHR_surface}. */
    private MethodHandle surfaceSupport() {
        if (vkGetPhysicalDeviceSurfaceSupportKHR == null) {
            vkGetPhysicalDeviceSurfaceSupportKHR = VkLoader.instanceCommand(handle,
                    "vkGetPhysicalDeviceSurfaceSupportKHR",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS));
        }
        return vkGetPhysicalDeviceSurfaceSupportKHR;
    }

    private MethodHandle destroySurface() {
        if (vkDestroySurfaceKHR == null) {
            vkDestroySurfaceKHR = VkLoader.instanceCommand(handle, "vkDestroySurfaceKHR",
                    FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, ADDRESS));
        }
        return vkDestroySurfaceKHR;
    }

    /**
     * The instance layers to enable: the validation layer if it was asked for <em>and</em> is installed, nothing
     * otherwise. A layer named in {@code ppEnabledLayerNames} that the loader cannot find fails
     * {@code vkCreateInstance} outright, so asking is not enough — an absent layer has to degrade to a warning,
     * or every machine without the Vulkan SDK stops being able to start the application at all.
     */
    private static List<String> requestedLayers() {
        if (!validationRequested()) {
            return List.of();
        }
        if (!layerAvailable(VALIDATION_LAYER)) {
            System.err.println("[vexelray] " + VALIDATION_LAYER + " was requested but the loader cannot find it; "
                    + "install the Vulkan SDK to enable it. Continuing without validation.");
            return List.of();
        }
        System.err.println("[vexelray] " + VALIDATION_LAYER + " enabled — expect lower frame rates.");
        return List.of(VALIDATION_LAYER);
    }

    /**
     * Whether an instance created now would actually be validated: the layer was asked for <em>and</em> the
     * loader can find it.
     *
     * <p>Public because a test that brackets its work with {@link VulkanDebugMessenger#errorCount()} needs to
     * know whether that number can move. Without the layer it cannot, so such a test passes while measuring
     * nothing — and a check that cannot fail is worse than an absent one, because it reads in the build log
     * exactly like a check that can. {@code VulkanDebugMessenger.available()} is not the same question: the
     * extension comes from the loader and is present almost everywhere, while the layer ships with the SDK
     * and usually is not.
     */
    public static boolean validationLayerActive() {
        return validationRequested() && layerAvailable(VALIDATION_LAYER);
    }

    /** The system property wins over the environment variable; a bare {@code -D} with no value counts as on. */
    private static boolean validationRequested() {
        String property = System.getProperty("vexelray.vulkan.validation");
        return truthy(property != null ? property : System.getenv("VEXELRAY_VULKAN_VALIDATION"));
    }

    private static boolean truthy(String value) {
        return value != null && (value.isEmpty() || value.equals("1") || Boolean.parseBoolean(value));
    }

    /** Whether the loader can see a layer by this name, asked before it is named in the create info. */
    private static boolean layerAvailable(String name) {
        MethodHandle enumerateLayers = VkLoader.globalCommand("vkEnumerateInstanceLayerProperties",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment pCount = temp.allocate(JAVA_INT);
            if ((int) enumerateLayers.invokeExact(pCount, MemorySegment.NULL) != VK_SUCCESS) {
                return false;
            }
            int count = pCount.get(JAVA_INT, 0);
            if (count == 0) {
                return false;
            }
            MemorySegment properties = temp.allocate(LAYER_PROPERTIES, count);
            if ((int) enumerateLayers.invokeExact(pCount, properties) != VK_SUCCESS) {
                return false;
            }
            // The second call may report fewer layers than the first: trust the count it just wrote, not the
            // count the array was sized from, or the tail is read as layer names out of uninitialised memory.
            count = Math.min(count, pCount.get(JAVA_INT, 0));
            for (int i = 0; i < count; i++) {
                if (name.equals(properties.getString(i * LAYER_STRIDE + LAYER_NAME_OFFSET))) {
                    return true;
                }
            }
            return false;
        } catch (Throwable t) {
            throw NativeException.rethrow("vkEnumerateInstanceLayerProperties", t);
        }
    }

    /** The {@code VkInstance} handle. */
    public MemorySegment handle() {
        return handle;
    }

    /** The raw {@code VkInstance} address — for passing to {@code NativeWindow.createVulkanSurface}. */
    public long handleAddress() {
        return handle.address();
    }

    /** Handles of every physical device visible to this instance. */
    public List<MemorySegment> physicalDevices() {
        List<MemorySegment> devices = new ArrayList<>();
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment pCount = temp.allocate(JAVA_INT);
            check(enumerate(pCount, MemorySegment.NULL), "vkEnumeratePhysicalDevices(count)");
            int count = pCount.get(JAVA_INT, 0);
            if (count == 0) {
                return devices;
            }
            MemorySegment pDevices = temp.allocate(ADDRESS, count);
            check(enumerate(pCount, pDevices), "vkEnumeratePhysicalDevices(list)");
            for (int i = 0; i < count; i++) {
                devices.add(pDevices.getAtIndex(ADDRESS, i));
            }
        }
        return devices;
    }

    /** The reported name of every physical device (convenience over {@link #physicalDevices()}). */
    public List<String> physicalDeviceNames() {
        List<String> names = new ArrayList<>();
        for (MemorySegment device : physicalDevices()) {
            names.add(deviceName(device));
        }
        return names;
    }

    /**
     * Pick a device and a queue family that supports both graphics and presentation to {@code surface}. Prefers a
     * discrete GPU when one qualifies; otherwise the first that does. Empty if none can present to the surface.
     */
    public Optional<DeviceSelection> selectGraphicsPresentDevice(long surface) {
        DeviceSelection firstMatch = null;
        for (MemorySegment device : physicalDevices()) {
            int family = graphicsPresentQueueFamily(device, surface);
            if (family < 0) {
                continue;
            }
            DeviceSelection selection = new DeviceSelection(device, family, deviceName(device),
                    maxPushConstantBytes(device));
            if (deviceType(device) == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                return Optional.of(selection);
            }
            if (firstMatch == null) {
                firstMatch = selection;
            }
        }
        return Optional.ofNullable(firstMatch);
    }

    /**
     * Pick a device and a graphics queue family, without requiring presentation — the headless/offscreen path
     * (no window, no surface). Prefers a discrete GPU. Empty if none exposes a graphics queue.
     */
    public Optional<DeviceSelection> selectGraphicsDevice() {
        DeviceSelection firstMatch = null;
        for (MemorySegment device : physicalDevices()) {
            int family = graphicsQueueFamily(device);
            if (family < 0) {
                continue;
            }
            DeviceSelection selection = new DeviceSelection(device, family, deviceName(device),
                    maxPushConstantBytes(device));
            if (deviceType(device) == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                return Optional.of(selection);
            }
            if (firstMatch == null) {
                firstMatch = selection;
            }
        }
        return Optional.ofNullable(firstMatch);
    }

    /**
     * A physical device as a compute caller sees it: enough to choose one, and to make a device on it.
     *
     * @param index              its position among {@link #physicalDevices()}
     * @param type               {@code VkPhysicalDeviceType}: 1 integrated, 2 discrete, 3 virtual, 4 cpu
     * @param computeQueueFamily the first queue family that can run compute, or -1 if there is none
     * @param computeQueueCount  how many queues that family offers — the ceiling on how many a device may ask for
     */
    public record DeviceInfo(MemorySegment physicalDevice, int index, String name, int type,
                             int computeQueueFamily, int computeQueueCount) {

        public boolean compute() {
            return computeQueueFamily >= 0;
        }
    }

    /**
     * Every physical device, described. Choosing among them is the caller's: which GPU to prefer is a policy,
     * and this layer has no opinion on it.
     */
    public List<DeviceInfo> deviceInfos() {
        List<DeviceInfo> infos = new ArrayList<>();
        List<MemorySegment> devices = physicalDevices();
        for (int i = 0; i < devices.size(); i++) {
            MemorySegment device = devices.get(i);
            int[] family = computeQueueFamily(device);
            infos.add(new DeviceInfo(device, i, deviceName(device), deviceType(device), family[0], family[1]));
        }
        return infos;
    }

    /** The selection for a device and its compute queue family — what {@link VulkanDevice} is made from. */
    public DeviceSelection selectionFor(DeviceInfo info) {
        if (!info.compute()) {
            throw new IllegalArgumentException(info.name() + " has no compute queue family");
        }
        return new DeviceSelection(info.physicalDevice(), info.computeQueueFamily(), info.name(),
                maxPushConstantBytes(info.physicalDevice()));
    }

    /**
     * The device's {@code maxComputeSharedMemorySize}: the most workgroup memory one compute pipeline may
     * declare. 16 KB at least; 32 to 64 KB on current desktop hardware.
     */
    public long maxComputeSharedMemoryBytes(MemorySegment physicalDevice) {
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment props = temp.allocate(PHYSICAL_DEVICE_PROPERTIES);
            properties(physicalDevice, props);
            long reported = Integer.toUnsignedLong((int) PDP_maxComputeSharedMemorySize.get(props));
            // The guaranteed floor, for the reason maxPushConstantBytes states: a number below it means this
            // file's layout is wrong, not that the device is odd, and a wrong offset otherwise reads as plausible.
            if (reported < 16384) {
                throw new NativeException("maxComputeSharedMemorySize read as " + reported + ", below the 16384 "
                        + "every Vulkan device guarantees; suspect this file's VkPhysicalDeviceProperties layout");
            }
            return reported;
        }
    }

    /** Whether queue family {@code family} of {@code physicalDevice} has every bit of {@code flags} ({@code VK_QUEUE_*}). */
    public boolean queueFamilySupports(MemorySegment physicalDevice, int family, int flags) {
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment pCount = temp.allocate(JAVA_INT);
            try {
                vkGetPhysicalDeviceQueueFamilyProperties.invokeExact(physicalDevice, pCount, MemorySegment.NULL);
            } catch (Throwable t) {
                throw NativeException.rethrow("vkGetPhysicalDeviceQueueFamilyProperties", t);
            }
            int count = pCount.get(JAVA_INT, 0);
            if (family < 0 || family >= count) {
                return false;
            }
            MemorySegment props = temp.allocate(QUEUE_FAMILY_PROPERTIES, count);
            try {
                vkGetPhysicalDeviceQueueFamilyProperties.invokeExact(physicalDevice, pCount, props);
            } catch (Throwable t) {
                throw NativeException.rethrow("vkGetPhysicalDeviceQueueFamilyProperties", t);
            }
            int have = (int) QFP_queueFlags.get(props.asSlice(family * QFP_STRIDE, QFP_STRIDE));
            return (have & flags) == flags;
        }
    }

    /** {@code VK_QUEUE_COMPUTE_BIT}, for {@link #queueFamilySupports}. */
    public static final int QUEUE_COMPUTE = VK_QUEUE_COMPUTE_BIT;

    /** {@code {family, queueCount}} of the first queue family with compute, or {@code {-1, 0}}. */
    private int[] computeQueueFamily(MemorySegment device) {
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment pCount = temp.allocate(JAVA_INT);
            try {
                vkGetPhysicalDeviceQueueFamilyProperties.invokeExact(device, pCount, MemorySegment.NULL);
            } catch (Throwable t) {
                throw NativeException.rethrow("vkGetPhysicalDeviceQueueFamilyProperties", t);
            }
            int count = pCount.get(JAVA_INT, 0);
            MemorySegment props = temp.allocate(QUEUE_FAMILY_PROPERTIES, count);
            try {
                vkGetPhysicalDeviceQueueFamilyProperties.invokeExact(device, pCount, props);
            } catch (Throwable t) {
                throw NativeException.rethrow("vkGetPhysicalDeviceQueueFamilyProperties", t);
            }
            for (int f = 0; f < count; f++) {
                MemorySegment family = props.asSlice(f * QFP_STRIDE, QFP_STRIDE);
                if (((int) QFP_queueFlags.get(family) & VK_QUEUE_COMPUTE_BIT) != 0) {
                    return new int[] {f, (int) QFP_queueCount.get(family)};
                }
            }
            return new int[] {-1, 0};
        }
    }

    /** Destroy a surface created for this instance. */
    public void destroySurface(long surface) {
        try {
            destroySurface().invokeExact(handle, surface, MemorySegment.NULL);
        } catch (Throwable t) {
            throw NativeException.rethrow("vkDestroySurfaceKHR", t);
        }
    }

    private int graphicsQueueFamily(MemorySegment device) {
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment pCount = temp.allocate(JAVA_INT);
            try {
                vkGetPhysicalDeviceQueueFamilyProperties.invokeExact(device, pCount, MemorySegment.NULL);
            } catch (Throwable t) {
                throw NativeException.rethrow("vkGetPhysicalDeviceQueueFamilyProperties", t);
            }
            int count = pCount.get(JAVA_INT, 0);
            MemorySegment props = temp.allocate(QUEUE_FAMILY_PROPERTIES, count);
            try {
                vkGetPhysicalDeviceQueueFamilyProperties.invokeExact(device, pCount, props);
            } catch (Throwable t) {
                throw NativeException.rethrow("vkGetPhysicalDeviceQueueFamilyProperties", t);
            }
            for (int f = 0; f < count; f++) {
                int flags = (int) QFP_queueFlags.get(props.asSlice(f * QFP_STRIDE, QFP_STRIDE));
                if ((flags & VK_QUEUE_GRAPHICS_BIT) != 0) {
                    return f;
                }
            }
            return -1;
        }
    }

    private int graphicsPresentQueueFamily(MemorySegment device, long surface) {
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment pCount = temp.allocate(JAVA_INT);
            try {
                vkGetPhysicalDeviceQueueFamilyProperties.invokeExact(device, pCount, MemorySegment.NULL);
            } catch (Throwable t) {
                throw NativeException.rethrow("vkGetPhysicalDeviceQueueFamilyProperties", t);
            }
            int count = pCount.get(JAVA_INT, 0);
            MemorySegment props = temp.allocate(QUEUE_FAMILY_PROPERTIES, count);
            try {
                vkGetPhysicalDeviceQueueFamilyProperties.invokeExact(device, pCount, props);
            } catch (Throwable t) {
                throw NativeException.rethrow("vkGetPhysicalDeviceQueueFamilyProperties", t);
            }
            MemorySegment pSupported = temp.allocate(JAVA_INT);
            for (int f = 0; f < count; f++) {
                int flags = (int) QFP_queueFlags.get(props.asSlice(f * QFP_STRIDE, QFP_STRIDE));
                boolean graphics = (flags & VK_QUEUE_GRAPHICS_BIT) != 0;
                if (graphics && surfaceSupported(device, f, surface, pSupported)) {
                    return f;
                }
            }
            return -1;
        }
    }

    private boolean surfaceSupported(MemorySegment device, int family, long surface, MemorySegment pSupported) {
        try {
            int r = (int) surfaceSupport().invokeExact(device, family, surface, pSupported);
            check(r, "vkGetPhysicalDeviceSurfaceSupportKHR");
            return pSupported.get(JAVA_INT, 0) == VK_TRUE;
        } catch (Throwable t) {
            throw NativeException.rethrow("vkGetPhysicalDeviceSurfaceSupportKHR", t);
        }
    }

    private String deviceName(MemorySegment device) {
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment props = temp.allocate(PHYSICAL_DEVICE_PROPERTIES);
            properties(device, props);
            return props.getString(DEVICE_NAME_OFFSET);
        }
    }

    private int deviceType(MemorySegment device) {
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment props = temp.allocate(PHYSICAL_DEVICE_PROPERTIES);
            properties(device, props);
            return (int) PDP_deviceType.get(props);
        }
    }

    /**
     * What {@code device} reports for {@code maxPushConstantsSize}, in bytes.
     *
     * <p>Refuses to believe anything below the floor every Vulkan device guarantees. That check is not
     * defensive about drivers — it is defensive about <em>this file</em>: the only plausible way to read a
     * number under 128 here is that the struct offset is wrong and a neighbouring field is being read, and a
     * wrong offset otherwise produces a number that looks fine and is not. Falling back to the floor is
     * always safe, because every device has at least that much.
     */
    private int maxPushConstantBytes(MemorySegment device) {
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment props = temp.allocate(PHYSICAL_DEVICE_PROPERTIES);
            properties(device, props);
            int reported = (int) PDP_maxPushConstantsSize.get(props);
            if (reported < MIN_MAX_PUSH_CONSTANTS_SIZE) {
                System.err.println("[vexelray] maxPushConstantsSize read as " + reported + ", below the "
                        + MIN_MAX_PUSH_CONSTANTS_SIZE + " every Vulkan device guarantees; using the floor. "
                        + "Suspect this file's VkPhysicalDeviceProperties layout rather than the driver.");
                return MIN_MAX_PUSH_CONSTANTS_SIZE;
            }
            return reported;
        }
    }

    private void properties(MemorySegment device, MemorySegment props) {
        try {
            vkGetPhysicalDeviceProperties.invokeExact(device, props);
        } catch (Throwable t) {
            throw NativeException.rethrow("vkGetPhysicalDeviceProperties", t);
        }
    }

    private int enumerate(MemorySegment pCount, MemorySegment pDevices) {
        try {
            return (int) vkEnumeratePhysicalDevices.invokeExact(handle, pCount, pDevices);
        } catch (Throwable t) {
            throw NativeException.rethrow("vkEnumeratePhysicalDevices", t);
        }
    }

    private static void check(int result, String call) {
        if (result != VK_SUCCESS) {
            throw new NativeException(call + " failed: VkResult " + result);
        }
    }

    /** The debug messenger — present only when validation and {@code VK_EXT_debug_utils} were both available. */
    public Optional<VulkanDebugMessenger> debugMessenger() {
        return Optional.ofNullable(debugMessenger);
    }

    /**
     * Destroys the messenger before the instance, which is the order the spec requires and the reason the
     * messenger is owned here rather than left to a finalizer. The copy chained into the instance's {@code pNext}
     * outlives it and keeps reporting for the {@code vkDestroyInstance} below.
     */
    @Override
    public void close() {
        if (debugMessenger != null) {
            debugMessenger.close();
        }
        try {
            vkDestroyInstance.invokeExact(handle, MemorySegment.NULL);
        } catch (Throwable t) {
            throw NativeException.rethrow("vkDestroyInstance", t);
        }
    }
}
