package dev.supirvast.vulkan;

import dev.supirvast.ffi.Ffi;
import dev.supirvast.ffi.NativeException;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.concurrent.atomic.AtomicLong;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * The sink for {@code VK_LAYER_KHRONOS_validation}. Enabling the layer only makes it <em>produce</em> diagnostics;
 * without a {@code VK_EXT_debug_utils} messenger they go to the loader's own reporting and never reach the
 * process, so a validation error looks exactly like no validation error at all. This is what makes the layer
 * worth switching on: every message it emits arrives here and is printed with its severity and its message id.
 *
 * <p><strong>The callback does not throw.</strong> It is an upcall the driver invokes with native frames on the
 * stack, and an exception thrown through those frames does not unwind into Java — Panama terminates the VM. So
 * the engine-wide "never swallow, always throw" rule cannot apply literally here. The equivalent is to report
 * every message and to <em>count</em> the errors: {@link #errorCount()} is a monotonic tally that a test or a
 * frame loop can assert on at a safe point, which turns a validation error into a build failure without
 * unwinding through the driver. {@link #failOnError()} does exactly that check.
 *
 * <p>The messenger is created twice over, deliberately. A messenger object cannot exist before the instance that
 * owns it, which leaves {@code vkCreateInstance} and {@code vkDestroyInstance} — the two calls most likely to
 * report a bad layer or extension list — unwatched. Chaining an identical create-info into the instance's
 * {@code pNext} covers exactly that window; see {@link #createInfo}.
 */
public final class VulkanDebugMessenger implements AutoCloseable {

    /** {@code VK_EXT_debug_utils} — the extension carrying the messenger; requested only alongside validation. */
    public static final String EXTENSION = "VK_EXT_debug_utils";

    private static final int VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT = 1000128003;
    private static final int VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT = 1000128004;
    private static final int VK_SUCCESS = 0;
    private static final int VK_FALSE = 0;
    private static final int VK_MAX_EXTENSION_NAME_SIZE = 256;

    private static final int SEVERITY_VERBOSE = 0x0001;
    private static final int SEVERITY_INFO = 0x0010;
    private static final int SEVERITY_WARNING = 0x0100;
    private static final int SEVERITY_ERROR = 0x1000;

    private static final int TYPE_GENERAL = 0x1;
    private static final int TYPE_VALIDATION = 0x2;
    private static final int TYPE_PERFORMANCE = 0x4;

    /**
     * Severities asked for by default: the two that mean something is wrong. Info and verbose are left out
     * because they are not diagnostics — the loader alone narrates its whole ICD and layer discovery at info
     * level, tens of lines before the instance even exists, and a log that has to be scrolled past is one that
     * stops being read. Both are available with {@code -Dvexelray.vulkan.validation.verbose}.
     */
    private static final int SEVERITIES = SEVERITY_WARNING | SEVERITY_ERROR;
    private static final int TYPES = TYPE_GENERAL | TYPE_VALIDATION | TYPE_PERFORMANCE;

    private static final GroupLayout CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"),
            JAVA_INT.withName("messageSeverity"),
            JAVA_INT.withName("messageType"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pfnUserCallback"),
            ADDRESS.withName("pUserData")
    ).withName("VkDebugUtilsMessengerCreateInfoEXT");

    /** {@code VkDebugUtilsMessengerCallbackDataEXT} — only the id name and the message text are read. */
    private static final GroupLayout CALLBACK_DATA = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pMessageIdName"),
            JAVA_INT.withName("messageIdNumber"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pMessage"),
            JAVA_INT.withName("queueLabelCount"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pQueueLabels"),
            JAVA_INT.withName("cmdBufLabelCount"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pCmdBufLabels"),
            JAVA_INT.withName("objectCount"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pObjects")
    ).withName("VkDebugUtilsMessengerCallbackDataEXT");

    /** {@code VkExtensionProperties} — enough of it to read an extension's name back out. */
    private static final GroupLayout EXTENSION_PROPERTIES = MemoryLayout.structLayout(
            MemoryLayout.sequenceLayout(VK_MAX_EXTENSION_NAME_SIZE, JAVA_BYTE).withName("extensionName"),
            JAVA_INT.withName("specVersion")
    ).withName("VkExtensionProperties");

    private static final long EXTENSION_NAME_OFFSET =
            EXTENSION_PROPERTIES.byteOffset(MemoryLayout.PathElement.groupElement("extensionName"));
    private static final long EXTENSION_STRIDE = EXTENSION_PROPERTIES.byteSize();

    private static final VarHandle CI_sType = Ffi.field(CREATE_INFO, "sType");
    private static final VarHandle CI_messageSeverity = Ffi.field(CREATE_INFO, "messageSeverity");
    private static final VarHandle CI_messageType = Ffi.field(CREATE_INFO, "messageType");
    private static final VarHandle CI_pfnUserCallback = Ffi.field(CREATE_INFO, "pfnUserCallback");

    private static final VarHandle CD_sType = Ffi.field(CALLBACK_DATA, "sType");
    private static final VarHandle CD_pMessageIdName = Ffi.field(CALLBACK_DATA, "pMessageIdName");
    private static final VarHandle CD_messageIdNumber = Ffi.field(CALLBACK_DATA, "messageIdNumber");
    private static final VarHandle CD_pMessage = Ffi.field(CALLBACK_DATA, "pMessage");

    /**
     * The callback stub. Bound to {@link Ffi#GLOBAL} because the layer may call it from a driver thread at any
     * point before the messenger is destroyed; a confined arena closed at the end of construction would be a
     * use-after-free the first time validation had something to say.
     */
    private static final MemorySegment CALLBACK = Ffi.upcall(MethodHandles.lookup(), VulkanDebugMessenger.class,
            "onMessage", FunctionDescriptor.of(JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS), Ffi.GLOBAL);

    /** Errors seen since the process started. Static because the callback stub must be a static method. */
    private static final AtomicLong ERRORS = new AtomicLong();

    private final MemorySegment instance;
    private final long messenger;
    private final MethodHandle vkDestroyDebugUtilsMessengerEXT;

    private VulkanDebugMessenger(MemorySegment instance, long messenger, MethodHandle destroy) {
        this.instance = instance;
        this.messenger = messenger;
        this.vkDestroyDebugUtilsMessengerEXT = destroy;
    }

    /**
     * Fill {@code out} with the create-info describing this messenger. Public because the same struct is chained
     * into {@code VkInstanceCreateInfo.pNext} to cover instance creation and destruction, when no messenger
     * object can exist yet.
     */
    public static MemorySegment createInfo(Arena arena) {
        MemorySegment info = arena.allocate(CREATE_INFO);
        CI_sType.set(info, VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT);
        CI_messageSeverity.set(info, verbose() ? SEVERITIES | SEVERITY_INFO | SEVERITY_VERBOSE : SEVERITIES);
        CI_messageType.set(info, TYPES);
        CI_pfnUserCallback.set(info, CALLBACK);
        return info;
    }

    /** Create the messenger against a live instance. The caller must have enabled {@link #EXTENSION}. */
    public static VulkanDebugMessenger attach(MemorySegment instance) {
        MethodHandle create = VkLoader.instanceCommand(instance, "vkCreateDebugUtilsMessengerEXT",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
        MethodHandle destroy = VkLoader.instanceCommand(instance, "vkDestroyDebugUtilsMessengerEXT",
                FunctionDescriptor.ofVoid(ADDRESS, java.lang.foreign.ValueLayout.JAVA_LONG, ADDRESS));
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment info = createInfo(temp);
            MemorySegment pMessenger = temp.allocate(java.lang.foreign.ValueLayout.JAVA_LONG);
            int result = (int) create.invokeExact(instance, info, MemorySegment.NULL, pMessenger);
            if (result != VK_SUCCESS) {
                throw new NativeException("vkCreateDebugUtilsMessengerEXT failed: VkResult " + result);
            }
            return new VulkanDebugMessenger(instance, pMessenger.get(java.lang.foreign.ValueLayout.JAVA_LONG, 0),
                    destroy);
        } catch (NativeException e) {
            throw e;
        } catch (Throwable t) {
            throw NativeException.rethrow("vkCreateDebugUtilsMessengerEXT", t);
        }
    }

    /**
     * Whether the loader can see {@link #EXTENSION}. Asked before it is named in the create info: an extension
     * the loader cannot find fails {@code vkCreateInstance} outright, so a machine without the Vulkan SDK must
     * degrade to no messenger rather than to no application.
     */
    public static boolean available() {
        MethodHandle enumerate = VkLoader.globalCommand("vkEnumerateInstanceExtensionProperties",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment pCount = temp.allocate(JAVA_INT);
            if ((int) enumerate.invokeExact(MemorySegment.NULL, pCount, MemorySegment.NULL) != VK_SUCCESS) {
                return false;
            }
            int count = pCount.get(JAVA_INT, 0);
            if (count == 0) {
                return false;
            }
            MemorySegment properties = temp.allocate(EXTENSION_PROPERTIES, count);
            if ((int) enumerate.invokeExact(MemorySegment.NULL, pCount, properties) != VK_SUCCESS) {
                return false;
            }
            // As with layers: the second call may report fewer than the first, and trusting the array's size
            // over the count just written reads names out of uninitialised memory.
            count = Math.min(count, pCount.get(JAVA_INT, 0));
            for (int i = 0; i < count; i++) {
                if (EXTENSION.equals(properties.getString(i * EXTENSION_STRIDE + EXTENSION_NAME_OFFSET))) {
                    return true;
                }
            }
            return false;
        } catch (Throwable t) {
            throw NativeException.rethrow("vkEnumerateInstanceExtensionProperties", t);
        }
    }

    /**
     * The callback the validation layer invokes. Returns {@code VK_FALSE} — the value that means "I have
     * reported this, carry on"; returning true asks the layer to abort the offending call, which the spec
     * reserves for layer development.
     *
     * <p>Every failure path here is caught and reduced to a printed line. That is not the engine's usual
     * never-swallow rule going soft: this frame was entered from the driver, and letting anything at all
     * propagate out of it takes the VM down with no diagnostic. Losing one malformed message is strictly better.
     */
    @SuppressWarnings("unused") // invoked from native through the upcall stub
    private static int onMessage(int severity, int types, MemorySegment callbackData, MemorySegment userData) {
        try {
            if ((severity & SEVERITY_ERROR) != 0) {
                ERRORS.incrementAndGet();
            }
            MemorySegment data = callbackData.reinterpret(CALLBACK_DATA.byteSize());
            String id = readString((MemorySegment) CD_pMessageIdName.get(data));
            int number = (int) CD_messageIdNumber.get(data);
            String message = readString((MemorySegment) CD_pMessage.get(data));
            System.err.println("[vulkan " + severityName(severity) + " " + typeNames(types) + "] "
                    + (id == null ? "(" + number + ")" : id) + ": "
                    + (message == null ? "(no message text)" : message));
        } catch (Throwable t) {
            System.err.println("[vexelray] the Vulkan debug callback itself failed: " + t);
        }
        return VK_FALSE;
    }

    /** Read a {@code const char*} that arrived through an upcall, where every segment is zero-length. */
    private static String readString(MemorySegment pointer) {
        if (pointer == null || pointer.equals(MemorySegment.NULL)) {
            return null;
        }
        return pointer.reinterpret(Long.MAX_VALUE).getString(0);
    }

    private static String severityName(int severity) {
        if ((severity & SEVERITY_ERROR) != 0) {
            return "error";
        }
        if ((severity & SEVERITY_WARNING) != 0) {
            return "warning";
        }
        if ((severity & SEVERITY_INFO) != 0) {
            return "info";
        }
        return "verbose";
    }

    private static String typeNames(int types) {
        StringBuilder out = new StringBuilder();
        if ((types & TYPE_VALIDATION) != 0) {
            out.append("validation");
        }
        if ((types & TYPE_PERFORMANCE) != 0) {
            out.append(out.isEmpty() ? "" : ",").append("performance");
        }
        if ((types & TYPE_GENERAL) != 0) {
            out.append(out.isEmpty() ? "" : ",").append("general");
        }
        return out.isEmpty() ? "unknown" : out.toString();
    }

    private static boolean verbose() {
        String value = System.getProperty("vexelray.vulkan.validation.verbose");
        return value != null && (value.isEmpty() || value.equals("1") || Boolean.parseBoolean(value));
    }

    /**
     * Push a message into the messenger chain as if a layer had raised it, via
     * {@code vkSubmitDebugUtilsMessageEXT}. This is how the plumbing gets proven on a machine with no Vulkan SDK:
     * the struct layouts, the upcall stub, and the {@code const char*} reads are all exercised for real, without
     * needing the validation layer to be installed to generate a message first. Used by the debug-messenger
     * smoke check; harmless in production, but there is no reason to call it there.
     *
     * @param severity one of the {@code VK_DEBUG_UTILS_MESSAGE_SEVERITY_*} bits — see {@link #ERROR}
     */
    public void submit(int severity, String messageIdName, String message) {
        MethodHandle submit = VkLoader.instanceCommand(instance, "vkSubmitDebugUtilsMessageEXT",
                FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, JAVA_INT, ADDRESS));
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment data = temp.allocate(CALLBACK_DATA);
            CD_sType.set(data, VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT);
            CD_pMessageIdName.set(data, temp.allocateFrom(messageIdName));
            CD_messageIdNumber.set(data, 0);
            CD_pMessage.set(data, temp.allocateFrom(message));
            submit.invokeExact(instance, severity, TYPE_VALIDATION, data);
        } catch (Throwable t) {
            throw NativeException.rethrow("vkSubmitDebugUtilsMessageEXT", t);
        }
    }

    /** The error severity bit, for {@link #submit}. */
    public static final int ERROR = SEVERITY_ERROR;

    /** The warning severity bit, for {@link #submit}. */
    public static final int WARNING = SEVERITY_WARNING;

    /** Validation errors reported since the process started. Never resets; compare two readings for a window. */
    public static long errorCount() {
        return ERRORS.get();
    }

    /**
     * Throw if the layer has reported an error since {@code since} — the safe point at which a validation error
     * becomes a failure, called from Java frames rather than from inside the driver callback. A smoke test that
     * brackets its work with {@link #errorCount()} and this call turns a silent GPU-lifetime bug into a red build.
     */
    public static void failOnError(long since) {
        long now = ERRORS.get();
        if (now > since) {
            throw new NativeException((now - since) + " Vulkan validation error(s) were reported; "
                    + "see the [vulkan error ...] lines on stderr");
        }
    }

    /** Errors reported since the process started, as a failure if there are any. */
    public static void failOnError() {
        failOnError(0);
    }

    @Override
    public void close() {
        try {
            vkDestroyDebugUtilsMessengerEXT.invokeExact(instance, messenger, MemorySegment.NULL);
        } catch (Throwable t) {
            throw NativeException.rethrow("vkDestroyDebugUtilsMessengerEXT", t);
        }
    }
}
