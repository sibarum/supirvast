package dev.supirvast.vastir.preview;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkApplicationInfo;
import org.lwjgl.vulkan.VkAttachmentDescription;
import org.lwjgl.vulkan.VkAttachmentReference;
import org.lwjgl.vulkan.VkClearValue;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkCommandBufferAllocateInfo;
import org.lwjgl.vulkan.VkCommandBufferBeginInfo;
import org.lwjgl.vulkan.VkCommandPoolCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkDeviceCreateInfo;
import org.lwjgl.vulkan.VkDeviceQueueCreateInfo;
import org.lwjgl.vulkan.VkExtent2D;
import org.lwjgl.vulkan.VkFenceCreateInfo;
import org.lwjgl.vulkan.VkFramebufferCreateInfo;
import org.lwjgl.vulkan.VkImageViewCreateInfo;
import org.lwjgl.vulkan.VkInstance;
import org.lwjgl.vulkan.VkInstanceCreateInfo;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkPresentInfoKHR;
import org.lwjgl.vulkan.VkQueue;
import org.lwjgl.vulkan.VkQueueFamilyProperties;
import org.lwjgl.vulkan.VkRect2D;
import org.lwjgl.vulkan.VkRenderPassBeginInfo;
import org.lwjgl.vulkan.VkRenderPassCreateInfo;
import org.lwjgl.vulkan.VkSemaphoreCreateInfo;
import org.lwjgl.vulkan.VkSubmitInfo;
import org.lwjgl.vulkan.VkSubpassDescription;
import org.lwjgl.vulkan.VkSurfaceCapabilitiesKHR;
import org.lwjgl.vulkan.VkSurfaceFormatKHR;
import org.lwjgl.vulkan.VkSwapchainCreateInfoKHR;

import java.nio.IntBuffer;
import java.nio.LongBuffer;

import static org.lwjgl.glfw.GLFW.GLFW_CLIENT_API;
import static org.lwjgl.glfw.GLFW.GLFW_NO_API;
import static org.lwjgl.glfw.GLFW.glfwCreateWindow;
import static org.lwjgl.glfw.GLFW.glfwDestroyWindow;
import static org.lwjgl.glfw.GLFW.glfwInit;
import static org.lwjgl.glfw.GLFW.glfwPollEvents;
import static org.lwjgl.glfw.GLFW.glfwTerminate;
import static org.lwjgl.glfw.GLFW.glfwWindowHint;
import static org.lwjgl.glfw.GLFW.glfwWindowShouldClose;
import static org.lwjgl.glfw.GLFWVulkan.glfwCreateWindowSurface;
import static org.lwjgl.glfw.GLFWVulkan.glfwGetRequiredInstanceExtensions;
import static org.lwjgl.glfw.GLFWVulkan.glfwVulkanSupported;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.vulkan.KHRSurface.*;
import static org.lwjgl.vulkan.KHRSwapchain.*;
import static org.lwjgl.vulkan.VK13.*;

/**
 * The windowed Vulkan previewer. Step 3 milestone: open a GLFW window, stand up a Vulkan 1.3 device with a
 * swapchain, and run a render loop that clears each frame to a solid color and presents it. The shader pipeline
 * and model drawing land in later steps; the CLI already parses the shader/model paths so the surface is stable.
 *
 * <p>Resource ownership and teardown mirror {@code GpuContext} in vastir-tools: a {@code check()} guards every
 * {@code VkResult}, {@link MemoryStack} scopes native allocations, and {@link #close()} destroys in reverse
 * creation order. Windows-only (matching the project's native story); requires a Vulkan 1.3 device.
 */
public final class PreviewApp implements AutoCloseable {

    // A recognizable, non-black clear color (cornflower-ish) so "did anything render" is obvious.
    private static final float[] CLEAR_RGBA = {0.39f, 0.58f, 0.93f, 1.0f};

    private final PreviewOptions options;

    private long window;
    private VkInstance instance;
    private long surface;
    private VkPhysicalDevice physical;
    private VkDevice device;
    private int queueFamily;
    private VkQueue queue;

    private long swapchain;
    private int swapchainFormat;
    private int extentWidth;
    private int extentHeight;
    private long[] imageViews = new long[0];
    private long[] framebuffers = new long[0];
    private long renderPass;
    private long commandPool;
    private VkCommandBuffer[] commandBuffers = new VkCommandBuffer[0];

    private long imageAvailable;
    private long renderFinished;
    private long inFlight;

    public PreviewApp(PreviewOptions options) {
        this.options = options;
    }

    public static void main(String[] args) {
        PreviewOptions options;
        try {
            options = PreviewOptions.parse(args);
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            System.err.println(PreviewOptions.usage());
            System.exit(2);
            return;
        }
        try (PreviewApp app = new PreviewApp(options)) {
            app.run();
        }
    }

    /** Initializes window + Vulkan, runs the render loop until the window closes (or the frame cap is hit). */
    public void run() {
        initWindow();
        initVulkan();
        if (options.screenshot().isPresent()) {
            System.err.println("[preview] --screenshot is not wired yet (lands in step 5); rendering to the window");
        }
        loop();
    }

    private void initWindow() {
        if (!glfwInit()) {
            throw new IllegalStateException("failed to initialize GLFW");
        }
        if (!glfwVulkanSupported()) {
            throw new IllegalStateException("GLFW reports Vulkan is not supported on this machine");
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);   // no OpenGL context; this is a Vulkan window
        window = glfwCreateWindow(options.width(), options.height(), "vastir-preview", NULL, NULL);
        if (window == NULL) {
            throw new IllegalStateException("failed to create the GLFW window");
        }
    }

    private void initVulkan() {
        try (MemoryStack stack = stackPush()) {
            createInstance(stack);
            createSurface(stack);
            pickPhysicalDevice(stack);
            createDevice(stack);
            createSwapchain(stack);
            createRenderPass(stack);
            createFramebuffers(stack);
            createCommandBuffers(stack);
            createSyncObjects(stack);
        }
    }

    private void createInstance(MemoryStack stack) {
        PointerBuffer glfwExtensions = glfwGetRequiredInstanceExtensions();
        if (glfwExtensions == null) {
            throw new IllegalStateException("GLFW could not enumerate the required Vulkan instance extensions");
        }
        VkApplicationInfo app = VkApplicationInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_APPLICATION_INFO)
                .pApplicationName(stack.UTF8("vastir-preview"))
                .apiVersion(VK_API_VERSION_1_3);
        VkInstanceCreateInfo info = VkInstanceCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
                .pApplicationInfo(app)
                .ppEnabledExtensionNames(glfwExtensions);
        PointerBuffer pInstance = stack.mallocPointer(1);
        check(vkCreateInstance(info, null, pInstance), "vkCreateInstance");
        instance = new VkInstance(pInstance.get(0), info);
    }

    private void createSurface(MemoryStack stack) {
        LongBuffer pSurface = stack.mallocLong(1);
        check(glfwCreateWindowSurface(instance, window, null, pSurface), "glfwCreateWindowSurface");
        surface = pSurface.get(0);
    }

    private void pickPhysicalDevice(MemoryStack stack) {
        IntBuffer count = stack.mallocInt(1);
        check(vkEnumeratePhysicalDevices(instance, count, null), "vkEnumeratePhysicalDevices");
        if (count.get(0) == 0) {
            throw new IllegalStateException("no Vulkan physical devices found");
        }
        PointerBuffer devices = stack.mallocPointer(count.get(0));
        check(vkEnumeratePhysicalDevices(instance, count, devices), "vkEnumeratePhysicalDevices");
        for (int i = 0; i < devices.capacity(); i++) {
            VkPhysicalDevice candidate = new VkPhysicalDevice(devices.get(i), instance);
            int family = findGraphicsPresentQueueFamily(candidate, stack);
            if (family >= 0) {
                physical = candidate;
                queueFamily = family;
                return;
            }
        }
        throw new IllegalStateException("no device with a graphics + present queue family for this surface");
    }

    /** A queue family that supports both graphics and presentation to our surface (the common case). */
    private int findGraphicsPresentQueueFamily(VkPhysicalDevice device, MemoryStack stack) {
        IntBuffer count = stack.mallocInt(1);
        vkGetPhysicalDeviceQueueFamilyProperties(device, count, null);
        VkQueueFamilyProperties.Buffer families = VkQueueFamilyProperties.malloc(count.get(0), stack);
        vkGetPhysicalDeviceQueueFamilyProperties(device, count, families);
        IntBuffer presentSupport = stack.mallocInt(1);
        for (int i = 0; i < families.capacity(); i++) {
            boolean graphics = (families.get(i).queueFlags() & VK_QUEUE_GRAPHICS_BIT) != 0;
            check(vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, presentSupport),
                    "vkGetPhysicalDeviceSurfaceSupportKHR");
            if (graphics && presentSupport.get(0) == VK_TRUE) {
                return i;
            }
        }
        return -1;
    }

    private void createDevice(MemoryStack stack) {
        VkDeviceQueueCreateInfo.Buffer queues = VkDeviceQueueCreateInfo.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO)
                .queueFamilyIndex(queueFamily)
                .pQueuePriorities(stack.floats(1.0f));
        VkDeviceCreateInfo info = VkDeviceCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO)
                .pQueueCreateInfos(queues)
                .ppEnabledExtensionNames(stack.pointers(stack.UTF8(VK_KHR_SWAPCHAIN_EXTENSION_NAME)));
        PointerBuffer pDevice = stack.mallocPointer(1);
        check(vkCreateDevice(physical, info, null, pDevice), "vkCreateDevice");
        device = new VkDevice(pDevice.get(0), physical, info);

        PointerBuffer pQueue = stack.mallocPointer(1);
        vkGetDeviceQueue(device, queueFamily, 0, pQueue);
        queue = new VkQueue(pQueue.get(0), device);
    }

    private void createSwapchain(MemoryStack stack) {
        VkSurfaceCapabilitiesKHR caps = VkSurfaceCapabilitiesKHR.malloc(stack);
        check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, caps),
                "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

        VkSurfaceFormatKHR.Buffer formats = surfaceFormats(stack);
        VkSurfaceFormatKHR chosen = chooseSurfaceFormat(formats);
        swapchainFormat = chosen.format();

        chooseExtent(caps);
        int imageCount = caps.minImageCount() + 1;
        if (caps.maxImageCount() > 0 && imageCount > caps.maxImageCount()) {
            imageCount = caps.maxImageCount();
        }

        VkSwapchainCreateInfoKHR info = VkSwapchainCreateInfoKHR.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR)
                .surface(surface)
                .minImageCount(imageCount)
                .imageFormat(swapchainFormat)
                .imageColorSpace(chosen.colorSpace())
                .imageArrayLayers(1)
                // TRANSFER_SRC so a later step can copy a presented image out for a PNG screenshot.
                .imageUsage(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
                .imageSharingMode(VK_SHARING_MODE_EXCLUSIVE)
                .preTransform(caps.currentTransform())
                .compositeAlpha(VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
                .presentMode(VK_PRESENT_MODE_FIFO_KHR)
                .clipped(true)
                .oldSwapchain(VK_NULL_HANDLE);
        info.imageExtent().width(extentWidth).height(extentHeight);

        LongBuffer pSwapchain = stack.mallocLong(1);
        check(vkCreateSwapchainKHR(device, info, null, pSwapchain), "vkCreateSwapchainKHR");
        swapchain = pSwapchain.get(0);

        long[] images = swapchainImages(stack);
        createImageViews(stack, images);
    }

    private VkSurfaceFormatKHR.Buffer surfaceFormats(MemoryStack stack) {
        IntBuffer count = stack.mallocInt(1);
        check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, count, null),
                "vkGetPhysicalDeviceSurfaceFormatsKHR");
        VkSurfaceFormatKHR.Buffer formats = VkSurfaceFormatKHR.malloc(count.get(0), stack);
        check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, count, formats),
                "vkGetPhysicalDeviceSurfaceFormatsKHR");
        return formats;
    }

    /** Prefer 8-bit BGRA UNORM/SRGB; otherwise take whatever the surface offers first. */
    private static VkSurfaceFormatKHR chooseSurfaceFormat(VkSurfaceFormatKHR.Buffer formats) {
        for (int i = 0; i < formats.capacity(); i++) {
            VkSurfaceFormatKHR f = formats.get(i);
            boolean bgra = f.format() == VK_FORMAT_B8G8R8A8_UNORM || f.format() == VK_FORMAT_B8G8R8A8_SRGB;
            if (bgra && f.colorSpace() == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return f;
            }
        }
        return formats.get(0);
    }

    private void chooseExtent(VkSurfaceCapabilitiesKHR caps) {
        VkExtent2D current = caps.currentExtent();
        if (current.width() != 0xFFFFFFFF) {   // the surface dictates the size
            extentWidth = current.width();
            extentHeight = current.height();
            return;
        }
        // Otherwise clamp the requested size to the surface's allowed range.
        extentWidth = clamp(options.width(), caps.minImageExtent().width(), caps.maxImageExtent().width());
        extentHeight = clamp(options.height(), caps.minImageExtent().height(), caps.maxImageExtent().height());
    }

    private long[] swapchainImages(MemoryStack stack) {
        IntBuffer count = stack.mallocInt(1);
        check(vkGetSwapchainImagesKHR(device, swapchain, count, null), "vkGetSwapchainImagesKHR");
        LongBuffer pImages = stack.mallocLong(count.get(0));
        check(vkGetSwapchainImagesKHR(device, swapchain, count, pImages), "vkGetSwapchainImagesKHR");
        long[] images = new long[count.get(0)];
        pImages.get(images);
        return images;
    }

    private void createImageViews(MemoryStack stack, long[] images) {
        imageViews = new long[images.length];
        for (int i = 0; i < images.length; i++) {
            VkImageViewCreateInfo info = VkImageViewCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO)
                    .image(images[i])
                    .viewType(VK_IMAGE_VIEW_TYPE_2D)
                    .format(swapchainFormat);
            info.subresourceRange()
                    .aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                    .baseMipLevel(0).levelCount(1)
                    .baseArrayLayer(0).layerCount(1);
            LongBuffer pView = stack.mallocLong(1);
            check(vkCreateImageView(device, info, null, pView), "vkCreateImageView");
            imageViews[i] = pView.get(0);
        }
    }

    private void createRenderPass(MemoryStack stack) {
        VkAttachmentDescription.Buffer color = VkAttachmentDescription.calloc(1, stack);
        color.get(0)
                .format(swapchainFormat)
                .samples(VK_SAMPLE_COUNT_1_BIT)
                .loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR)
                .storeOp(VK_ATTACHMENT_STORE_OP_STORE)
                .stencilLoadOp(VK_ATTACHMENT_LOAD_OP_DONT_CARE)
                .stencilStoreOp(VK_ATTACHMENT_STORE_OP_DONT_CARE)
                .initialLayout(VK_IMAGE_LAYOUT_UNDEFINED)
                .finalLayout(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

        VkAttachmentReference.Buffer colorRef = VkAttachmentReference.calloc(1, stack)
                .attachment(0)
                .layout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        VkSubpassDescription.Buffer subpass = VkSubpassDescription.calloc(1, stack)
                .pipelineBindPoint(VK_PIPELINE_BIND_POINT_GRAPHICS)
                .colorAttachmentCount(1)
                .pColorAttachments(colorRef);

        VkRenderPassCreateInfo info = VkRenderPassCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO)
                .pAttachments(color)
                .pSubpasses(subpass);
        LongBuffer pRenderPass = stack.mallocLong(1);
        check(vkCreateRenderPass(device, info, null, pRenderPass), "vkCreateRenderPass");
        renderPass = pRenderPass.get(0);
    }

    private void createFramebuffers(MemoryStack stack) {
        framebuffers = new long[imageViews.length];
        for (int i = 0; i < imageViews.length; i++) {
            VkFramebufferCreateInfo info = VkFramebufferCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO)
                    .renderPass(renderPass)
                    .pAttachments(stack.longs(imageViews[i]))
                    .width(extentWidth)
                    .height(extentHeight)
                    .layers(1);
            LongBuffer pFramebuffer = stack.mallocLong(1);
            check(vkCreateFramebuffer(device, info, null, pFramebuffer), "vkCreateFramebuffer");
            framebuffers[i] = pFramebuffer.get(0);
        }
    }

    private void createCommandBuffers(MemoryStack stack) {
        VkCommandPoolCreateInfo poolInfo = VkCommandPoolCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO)
                .queueFamilyIndex(queueFamily);
        LongBuffer pPool = stack.mallocLong(1);
        check(vkCreateCommandPool(device, poolInfo, null, pPool), "vkCreateCommandPool");
        commandPool = pPool.get(0);

        int n = framebuffers.length;
        VkCommandBufferAllocateInfo allocInfo = VkCommandBufferAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO)
                .commandPool(commandPool)
                .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
                .commandBufferCount(n);
        PointerBuffer pBuffers = stack.mallocPointer(n);
        check(vkAllocateCommandBuffers(device, allocInfo, pBuffers), "vkAllocateCommandBuffers");
        commandBuffers = new VkCommandBuffer[n];
        for (int i = 0; i < n; i++) {
            commandBuffers[i] = new VkCommandBuffer(pBuffers.get(i), device);
            recordClear(commandBuffers[i], framebuffers[i], stack);
        }
    }

    /** Records a render pass that only clears the framebuffer — the static content for the step-3 milestone. */
    private void recordClear(VkCommandBuffer cmd, long framebuffer, MemoryStack stack) {
        VkCommandBufferBeginInfo begin = VkCommandBufferBeginInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
        check(vkBeginCommandBuffer(cmd, begin), "vkBeginCommandBuffer");

        VkClearValue.Buffer clear = VkClearValue.calloc(1, stack);
        clear.get(0).color().float32(0, CLEAR_RGBA[0]).float32(1, CLEAR_RGBA[1])
                .float32(2, CLEAR_RGBA[2]).float32(3, CLEAR_RGBA[3]);

        VkRect2D area = VkRect2D.calloc(stack);
        area.offset().set(0, 0);
        area.extent().width(extentWidth).height(extentHeight);

        VkRenderPassBeginInfo rpBegin = VkRenderPassBeginInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO)
                .renderPass(renderPass)
                .framebuffer(framebuffer)
                .renderArea(area)
                .pClearValues(clear);
        vkCmdBeginRenderPass(cmd, rpBegin, VK_SUBPASS_CONTENTS_INLINE);
        // (Pipeline bind + draw land in step 4.)
        vkCmdEndRenderPass(cmd);
        check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
    }

    private void createSyncObjects(MemoryStack stack) {
        VkSemaphoreCreateInfo semInfo = VkSemaphoreCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
        VkFenceCreateInfo fenceInfo = VkFenceCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
                .flags(VK_FENCE_CREATE_SIGNALED_BIT);   // start signaled so the first wait returns immediately
        LongBuffer p = stack.mallocLong(1);
        check(vkCreateSemaphore(device, semInfo, null, p), "vkCreateSemaphore");
        imageAvailable = p.get(0);
        check(vkCreateSemaphore(device, semInfo, null, p), "vkCreateSemaphore");
        renderFinished = p.get(0);
        check(vkCreateFence(device, fenceInfo, null, p), "vkCreateFence");
        inFlight = p.get(0);
    }

    private void loop() {
        int rendered = 0;
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
            rendered++;
            if (options.frames().isPresent() && rendered >= options.frames().get()) {
                break;
            }
        }
        vkDeviceWaitIdle(device);
        System.out.println("[preview] rendered " + rendered + " frame(s)");
    }

    private void drawFrame() {
        try (MemoryStack stack = stackPush()) {
            check(vkWaitForFences(device, stack.longs(inFlight), true, Long.MAX_VALUE), "vkWaitForFences");
            check(vkResetFences(device, stack.longs(inFlight)), "vkResetFences");

            IntBuffer pImageIndex = stack.mallocInt(1);
            check(vkAcquireNextImageKHR(device, swapchain, Long.MAX_VALUE, imageAvailable, VK_NULL_HANDLE,
                    pImageIndex), "vkAcquireNextImageKHR");
            int imageIndex = pImageIndex.get(0);

            VkSubmitInfo submit = VkSubmitInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                    .waitSemaphoreCount(1)
                    .pWaitSemaphores(stack.longs(imageAvailable))
                    .pWaitDstStageMask(stack.ints(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT))
                    .pCommandBuffers(stack.pointers(commandBuffers[imageIndex]))
                    .pSignalSemaphores(stack.longs(renderFinished));
            check(vkQueueSubmit(queue, submit, inFlight), "vkQueueSubmit");

            VkPresentInfoKHR present = VkPresentInfoKHR.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PRESENT_INFO_KHR)
                    .pWaitSemaphores(stack.longs(renderFinished))
                    .swapchainCount(1)
                    .pSwapchains(stack.longs(swapchain))
                    .pImageIndices(stack.ints(imageIndex));
            check(vkQueuePresentKHR(queue, present), "vkQueuePresentKHR");
        }
    }

    @Override
    public void close() {
        if (device != null) {
            vkDeviceWaitIdle(device);
            if (inFlight != VK_NULL_HANDLE) {
                vkDestroyFence(device, inFlight, null);
            }
            if (renderFinished != VK_NULL_HANDLE) {
                vkDestroySemaphore(device, renderFinished, null);
            }
            if (imageAvailable != VK_NULL_HANDLE) {
                vkDestroySemaphore(device, imageAvailable, null);
            }
            if (commandPool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, commandPool, null);
            }
            for (long framebuffer : framebuffers) {
                vkDestroyFramebuffer(device, framebuffer, null);
            }
            if (renderPass != VK_NULL_HANDLE) {
                vkDestroyRenderPass(device, renderPass, null);
            }
            for (long view : imageViews) {
                vkDestroyImageView(device, view, null);
            }
            if (swapchain != VK_NULL_HANDLE) {
                vkDestroySwapchainKHR(device, swapchain, null);
            }
            vkDestroyDevice(device, null);
        }
        if (surface != VK_NULL_HANDLE && instance != null) {
            vkDestroySurfaceKHR(instance, surface, null);
        }
        if (instance != null) {
            vkDestroyInstance(instance, null);
        }
        if (window != NULL) {
            glfwDestroyWindow(window);
        }
        glfwTerminate();
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static void check(int result, String operation) {
        if (result != VK_SUCCESS) {
            throw new IllegalStateException(operation + " failed: VkResult " + result);
        }
    }
}
